"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 
 Base learning framework.
"""

import os
import random
import shutil
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import src.eval
from src.utils.ops import var_cuda, zeros_var_cuda
import src.utils.ops as ops


class LFramework(nn.Module):
    def __init__(self, args, kg, mdl):
        super(LFramework, self).__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.model_dir = args.model_dir
        self.model = args.model

        # Training hyperparameters
        self.batch_size = args.batch_size
        self.train_batch_size = args.train_batch_size
        self.dev_batch_size = args.dev_batch_size
        self.start_epoch = args.start_epoch
        self.num_epochs = args.num_epochs
        self.num_wait_epochs = args.num_wait_epochs
        self.num_peek_epochs = args.num_peek_epochs
        self.learning_rate = args.learning_rate
        self.grad_norm = args.grad_norm
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.optim = None

        self.inference = not args.train
        self.run_analysis = args.run_analysis

        self.kg = kg
        self.mdl = mdl
        print('{} module created'.format(self.model))

    def print_all_model_parameters(self):
        print('\nModel Parameters')
        print('--------------------------')
        for name, param in self.named_parameters():
            print(name, param.numel(), 'requires_grad={}'.format(param.requires_grad))
        param_sizes = [param.numel() for param in self.parameters()]
        print('Total # parameters = {}'.format(sum(param_sizes)))
        print('--------------------------')
        print()

    def linear_decay(self, epochid, r=0.8): # 可以修改0.8
        t = -1 
        if self.args.data_dir == 'umls':
            t = 40
        if self.args.data_dir == 'FB15K-237':
            t = 20 
        decayed_x = self.kg.x.data * 1.0 * (1- epochid / t)# 线性衰减；
        self.kg.x.data = torch.clone(decayed_x)
        decayed_y = self.kg.y.data * 1.0 * (1 -epochid / t)
        self.kg.y.data = torch.clone(decayed_y)
        print('use linear decay')
        return 1.0 * (1- epochid / t)
    def exp_decay(self, epochid, r=0.8): # 可以修改0.8
        t = 1  # 这个t 可以修改；   根据我们的without的曲线探索着来；
        decay_factor = 0.8 ** (epochid / t)  # 指数衰减因子，可以根据需要调整t的值
        decayed_x = self.kg.x.data * decay_factor
        self.kg.x.data = torch.clone(decayed_x)
        decayed_y = self.kg.y.data * decay_factor
        self.kg.y.data = torch.clone(decayed_y)
        print('use exp decay')
        return decay_factor 
    def run_train(self, train_data, dev_data):
        self.print_all_model_parameters()
        if self.optim is None:
            self.optim = optim.Adam(
                filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
        # Track dev metrics changes
        best_dev_metrics = 0
        dev_metrics_history = []
        epoch_loss = []
        for epoch_id in range(self.start_epoch, self.num_epochs):
            print('Epoch {}'.format(epoch_id))
            if epoch_id > 200 and os.path.basename(self.args.data_dir) == 'kinship':
                break
            if epoch_id > 20 and os.path.basename(self.args.data_dir) == 'NELL-995':
                break
            decay_factor = 1
            initial_value = 1.0  # 初始值
            target_value = 0.001 # 目标值
            if os.path.basename(self.args.data_dir) == 'umls':
                num_epochs = 15  # target_value = 0.001
                target_value = 0.001
            if os.path.basename(self.args.data_dir) == 'kinship':
                num_epochs = 2
                target_value = 0.02
            if os.path.basename(self.args.data_dir) == 'WN18RR':
                num_epochs = 3
                target_value = 0.001
            print(os.path.basename(self.args.data_dir))
            if os.path.basename(self.args.data_dir) == 'FB15K-237':
                num_epochs = 10
                target_value = 0.001
            if os.path.basename(self.args.data_dir) == 'NELL-995':
                num_epochs = 6
                target_value = 0.001
            if self.args.use_decay:
                decay_factor = (target_value / initial_value) ** (1 / num_epochs)
                if epoch_id > 0 and self.args.aggregator_type == '14':
                    decay_factor = initial_value * (decay_factor ** epoch_id)
                    inner_e = torch.mul(self.kg.entity_embeddings.weight.data, self.kg.x)
                    inner_r = torch.mul(self.kg.relation_embeddings.weight.data, self.kg.y)
                    w_e = torch.nn.functional.softmax(inner_e, dim=-1)
                    w_r = torch.nn.functional.softmax(inner_r, dim=-1)
                    self.kg.w_e.data = w_e
                    self.kg.w_r.data = w_r
                    self.kg.entity_embeddings.weight.data += (1-self.kg.w_e)*self.kg.x * decay_factor
                    self.kg.relation_embeddings.weight.data += (1-self.kg.w_r)*self.kg.y  * decay_factor

            self.train()
            if self.rl_variation_tag.startswith('rs'):
                self.fn.eval()
                self.fn_kg.eval()
                if self.model.endswith('hypere'):
                    self.fn_secondary_kg.eval()
            self.batch_size = self.train_batch_size
            random.shuffle(train_data)
            batch_losses = []
            entropies = []
            if self.run_analysis:
                rewards = None
                fns = None
            for example_id in tqdm(range(0, len(train_data), self.batch_size)):
                self.optim.zero_grad()

                mini_batch = train_data[example_id:example_id + self.batch_size]
                if len(mini_batch) < self.batch_size:
                    continue  
                # 每个epoch来修改实体和关系的嵌入表征；
                loss = self.loss(mini_batch)
                # decay_factor = None # 不去使用正则化项；
                if self.args.use_regu and self.args.path_encoder == 'USE':
                    (loss['model_loss'] + decay_factor * loss['regular']).backward()  ### 
                else : 
                    loss['model_loss'].backward()
                if self.grad_norm > 0:
                    clip_grad_norm_(self.parameters(), self.grad_norm)

                self.optim.step()

                batch_losses.append(np.array(loss['model_loss'].item()))   ### 
                if 'entropy' in loss:
                    entropies.append(loss['entropy'])
                if self.run_analysis:
                    if rewards is None:
                        rewards = loss['reward']
                    else:
                        rewards = torch.cat([rewards, loss['reward']])
                    if fns is None:
                        fns = loss['fn']
                    else:
                        fns = torch.cat([fns, loss['fn']]) 
            epoch_loss.append(np.mean(batch_losses))
            # Check training statistics
            stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch_id, np.mean(np.array(loss['model_loss'].item())))
            if entropies:
                # stdout_msg += ' entropy = {}'.format(np.mean(entropies))
                pass 
            print(stdout_msg)
            self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id)
            if self.run_analysis:
                print('* Analysis: # path types seen = {}'.format(self.num_path_types))
                num_hits = float(rewards.sum())
                hit_ratio = num_hits / len(rewards)
                print('* Analysis: # hits = {} ({})'.format(num_hits, hit_ratio))
                num_fns = float(fns.sum())
                fn_ratio = num_fns / len(fns)
                print('* Analysis: false negative ratio = {}'.format(fn_ratio))

            # Check dev set performance
            # Check dev set performance
            if epoch_id > 0 and epoch_id % self.num_peek_epochs == 0:
                self.eval()
                self.batch_size = self.dev_batch_size
                with torch.no_grad():
                    dev_scores = self.forward(dev_data, verbose=False)
                print('Dev set performance: (correct evaluation)')
                _, _, _, _, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.dev_objects, verbose=True)
                # metrics = mrr
                print('Dev set performance: (include test set labels)')
                hits_at_1, hits_at_3, hits_at_5, hits_at_10, mrr = src.eval.hits_and_ranks(dev_data, dev_scores, self.kg.all_objects, verbose=True)
                metrics = mrr 
                hits_at_1_file = os.path.join(self.model_dir, 'hits_at_1.dat')
                hits_at_3_file = os.path.join(self.model_dir, 'hits_at_3.dat')
                hits_at_5_file = os.path.join(self.model_dir, 'hits_at_5.dat')
                hits_at_10_file = os.path.join(self.model_dir, 'hits_at_10.dat')
                mrr_file = os.path.join(self.model_dir, 'mrr.dat')
                if epoch_id == 0:
                    with open(hits_at_1_file, 'w') as o_f:
                        o_f.write('{}\n'.format(hits_at_1))
                    with open(hits_at_3_file, 'w') as o_f:
                        o_f.write('{}\n'.format(hits_at_3))
                    with open(hits_at_5_file, 'w') as o_f:
                        o_f.write('{}\n'.format(hits_at_5))
                    with open(hits_at_10_file, 'w') as o_f:
                        o_f.write('{}\n'.format(hits_at_10))
                    with open(mrr_file, 'w') as o_f:
                        o_f.write('{}\n'.format(mrr))
                else:
                    with open(hits_at_1_file, 'a') as o_f:
                        o_f.write('{}\n'.format(hits_at_1))
                    with open(hits_at_3_file, 'a') as o_f:
                        o_f.write('{}\n'.format(hits_at_3))
                    with open(hits_at_5_file, 'a') as o_f:
                        o_f.write('{}\n'.format(hits_at_5))
                    with open(hits_at_10_file, 'a') as o_f:
                        o_f.write('{}\n'.format(hits_at_10))
                    with open(mrr_file, 'a') as o_f:
                        o_f.write('{}\n'.format(mrr))
                # Action dropout anneaking
                if self.model.startswith('point'):
                    eta = self.action_dropout_anneal_interval
                    if len(dev_metrics_history) > eta and metrics < min(dev_metrics_history[-eta:]):
                        old_action_dropout_rate = self.action_dropout_rate
                        # Decrease the action dropout rate once the dev set results stopped increase
                        self.action_dropout_rate *= self.action_dropout_anneal_factor 
                        print('Decreasing action dropout rate: {} -> {}'.format(
                            old_action_dropout_rate, self.action_dropout_rate))
                # Save checkpoint
                if metrics > best_dev_metrics:
                    self.save_checkpoint(checkpoint_id=epoch_id, epoch_id=epoch_id, is_best=True)
                    best_dev_metrics = metrics
                else:
                    # Early stopping
                    if epoch_id >= self.num_wait_epochs and metrics < np.mean(dev_metrics_history[-self.num_wait_epochs:]):
                        print('early stop the training')
                        break 
                dev_metrics_history.append(metrics)
        if self.args.path_encoder == 'USE':
            if self.args.path_encoder_type == "Conv":
                file_path = str(self.args.data_dir) + '_epoch_loss_with_conv_' + str(
                    self.args.number_multihop_valid_paths) + "_" + str(self.args.ratio_valid_invalid) + "_" + str(
                    self.args.aggregator_type) + 'regu' + str(int(self.args.use_regu)) + str(int(self.args.use_decay)) + '.txt'
                file_path_dev = str(self.args.data_dir) + '_dev_mrr_with_conv_'  + str(
                    self.args.number_multihop_valid_paths) + "_" + str(self.args.ratio_valid_invalid) + "_" + str(self.args.aggregator_type) + 'regu'  + str(int(self.args.use_regu)) + str(int(self.args.use_decay)) + '.txt'
            else:
                file_path = str(self.args.data_dir) + '_epoch_loss_with_attn_' + str(
                    self.args.number_multihop_valid_paths) + "_" + str(self.args.ratio_valid_invalid) + "_" + str(
                    self.args.aggregator_type) + 'regu'  + str(int(self.args.use_regu)) + str(int(self.args.use_decay)) + '.txt'
                file_path_dev = str(self.args.data_dir) + '_dev_mrr_with_attn_' + str(
                    self.args.number_multihop_valid_paths) + "_" + str(self.args.ratio_valid_invalid) + "_" + str(
                    self.args.aggregator_type) + 'regu'  + str(int(self.args.use_regu)) + str(int(self.args.use_decay)) + '.txt'
        else:
            file_path = str(self.args.data_dir) + '_epoch_loss_without_' + '.txt'  #   
            file_path_dev = str(self.args.data_dir) + '_dev_mrr_without_'  + '.txt'  #  
        with open(file_path, 'w') as file:
            for loss in epoch_loss:
                file.write(f"{loss}\n")
        with open(file_path_dev, 'w') as file:
            for loss in dev_metrics_history:
                file.write(f"{loss}\n")

        print(f"Epoch loss values saved to {file_path}")
    def forward(self, examples, verbose=False):
        pred_scores = []
        for example_id in tqdm(range(0, len(examples), self.batch_size)):
            mini_batch = examples[example_id:example_id + self.batch_size]
            mini_batch_size = len(mini_batch)
            if len(mini_batch) < self.batch_size:
                self.make_full_batch(mini_batch, self.batch_size)
            pred_score = self.predict(mini_batch, verbose=verbose)
            pred_scores.append(pred_score[:mini_batch_size])
        scores = torch.cat(pred_scores)
        return scores

    def format_batch(self, batch_data, num_labels=-1, num_tiles=1):
        """
        Convert batched tuples to the tensors accepted by the NN.
        """
        def convert_to_binary_multi_subject(e1):
            e1_label = zeros_var_cuda([len(e1), num_labels])
            for i in range(len(e1)):
                e1_label[i][e1[i]] = 1
            return e1_label

        def convert_to_binary_multi_object(e2):
            e2_label = zeros_var_cuda([len(e2), num_labels])
            for i in range(len(e2)):
                e2_label[i][e2[i]] = 1
            return e2_label

        batch_e1, batch_e2, batch_r = [], [], []
        for i in range(len(batch_data)):
            e1, e2, r = batch_data[i]
            batch_e1.append(e1)
            batch_e2.append(e2)
            batch_r.append(r)
        batch_e1 = var_cuda(torch.LongTensor(batch_e1), requires_grad=False)
        batch_r = var_cuda(torch.LongTensor(batch_r), requires_grad=False)
        if type(batch_e2[0]) is list:
            batch_e2 = convert_to_binary_multi_object(batch_e2)
        elif type(batch_e1[0]) is list:
            batch_e1 = convert_to_binary_multi_subject(batch_e1)
        else:
            batch_e2 = var_cuda(torch.LongTensor(batch_e2), requires_grad=False)
        # Rollout multiple times for each example
        if num_tiles > 1:
            batch_e1 = ops.tile_along_beam(batch_e1, num_tiles)
            batch_r = ops.tile_along_beam(batch_r, num_tiles)
            batch_e2 = ops.tile_along_beam(batch_e2, num_tiles)
        return batch_e1, batch_e2, batch_r

    def make_full_batch(self, mini_batch, batch_size, multi_answers=False):
        dummy_e = self.kg.dummy_e
        dummy_r = self.kg.dummy_r
        if multi_answers:
            dummy_example = (dummy_e, [dummy_e], dummy_r)
        else:
            dummy_example = (dummy_e, dummy_e, dummy_r)
        for _ in range(batch_size - len(mini_batch)):
            mini_batch.append(dummy_example)

    def save_checkpoint(self, checkpoint_id, epoch_id=None, is_best=False):
        """
        Save model checkpoint.
        :param checkpoint_id: Model checkpoint index assigned by training loop.
        :param epoch_id: Model epoch index assigned by training loop.
        :param is_best: if set, the model being saved is the best model on dev set.
        """
        checkpoint_dict = dict()
        checkpoint_dict['state_dict'] = self.state_dict()
        checkpoint_dict['epoch_id'] = epoch_id

        out_tar = os.path.join(self.model_dir, 'checkpoint-{}.tar'.format(checkpoint_id))
        if is_best:
            best_path = os.path.join(self.model_dir,   str(int(self.args.use_regu)) + str(int(self.args.use_decay)) + 'model_best.tar')
            # shutil.copyfile(out_tar, best_path)
            torch.save(checkpoint_dict,best_path)
            print('=> best model updated \'{}\''.format(best_path))
        else:
            # torch.save(checkpoint_dict, out_tar)
            print('=> print the current path: \'{}\''.format(out_tar))

    def load_checkpoint(self, input_file):
        """
        Load model checkpoint.
        :param n: Neural network module.
        :param kg: Knowledge graph module.
        :param input_file: Checkpoint file path.
        """
        if os.path.isfile(input_file):
            print('=> loading checkpoint \'{}\''.format(input_file))
            checkpoint = torch.load(input_file, map_location="cuda:{}".format(self.args.gpu))
            self.load_state_dict(checkpoint['state_dict'])
            if not self.inference:
                self.start_epoch = checkpoint['epoch_id'] + 1
                assert (self.start_epoch <= self.num_epochs)
        else:
            print('=> no checkpoint found at \'{}\''.format(input_file))

    def export_to_embedding_projector(self):
        """
        Export knowledge base embeddings into .tsv files accepted by the Tensorflow Embedding Projector.
        """
        vector_path = os.path.join(self.model_dir, 'vector.tsv')
        meta_data_path = os.path.join(self.model_dir, 'metadata.tsv')
        v_o_f = open(vector_path, 'w')
        m_o_f = open(meta_data_path, 'w')
        for r in self.kg.relation2id:
            if r.endswith('_inv'):
                continue
            r_id = self.kg.relation2id[r]
            R = self.kg.relation_embeddings.weight[r_id]
            r_print = ''
            for i in range(len(R)):
                r_print += '{}\t'.format(float(R[i]))
            v_o_f.write('{}\n'.format(r_print.strip()))
            m_o_f.write('{}\n'.format(r))
            print(r, '{}'.format(float(R.norm())))
        v_o_f.close()
        m_o_f.close()
        print('KG embeddings exported to {}'.format(vector_path))
        print('KG meta data exported to {}'.format(meta_data_path))

    @property
    def rl_variation_tag(self):
        parts = self.model.split('.')
        if len(parts) > 1:
            return parts[1]
        else:
            return ''
