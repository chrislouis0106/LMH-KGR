'''
process the data and keep the consistency for two models
use the default nell-995 dataset in the KBGAT model
'''
import collections
import random
import sys

import numpy as np
from pathlib import Path
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as   F
from tqdm import tqdm

EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31

import os, argparse

def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--dataset',type=str, default='kinship')
    parser.add_argument('--entity_dim',type=int,default=200)
    parser.add_argument('--input_drop',type=float,default=0.2)
    parser.add_argument('--feat_dropout_rate',type=float,default=0.2)
    parser.add_argument('--hidden_dropout_rate',type=float,default=0.3)
    parser.add_argument('--emb_2D_d1',type=int,default=10)
    parser.add_argument('--emb_2D_d2',type=int,default=20)
    parser.add_argument('--num_epochs', type=int, default=400,)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--label_smoothing_epsilon', type=float, default=1e-8, help='epsilon used for label smoothing')
    parser.add_argument('--learning_rate', type=float, default=0.0005,)
    parser.add_argument('--action_dropout_anneal_factor', type=float, default=0.9,)
    parser.add_argument('--epoch', type=int, default=200,)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--number_multihop_valid_paths', type=int, default=2)
    parser.add_argument('--xavier_init',type=bool,default=True) 
    parser.add_argument('--ratio_valid_invalid',type=int,default=3) 
    parser.add_argument('--USEGTDM',action='store_true',help='use the ground truth data mask in the process of building the multihop paths')
    parser.add_argument('--NOGTDM',action='store_true',help='no use the ground truth data mask')
    args = parser.parse_args()
    return  args
# 数据集 umls  umls 的 所有conv的模型结束；
# python encoder_conv_2.py --dataset umls --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 3  --cuda  end 很快
# python encoder_conv_2.py --dataset umls --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 3  --cuda end 很快
# python encoder_conv_2.py --dataset umls --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 3  --cuda end ok
# python encoder_conv_2.py --dataset umls --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 1  --cuda  end 
# python encoder_conv_2.py --dataset umls --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 1  --cuda end 
# python encoder_conv_2.py --dataset umls --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 1  --cuda end 
# python encoder_conv_2.py --dataset umls --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 2  --cuda begin end 
# python encoder_conv_2.py --dataset umls --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 2  --cuda begin emd 
# python encoder_conv_2.py --dataset umls --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 2  --cuda begin end 


# python encoder_conv_2.py --dataset umls --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 3  --cuda end 
# python encoder_conv_2.py --dataset umls --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 3  --cuda end 
# python encoder_conv_2.py --dataset umls --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 3  --cuda begin end 
# python encoder_conv_2.py --dataset umls --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 1  --cuda begin end 
# python encoder_conv_2.py --dataset umls --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 1  --cuda begin end 
# python encoder_conv_2.py --dataset umls --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 1  --cuda begin end 
# python encoder_conv_2.py --dataset umls --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 2  --cuda  begin end 
# python encoder_conv_2.py --dataset umls --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 2  --cuda  begin end 
# python encoder_conv_2.py --dataset umls --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 2  --cuda begin end 

# 数据集 kinship
# python encoder_conv_2.py --dataset kinship --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 3  --cuda begin end 
# python encoder_conv_2.py --dataset kinship --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 3  --cuda begin end 
# python encoder_conv_2.py --dataset kinship --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 3  --cuda begin end 
# python encoder_conv_2.py --dataset kinship --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 1  --cuda begin end 
# python encoder_conv_2.py --dataset kinship --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 1  --cuda begin end 
# python encoder_conv_2.py --dataset kinship --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 1  --cuda begin end 
# python encoder_conv_2.py --dataset kinship --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 2  --cuda 0 begin end 
# python encoder_conv_2.py --dataset kinship --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 2  --cuda 0 begin end 
# python encoder_conv_2.py --dataset kinship --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 2  --cuda 0 begin end 


# python encoder_conv_2.py --dataset kinship --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 3  --cuda 0 begin end 
# python encoder_conv_2.py --dataset kinship --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 3  --cuda begin end 
# python encoder_conv_2.py --dataset kinship --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 3  --cuda begin end 
# python encoder_conv_2.py --dataset kinship --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 1  --cuda begin  end 
# python encoder_conv_2.py --dataset kinship --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 1  --cuda begin end 
# python encoder_conv_2.py --dataset kinship --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 1  --cuda begin end 
# python encoder_conv_2.py --dataset kinship --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 2 --cuda   begin end 
# python encoder_conv_2.py --dataset kinship --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 2 --cuda  begin end 
# python encoder_conv_2.py --dataset kinship --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 2  --cuda  begin end 


def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, _ = line.strip().split()
            index[v] = i
            rev_index[i] = v
    return index, rev_index

# 构建模型，设计模型，模型输入；
class PathEncoder_Conv(nn.Module):
    def __init__(self,args,num_entity,num_relation):
        super(PathEncoder_Conv,self).__init__()
        self.entity_emb = nn.Embedding(num_entity, args.entity_dim)
        self.relation_emb = nn.Embedding(num_relation, args.entity_dim)
        self.args = args
        if args.xavier_init:
            torch.nn.init.xavier_normal_(self.entity_emb.weight)
            torch.nn.init.xavier_normal_(self.relation_emb.weight)
        else:
            with open(os.path.join(relative_path,'entity_xavier_vec.txt')) as f:
                self.entity_emb.weight.data = torch.tensor(np.loadtxt(f),dtype=torch.float32)
            with open(os.path.join(relative_path,'relation_xavier_vec.txt')) as f:
                self.relation_emb.weight.data = torch.tensor(np.loadtxt(f),dtype=torch.float32)
        self.input_drop = nn.Dropout(args.input_drop)
        self.hidden_drop = nn.Dropout(args.hidden_dropout_rate)
        self.feature_map_drop = nn.Dropout2d(args.feat_dropout_rate)
        self.W1 = nn.Linear(args.entity_dim * 2, args.entity_dim)
        self.W2 = nn.Linear(args.entity_dim * 2, args.entity_dim)
        self.bn0 = nn.BatchNorm1d(args.entity_dim * 2)   # 是否对输入的数据进行batchnorm
        self.b1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm1d(args.entity_dim)
        self.conv1 = nn.Conv2d(1,32,(3,3),1,0)
        self.fc = nn.Linear(32 * (self.args.emb_2D_d1 * 2 - 2) * (self.args.emb_2D_d2 -2), args.entity_dim)
    def forward(self, path):
        # 输入是一个N * 8 * 6 * 200的嵌入向量表示；其中每一行代表一个三元组中对应的有效路径和无效路径
        # 因此对应的一个输入为一个batch 的第一列； 然后第二列；然后下一列； N  * 6
        # r0_emb : N * 200;
        res = []
        for j in range(path.shape[1]):
            r0_emb = self.relation_emb(path[:,j,0])
            e0_emb = self.entity_emb(path[:,j,1])
            r1_emb = self.relation_emb(path[:,j,2])
            e1_emb = self.entity_emb(path[:,j,3])
            r2_emb = self.relation_emb(path[:,j,4])
            e2_emb = self.entity_emb(path[:,j,5])
            action1 = torch.cat([r0_emb,e0_emb],dim=1) # 3 * 400
            action2 = torch.cat([r1_emb,e1_emb],dim=1)
            action3 = torch.cat([r2_emb,e2_emb],dim=1)
            action1 = self.bn0(action1)
            action2 = self.bn0(action2)
            action3 = self.bn0(action3)
            # 3 * 200
            h1 = self.W1(action1)
            h2 = self.W1(action2)
            h3 = self.W1(action3)
            h1 = self.input_drop(h1)
            h2 = self.input_drop(h2)
            h3 = self.input_drop(h3)
            h1 = self.bn2(h1)
            h2 = self.bn2(h2)
            h3 = self.bn2(h3)
            h1 = h1.view(-1,1,self.args.emb_2D_d1, self.args.emb_2D_d2)
            h2 = h2.view(-1,1,self.args.emb_2D_d1, self.args.emb_2D_d2) # N * 1 * 10 * 20
            stack_inputs = torch.cat([h1,h2],dim=2)
            stack_inputs = self.b1(stack_inputs)
            x = self.conv1(stack_inputs)
            # X = self.bn1(X)
            x = F.relu(x)
            # 特征层0.3
            x = self.feature_map_drop(x)  #  N * 32 * 18 * 18 ;
            x = x.view(x.shape[0], -1)
            x = self.fc(x)
            # 隐藏层 0.2 ；
            x = self.hidden_drop(x)
            x = self.bn2(x)
            x = F.relu(x)
            # N * 200; N * 200 -> N * 1 * 200; N * 200 * 1 -> N * 1;
            x = torch.matmul(x.unsqueeze(1), h3.unsqueeze(2)).squeeze(2)
            # print(x.shape)  N * 1 ;
            res.append(torch.sigmoid(x))
        # print(len(res),res[0].shape)
        res = torch.cat(res,dim=1) # N * 8 ;
        return res

if __name__ == '__main__':
    args = define_args()
    if args.dataset == "kinship" or args.dataset == "umls" or args.dataset == "FB15K-237":
        bandwidth = 400
    elif args.dataset == 'NELL-995':
        bandwidth = 256
    elif args.dataset == 'WN18RR':
        bandwidth = 500 
    else:
        print('dataset is not exist')
    path = os.getcwd()
    relative_path = os.path.join(path, 'data', args.dataset)
    entity2id, id2entity = load_index(os.path.join(relative_path, 'entity2id.txt'))
    relation2id, id2relation = load_index(os.path.join(relative_path, 'relation2id.txt'))
    page_rank_path = os.path.join(relative_path, 'raw.pgrk')
    train_data_path = os.path.join(relative_path, 'train.triples')
    adj_list_path = os.path.join(relative_path, 'adj_list.pkl')
    r_no_op_id = relation2id['NO_OP_RELATION']
    dummy_relation_id = relation2id['DUMMY_RELATION']
    start_relation_id = relation2id['START_RELATION']
    dummy_entity_id = entity2id['DUMMY_ENTITY']

    train_triples = []
    train_triples_es_et = set()
    with open(adj_list_path, 'rb') as f: 
        adj_list = pickle.load(f)
        
    with open(train_data_path) as f:
        for l in f.readlines():
            es, et, rq = l.strip().split()
            train_triples.append([es, et, rq])
            train_triples_es_et.add((entity2id[es], entity2id[et]))

    # 每一行的数据样例：/m/02_j1w                      :0.0036551696698135204
    def load_page_rank_scores(input_path, entity2id):
        pgrk_scores = collections.defaultdict(float)
        with open(input_path) as f:
            for line in f:
                e, score = line.strip().split(':')
                e_id = entity2id[e.strip()]
                score = float(score)
                pgrk_scores[e_id] = score
        # 因此此时返回的是以实体id为key； 以pagerank打分为score的键值对
        return pgrk_scores
    page_rank_scores = load_page_rank_scores(page_rank_path, entity2id)
    def get_action_space(e1):
        action_space = []
        if e1 in adj_list:
            for r in adj_list[e1]:
                targets = adj_list[e1][r]
                for e2 in targets:
                    action_space.append((r, e2))
            if len(action_space) + 1 >= bandwidth:
                sorted_action_space = sorted(action_space, key=lambda x: page_rank_scores[x[1]], reverse=True)
                action_space = sorted_action_space[:bandwidth] 
        if len(action_space) > 0 :
            temp = random.randrange(len(action_space))
        else :
            temp = 0 
        action_space.insert(temp, (r_no_op_id, e1))
        return action_space
    

    # 根据邻接表构建对应实体的动作空间并裁剪 
    action_space_list = collections.defaultdict(dict)
    for e1 in range(len(entity2id)): 
        action_space_list[e1] = get_action_space(e1)

    path = {}
    path_valid = {}
    path_invalid = {}
    valid_path_number = {}
    invalid_path_number = {}
    for i, l in tqdm(enumerate(train_triples)):
        es, et, rq = l[0], l[1], l[2]
        es_id, et_id, rq_id = entity2id[es], entity2id[et], relation2id[rq]
        rq_inv_id = relation2id[rq+'_inv']
        path[i] = list()
        path_valid[i] = list()
        path_invalid[i] = list()
        m,n = 0, 0 
        S = [(es_id,rq_id,et_id), (et_id,rq_inv_id,es_id)]
        for action1 in action_space_list[es_id]: 
            for action2  in action_space_list[action1[1]]: 
                if ( (es_id,action1[0],action1[1])  in S or (action1[1],action2[0],action2[1]) in S ):
                    continue 
                if action2[1] == et_id :
                    path_valid[i].append([start_relation_id, es_id, action1[0],action1[1],action2[0],action2[1]])
                    m += 1 
                if action2[1] != et_id and (es_id, action2[1])  not in train_triples_es_et: 
                    path_invalid[i].append([start_relation_id, es_id, action1[0],action1[1],action2[0],action2[1]])
                    n += 1
                if m >= args.number_multihop_valid_paths  and n >= args.number_multihop_valid_paths *  args.ratio_valid_invalid -1 : 
                    break
            if m >= args.number_multihop_valid_paths  and  n >= args.number_multihop_valid_paths *  args.ratio_valid_invalid - 1: 
                break 
        if (len(path_valid[i]) == 0 or len(path_invalid[i]) == 0 ):
            del path_valid[i] 
            del path_invalid[i]
            del path[i]
            continue
        temp = []
        while (len(temp) < args.number_multihop_valid_paths):
            random_number = random.choice(range(len(path_valid[i])))
            temp.append(path_valid[i][random_number])
        path_valid[i] = temp 
        path_invalid[i].append([start_relation_id, es_id, dummy_relation_id, dummy_entity_id, dummy_relation_id, dummy_entity_id])
        temp = []
        while (len(temp) < args.number_multihop_valid_paths * args.ratio_valid_invalid):
            random_number = random.choice(range(len(path_invalid[i])))
            temp.append(path_invalid[i][random_number]) 
        path_invalid[i] = temp 
        path[i] = path_valid[i] + path_invalid[i]
    print('the multihop triple nubmer is {}, all triple number is {}'.format(len(path), len(train_triples)))

    train_data = torch.tensor([i for i in path.values()],dtype= torch.long)
    label_valid = torch.tensor([1] * args.number_multihop_valid_paths, dtype = torch.float32)
    label_invalid = torch.tensor([0] * args.number_multihop_valid_paths * args.ratio_valid_invalid, dtype = torch.float32)
    label = torch.cat((label_valid, label_invalid, ))
    label_all = label.unsqueeze(0).repeat(train_data.shape[0],1)
    print(label_all.shape, train_data.shape)
    torch.cuda.set_device(args.cuda) 
    if args.USEGTDM:
        model_save_path=os.path.join('model',args.dataset+'_'+'USEGTDM_'+str(args.number_multihop_valid_paths)+"_random_"+str(args.ratio_valid_invalid)+"_Conv_newpath"+'.tar')
    if args.NOGTDM:
        model_save_path=os.path.join('model',args.dataset+'_'+'NOGTDM_'+str(args.number_multihop_valid_paths)+"_random_"+str(args.ratio_valid_invalid)+"_Conv_newpath"+'.tar')
    print('model_save_path:', model_save_path)
    model = PathEncoder_Conv(args, len(entity2id), len(relation2id))
    model.cuda()
    criterion = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt,step_size=20,gamma=0.8,last_epoch=-1)
    loss_epoch = []
    best_metric =  np.inf
    for epoch in range(args.epoch):
        # 每个epoch 都 shuffle 数据；
        loss_batch = []
        model.train()
        idx = torch.randperm(train_data.shape[0])
        train_data = train_data[idx]
        for index in tqdm(range(0, len(train_data), args.batch_size)):
            opt.zero_grad()
            mini_batch = train_data[index:index + args.batch_size,:,:]
            mini_label = label_all[index:index+args.batch_size,:,]
            if len(mini_batch) < args.batch_size:
                continue
            mini_label = ((1 - args.label_smoothing_epsilon) * mini_label) + (1.0 / 1000)
            mini_label = mini_label.cuda()
            mini_batch = mini_batch.cuda()
            pred = model(mini_batch)
            loss_value_batch  = criterion(pred, mini_label)
            loss_batch.append(loss_value_batch.data.cpu().numpy())
            loss_value_batch.backward()
            opt.step()
        scheduler.step()
        stdout_msg = 'Epoch {}: average training loss = {}'.format(epoch, np.mean(loss_batch))
        current_epoch_loss = np.mean(loss_batch)
        loss_epoch.append(current_epoch_loss)
        print(stdout_msg)
        if np.mean(loss_batch) < best_metric:
            best_metric = np.mean(loss_batch)
            # / model / umls_trained_Attn.tar
            torch.save(model.state_dict(), model_save_path)
            print('save best model to {}'.format(model_save_path))
        # 可以正常训练，然后需要根据训练损失的变化情况保存模型；
        if epoch > 10 and current_epoch_loss > np.min(loss_epoch[-10:]):
            break