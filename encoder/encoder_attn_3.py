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

import torch.nn
import torch.nn as nn
from torch.nn import functional as   F
from tqdm import tqdm

EPSILON = float(np.finfo(float).eps)
HUGE_INT = 1e31

import os, argparse

def define_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--dataset', type=str, default='umls')
    parser.add_argument('--entity_dim', type=int, default=200)
    parser.add_argument('--input_drop', type=float, default=0.2)
    parser.add_argument('--feat_dropout_rate', type=float, default=0.2)
    parser.add_argument('--hidden_dropout_rate', type=float, default=0.3)
    parser.add_argument('--xavier_init', type=bool, default=True)
    parser.add_argument('--emb_2D_d1', type=int, default=10)
    parser.add_argument('--emb_2D_d2', type=int, default=20)
    parser.add_argument('--num_epochs', type=int, default=400, )
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--label_smoothing_epsilon', type=float, default=1e-8, help='epsilon used for label smoothing')
    parser.add_argument('--learning_rate', type=float, default=1e-3, )
    parser.add_argument("-w_gat", "--weight_decay_gat", type=float,default=5e-6, help='')
    parser.add_argument('--action_dropout_anneal_factor', type=float, default=0.9, )
    parser.add_argument('--epoch', type=int, default=2000, )
    parser.add_argument('--cuda',type=int,default=0)
    parser.add_argument('--number_multihop_valid_paths', type=int, default=2)
    parser.add_argument('--need_attn_values',type=bool,default=True)
    parser.add_argument('--ratio_valid_invalid',type=int,default=3)
    parser.add_argument('--USEGTDM',action='store_true',help='use the ground truth data mask in the process of building the multihop paths')
    parser.add_argument('--NOGTDM',action='store_true',help='no use the ground truth data mask')
    args = parser.parse_args()
    return args

# 数据集 WN18RR  ubuntu服务器
# python encoder_attn_3.py --dataset WN18RR --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 3  --cuda  begin end 1
# python encoder_attn_3.py --dataset WN18RR --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 3  --cuda begin end 1
# python encoder_attn_3.py --dataset WN18RR --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 3  --cuda begin end 1
# python encoder_attn_3.py --dataset WN18RR --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 1  --cuda  begin end 1
# python encoder_attn_3.py --dataset WN18RR --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 1  --cuda begin end 1 
# python encoder_attn_3.py --dataset WN18RR --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 1  --cuda begin end 1
# python encoder_attn_3.py --dataset WN18RR --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 2  --cuda  begin end 1
# python encoder_attn_3.py --dataset WN18RR --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 2  --cuda begin end 1
# python encoder_attn_3.py --dataset WN18RR --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 2  --cuda begin end 1   9个 


# python encoder_attn_3.py --dataset WN18RR --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 3  --cuda   begin end
# python encoder_attn_3.py --dataset WN18RR --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 3  --cuda  begin end
# python encoder_attn_3.py --dataset WN18RR --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 3  --cuda  begin end
# python encoder_attn_3.py --dataset WN18RR --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 1  --cuda   begin end
# python encoder_attn_3.py --dataset WN18RR --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 1  --cuda  begin end
# python encoder_attn_3.py --dataset WN18RR --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 1  --cuda  begin end
# python encoder_attn_3.py --dataset WN18RR --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 2  --cuda   begin end
# python encoder_attn_3.py --dataset WN18RR --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 2  --cuda  begin end
# python encoder_attn_3.py --dataset WN18RR --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 2  --cuda  begin end



# 数据集 NELL-995  lidong 
# python encoder_attn_3.py --dataset NELL-995 --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 3  --cuda  begin end
# python encoder_attn_3.py --dataset NELL-995 --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 3  --cuda begin end
# python encoder_attn_3.py --dataset NELL-995 --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 3  --cuda begin end
# python encoder_attn_3.py --dataset NELL-995 --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 1  --cuda  begin end
# python encoder_attn_3.py --dataset NELL-995 --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 1  --cuda begin end
# python encoder_attn_3.py --dataset NELL-995 --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 1  --cuda begin end
# python encoder_attn_3.py --dataset NELL-995 --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 2  --cuda  begin end
# python encoder_attn_3.py --dataset NELL-995 --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 2  --cuda begin end
# python encoder_attn_3.py --dataset NELL-995 --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 2  --cuda begin end 


# python encoder_attn_3.py --dataset NELL-995 --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 3  --cuda   begin end
# python encoder_attn_3.py --dataset NELL-995 --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 3  --cuda  begin end
# python encoder_attn_3.py --dataset NELL-995 --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 3  --cuda  begin end 
# python encoder_attn_3.py --dataset NELL-995 --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 1  --cuda 1 begin end
# python encoder_attn_3.py --dataset NELL-995 --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 1  --cuda  begin end
# python encoder_attn_3.py --dataset NELL-995 --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 1  --cuda  begin end 
# python encoder_attn_3.py --dataset NELL-995 --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 2  --cuda   begin end
# python encoder_attn_3.py --dataset NELL-995 --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 2  --cuda  begin end
# python encoder_attn_3.py --dataset NELL-995 --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 2  --cuda  begin end
 
# 数据集 FB15K-237 lidong 
# python3 encoder_attn_3.py --dataset FB15K-237 --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 3  --cuda 1  begin end 
# python3 encoder_attn_3.py --dataset FB15K-237 --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 3  --cuda 1 begin
# python encoder_attn_3.py --dataset FB15K-237 --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 3  --cuda 1  begin  
# python encoder_attn_3.py --dataset FB15K-237 --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 1  --cuda 2   begin 
# python encoder_attn_3.py --dataset FB15K-237 --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 1  --cuda 2  begin
# python encoder_attn_3.py --dataset FB15K-237 --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 1  --cuda 2  begin   
# python encoder_attn_3.py --dataset FB15K-237 --USEGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 2  --cuda 3   begin   
# python encoder_attn_3.py --dataset FB15K-237 --USEGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 2  --cuda 3  begin
# python encoder_attn_3.py --dataset FB15K-237 --USEGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 2  --cuda 3  begin    


# python encoder_attn_3.py --dataset FB15K-237 --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 3  --cuda      begin 
# python encoder_attn_3.py --dataset FB15K-237 --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 3  --cuda     begin 
# python encoder_attn_3.py --dataset FB15K-237 --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 3  --cuda     begin end
# python encoder_attn_3.py --dataset FB15K-237 --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid 1  --cuda      begin 
# python encoder_attn_3.py --dataset FB15K-237 --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 1  --cuda     begin 
# python encoder_attn_3.py --dataset FB15K-237 --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 1  --cuda     begin 
# python encoder_attn_3.py --dataset FB15K-237 --NOGTDM --number_multihop_valid_paths 2 --ratio_valid_invalid  2  --cuda     begin
# python encoder_attn_3.py --dataset FB15K-237 --NOGTDM --number_multihop_valid_paths 10 --ratio_valid_invalid 2  --cuda     begin end 
# python encoder_attn_3.py --dataset FB15K-237 --NOGTDM --number_multihop_valid_paths 20 --ratio_valid_invalid 2  --cuda     being end

def load_index(input_path):
    index, rev_index = {}, {}
    with open(input_path) as f:
        for i, line in enumerate(f.readlines()):
            v, _ = line.strip().split()
            index[v] = i
            rev_index[i] = v
    return index, rev_index

# 构建模型，设计模型，模型输入；
class PathEncoder_Attn(nn.Module):
    def __init__(self, args, num_entity, num_relation):
        super(PathEncoder_Attn, self).__init__()
        self.entity_emb = nn.Embedding(num_entity, args.entity_dim)
        self.relation_emb = nn.Embedding(num_relation, args.entity_dim)
        self.args = args
        if args.xavier_init:
            torch.nn.init.xavier_normal_(self.entity_emb.weight)
            torch.nn.init.xavier_normal_(self.relation_emb.weight)
        else:
            with open(os.path.join(relative_path, 'entity_xavier_vec.txt')) as f:
                self.entity_emb.weight.data = torch.tensor(np.loadtxt(f), dtype=torch.float32)
            with open(os.path.join(relative_path, 'relation_xavier_vec.txt')) as f:
                self.relation_emb.weight.data = torch.tensor(np.loadtxt(f), dtype=torch.float32)
        self.input_drop = nn.Dropout(args.input_drop)
        self.hidden_drop = nn.Dropout(args.hidden_dropout_rate)
        self.feature_map_drop = nn.Dropout2d(args.feat_dropout_rate)
        self.W1 = nn.Linear(args.entity_dim * 2, args.entity_dim)
        self.W2 = nn.Linear(args.entity_dim * 4  , 1)
        self.W3 = nn.Linear(args.entity_dim * 3, args.entity_dim)
        # batchnorm2d 1d 区别：
        self.bn0 = nn.BatchNorm1d(args.entity_dim * 2)  # 是否对输入的数据进行batchnorm
        self.b1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm1d(args.entity_dim)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, path, epoch, index):
        #  # print(train_data.shape) # train_triple_number * 8(路径数目) * 6(一条路径长度)
        # 输入是一个N * 8 * 6 * 200的嵌入向量表示；其中每一行代表一个三元组中对应的有效路径和无效路径
        # 因此对应的一个输入为一个batch 的第一列； 然后第二列；然后下一列； N  * 6
        # r0_emb : N * 200;
        # 所以输入的path 的 N * 8 * 6； 样本数据*路径数据*路径长度；一个样本元组对应的多个路径；
        r0_emb = self.relation_emb(path[:,:,0]) # N * 8 * 200; 1024 * 8 * 200
        e0_emb = self.entity_emb(path[:,:,1])  # 同上
        r1_emb = self.relation_emb(path[:,:,2]) # 同上
        e1_emb = self.entity_emb(path[:,:,3]) # 同上
        r2_emb = self.relation_emb(path[:,:,4]) # 同上
        e2_emb = self.entity_emb(path[:,:,5]) # 同上
        r3_emb = self.relation_emb(path[:,:,6])
        e3_emb = self.entity_emb(path[:,:,7]) # 
        action1 = torch.cat([r0_emb, e0_emb], dim=2)  # N * 8 * 400 ;
        action2 = torch.cat([r1_emb, e1_emb], dim=2) # 同上
        action3 = torch.cat([r2_emb, e2_emb], dim=2) # 同上
        action4 = torch.cat([r3_emb, e3_emb], dim=2)
        h1 = self.W1(action1)  # N * 8 * 200
        h2 = self.W1(action2) # 同上
        h3 = self.W1(action3) # 同上
        h4 = self.W1(action4)
        h1 = self.input_drop(h1) # N * 8 * 200
        h2 = self.input_drop(h2) # 同上
        h3 = self.input_drop(h3) # 同上
        h4 = self.input_drop(h4)
        h1 = torch.transpose(h1,1,2) # N * 200 * 8 把特征维度转至过来计算bn；
        h2 = torch.transpose(h2,1,2) # 同上
        h3 = torch.transpose(h3,1,2) # 同上
        h4 = torch.transpose(h4,1,2)
        h1 = self.bn2(h1)   # N * 200 * 8
        h2 = self.bn2(h2)   # 同上
        h3 = self.bn2(h3)   # 同上
        h4 = self.bn2(h4)
        h1 = torch.transpose(h1,1,2) # N * 8 * 200 把特征维度转至过来计算bn；
        h2 = torch.transpose(h2,1,2) # 同上
        h3 = torch.transpose(h3,1,2) # 同上
        h4 = torch.transpose(h4,1,2)
        b = torch.cat([h1, h2, h3,h4], dim=2)  # N * 8 * 600;
        a = self.W2(b) # N * 8 * 1 ;
        a = self.leakyrelu(a) # N * 8 * 1
        a = self.softmax(a) # N * 8 * 1
        if args.need_attn_values:
            if epoch == 0 and index == 0:
                with open(os.path.join(os.getcwd(), args.dataset+"_attn"+'.txt'),'w') as f:
                    print('save attn values, the path is :' + str( os.path.join(os.getcwd(), args.dataset+"_attn_values"+'.dat' )))
                    fls1 = a[:4,:1,0].clone().detach().cpu().numpy().tolist()
                    fls2 = a[:4,-3:,0].clone().detach().cpu().numpy().tolist()
                    fls = np.concatenate((fls1, fls2), axis=1)
                    np.savetxt(f,fls)
                    # f.write(str(a[:20]))
                    # f.write('\n')
            if epoch == 200 and index == 0:
                with open(os.path.join(os.getcwd(), args.dataset+"_attn"+'.txt'),'a') as f:
                    print('save attn values, the path is :' + str( os.path.join(os.getcwd(), args.dataset+"_attn_values"+'.dat' )))
                    fls1 =a[:4,:1,0].clone().detach().cpu().numpy().tolist()
                    fls2 = a[:4,-3:,0].clone().detach().cpu().numpy().tolist()
                    fls = np.concatenate((fls1, fls2), axis=1)
                    np.savetxt(f,fls)
                    # f.write(str(a[:20]))
                    # f.write('\n')
            if epoch == 400 and index == 0:
                with open(os.path.join(os.getcwd(), args.dataset+"_attn"+'.txt'),'a') as f:
                    print('save attn values, the path is :' + str( os.path.join(os.getcwd(), args.dataset+"_attn_values"+'.dat' )))
                    fls1 = a[:4,:1,0].clone().detach().cpu().numpy().tolist()
                    fls2 = a[:4,-3:,0].clone().detach().cpu().numpy().tolist()
                    fls = np.concatenate((fls1, fls2), axis=1)
                    np.savetxt(f,fls)
                    # f.write(str(a[:20]))
                    # f.write('\n')
        c = torch.cat([h1,h2,h3],dim=2) # N * 8 * 400
        c = self.W3(c) # N * 8 * 200;
        c = self.input_drop(c)
        c = torch.transpose(c,1,2) # N * 200 * 8
        c = self.bn2(c)
        c = torch.transpose(c,1,2) # N * 8 * 200
        t = h4 * c    # t: N * 8 * 200
        t = torch.sum(t,dim=2) # N * 8 ;
        a = torch.squeeze(a)
        res = a * t # N * 8 ; 将注意值 * 该路径的特征和；
        res = torch.sigmoid(res) # 将每个概率值映射到0-1空间；
        # return res : N * 8 ;
        return res


if __name__ == '__main__':

    args = define_args()
    path = os.getcwd()
    if args.dataset == "kinship" or args.dataset == "umls" or args.dataset == 'FB15K-237':
        bandwidth = 400
    elif args.dataset == 'NELL-995':
        bandwidth = 256
    elif args.dataset == 'WN18RR':
        bandwidth = 500 
    else:
        print('dataset is not exist')
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
    # M = 0
    # for i, l in enumerate(train_triples):
    #     es, et, rq = l[0], l[1], l[2]
    #     es_id, et_id, rq_id = entity2id[es], entity2id[et], relation2id[rq]   
    #     if  len(action_space_list[es_id]) <= 2 and set(action_space_list[es_id]) == set([(r_no_op_id, es_id),(rq_id,et_id)]) :
    #         M += 1 
    # print(M)
    # os._exit(0)
    '''
    两条边分别是原始ground truth data 与 自循环边； 
    NELL995数据集中以es开头的头实体，只有两条边的数目为 28198
    WN18RR数据集中以es开始的头实体，只有两条边的数目为 6041 
    FB15K数据集中以es开始的头实体，只有两条边的数目为 233 
    UMLS数据集中以es开始的头实体，只有两条边的数目为 0 
    kinship数据集中以es开始的实体，只有两条边的数目为 0 
    如果添加真实mask，则根本不可能找到目标实体的元组数目。
    WN18RR: Mask 真实元组 并找到可以游走的边，支持元组数目： 56120； 不支持数目 30715 这个是WN18RR数据集上相关模型精度低最根本的原因
    nell995 mask 真实元组，并找到可以游走的边，支持元组数据：106062，不支持数目， 43616 
    fb15k237  支持数目 在mask 真实元组  269808， 不支持数目2307
    '''
    
    path = {}
    path_valid = {}
    path_invalid = {}
    valid_path_number = {}
    invalid_path_number = {}

    # 构建每个元组的有效路径和无效路径；可以循环的边包括自循环边
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
                for action3 in action_space_list[action2[1]]: 
                    if ( (es_id,action1[0],action1[1])  in S or (action1[1],action2[0],action2[1]) in S  
                            or (action2[1],action3[0],action3[1])  in S ):
                        continue 
                    if action3[1] == et_id :
                        path_valid[i].append([start_relation_id, es_id, action1[0],action1[1],action2[0],action2[1],action3[0],action3[1]])
                        m += 1 
                    if action3[1] != et_id and (es_id, action3[1])  not in train_triples_es_et: 
                        path_invalid[i].append([start_relation_id, es_id, action1[0],action1[1],action2[0],action2[1],action3[0],action3[1]])
                        n += 1
                    if m >= args.number_multihop_valid_paths  and n >= args.number_multihop_valid_paths *  args.ratio_valid_invalid -1: 
                        break 
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
        path_invalid[i].append([start_relation_id, es_id,dummy_relation_id,dummy_entity_id,dummy_relation_id,dummy_entity_id,dummy_relation_id, dummy_entity_id])
        temp = []
        while (len(temp) < args.number_multihop_valid_paths * args.ratio_valid_invalid):
            random_number = random.choice(range(len(path_invalid[i])))
            temp.append(path_invalid[i][random_number]) 
        path_invalid[i] = temp 
        path[i] = path_valid[i] + path_invalid[i]

    print('the multihop triple nubmer is {}, all triple number is {}'.format(len(path), len(train_triples)))

    # wn18rr : the multihop triple nubmer is 86833, all triple number is 86835
    # model = PathEncoder_Attn(args,len(entity2id),len(relation2id))
    # loss = torch.nn.BCELoss()
    # label = torch.tensor([1,1,0,0,0,0,0,0],dtype=torch.float32)
    # label_all = label.unsqueeze(0).repeat(temp.shape[0],1)
    # e2_label = ((1 - self.label_smoothing_epsilon) * e2) + (1.0 / e2.size(1))
    # train_data = torch.tensor([i for i in path.values()]) 
    train_data = torch.tensor([i for i in path.values()],dtype= torch.long)
    label_valid = torch.tensor([1] * args.number_multihop_valid_paths, dtype = torch.float32)
    label_invalid = torch.tensor([0] * args.number_multihop_valid_paths * args.ratio_valid_invalid, dtype = torch.float32)
    label = torch.cat((label_valid, label_invalid, ))
    label_all = label.unsqueeze(0).repeat(train_data.shape[0],1) # [1., 1., 0., 0., 0., 0., 0., 0.]
    torch.cuda.set_device(args.cuda)
    if args.USEGTDM:
        model_save_path=os.path.join('model',args.dataset+'_'+'USEGTDM_'+str(args.number_multihop_valid_paths)+"_random_"+str(args.ratio_valid_invalid)+"_Attn_newpath"+'.tar')
    if args.NOGTDM:
        model_save_path=os.path.join('model',args.dataset+'_'+'NOGTDM_'+str(args.number_multihop_valid_paths)+"_random_"+str(args.ratio_valid_invalid)+"_Attn_newpath"+'.tar')
    print('model_save_path:', model_save_path)
    model = PathEncoder_Attn(args, len(entity2id), len(relation2id))
    model.cuda()
    criterion = nn.BCELoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate,weight_decay=args.weight_decay_gat)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=opt,step_size=100,gamma=0.5,last_epoch=-1)
    loss_epoch = []
    best_metric = np.inf
    for epoch in range(args.epoch):
        # 每个epoch 都 shuffle 数据；
        loss_batch = []
        model.train()
        idx = torch.randperm(train_data.shape[0])
        train_data = train_data[idx]
        for index in tqdm(range(0, len(train_data), args.batch_size)):
            opt.zero_grad()
            mini_batch = train_data[index:index + args.batch_size, :, :]
            mini_label = label_all[index:index + args.batch_size, :, ]
            if len(mini_batch) < args.batch_size:
                continue
            mini_label = ((1 - args.label_smoothing_epsilon) * mini_label) +  (1.0 / 1000)
            mini_label = mini_label.cuda()
            mini_batch = mini_batch.cuda()
            pred = model(mini_batch,epoch,index)
            loss_value_batch = criterion(pred, mini_label)
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
            torch.save(model.state_dict(), model_save_path)
            print('save best model to {}'.format(model_save_path))
        # 可以正常训练，然后需要根据训练损失的变化情况保存模型；
        if epoch > 100 and current_epoch_loss > np.mean(loss_epoch[-100:]):
            break