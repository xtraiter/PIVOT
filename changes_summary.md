# Báo cáo so sánh cấu trúc & chi tiết thay đổi mã nguồn (Detailed Code Diff & Changes Summary)

Bản báo cáo này đối chiếu chi tiết từng tệp tin, biến số và logic xử lý giữa mã nguồn đã được tối ưu hóa, tích hợp Learned Pruning (PIVOT) hiện tại với kho lưu trữ gốc của bài báo: [AndrewZhou924/one-shot-subgraph](https://github.com/AndrewZhou924/one-shot-subgraph/).

> [!IMPORTANT]
> **HƯỚNG DẪN CẬP NHẬT KHI CHUYỂN CHAT (CHAT TRANSFER INSTRUCTIONS):**
> 1. Khi mở cửa sổ chat mới, hãy dán câu lệnh sau vào ô chat đầu tiên để AI nhận diện dự án lập tức:
>    `Hãy đọc tệp changes_summary.md và quy tắc trong .agents/AGENTS.md để nắm bắt toàn bộ các chỉnh sửa so với repo gốc. Mỗi khi thực hiện tối ưu hóa hoặc thêm tính năng mới, hãy cập nhật lại changes_summary.md bằng cách ghi đè hoặc bổ sung phần code thay đổi (diff) tương ứng.`
> 2. Quy tắc tự động đã được cấu hình tại [AGENTS.md](file:///home/vanba/KLTN/one-shot-subgraph/.agents/AGENTS.md). Bất kỳ tác nhân AI nào khi tải workspace này đều sẽ tự động phát hiện quy tắc đó và duy trì cấu trúc tài liệu này.

---

## I. MÃ NGUỒN GỐC CỦA CÁC FILE ĐƯỢC CHỈNH SỬA (ORIGINAL CODE)

Dưới đây là nội dung toàn bộ mã nguồn của các file gốc (trước khi tối ưu hóa) để cung cấp ngữ cảnh đầy đủ cho bất kỳ AI nào đọc hiểu:

### 1. model.py (Original)
```python
import torch
import torch.nn as nn
from torch_scatter import scatter

class GNNLayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
        super(GNNLayer, self).__init__()
        self.n_rel = n_rel
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.attn_dim = attn_dim
        self.act = act
        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
        self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
        self.Wqr_attn = nn.Linear(in_dim, attn_dim)
        self.w_alpha  = nn.Linear(attn_dim, 1)
        self.W_h = nn.Linear(in_dim, out_dim, bias=False)
    
    def forward(self, q_sub, q_rel, r_idx, hidden, edges, n_node, shortcut=False):
        # edges: [h, r, t]
        sub = edges[:,0]
        rel = edges[:,1]
        obj = edges[:,2]
        hs = hidden[sub]
        hr = self.rela_embed(rel) # relation embedding of each edge
        h_qr = self.rela_embed(q_rel)[r_idx] # use batch_idx to get the query relation
        
        # message aggregation
        message = hs * hr
        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
        message = alpha * message        
        message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum') #ori
        
        # get new hidden representations
        hidden_new = self.act(self.W_h(message_agg))
        
        if shortcut: hidden_new = hidden_new + hidden
        
        return hidden_new

class GNN_auto(torch.nn.Module):
    def __init__(self, params, loader):
        super(GNN_auto, self).__init__()
        self.params = params
        self.n_layer = params.n_layer
        self.hidden_dim = params.hidden_dim
        self.attn_dim = params.attn_dim
        self.n_rel = params.n_rel
        self.n_ent = params.n_ent
        self.loader = loader
        acts = {'relu': nn.ReLU(), 'tanh': torch.tanh, 'idd': lambda x:x}
        act = acts[params.act]

        self.gnn_layers = []
        for i in range(self.n_layer):
            self.gnn_layers.append(GNNLayer(self.hidden_dim, self.hidden_dim, self.attn_dim, self.n_rel, act=act))
        self.gnn_layers = nn.ModuleList(self.gnn_layers)
        self.dropout = nn.Dropout(params.dropout)
        self.gate = nn.GRU(self.hidden_dim, self.hidden_dim)
        
        if self.params.initializer == 'relation': self.query_rela_embed = nn.Embedding(2*self.n_rel+1, self.hidden_dim)
        if self.params.readout == 'linear':
            if self.params.concatHidden:
                self.W_final = nn.Linear(self.hidden_dim * (self.n_layer+1), 1, bias=False)
            else:
                self.W_final = nn.Linear(self.hidden_dim, 1, bias=False)
        
    def forward(self, q_sub, q_rel, subgraph_data, mode='train'):
        ''' forward with extra propagation '''
        n = len(q_sub)
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = subgraph_data
        n_node = len(batch_idxs)
        h0 = torch.zeros((1, n_node, self.hidden_dim)).cuda()
        hidden = torch.zeros(n_node, self.hidden_dim).cuda()
        
        # initialize the hidden
        if self.params.initializer == 'binary':
            hidden[query_sub_idxs, :] = 1
        elif self.params.initializer == 'relation':
            hidden[query_sub_idxs, :] = self.query_rela_embed(q_rel)
        
        # store hidden at each layer or not
        if self.params.concatHidden: hidden_list = [hidden]
        
        # propagation
        for i in range(self.n_layer):
            # forward
            hidden = self.gnn_layers[i](q_sub, q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node,
                                        shortcut=self.params.shortcut)
            
            # act_signal is a binary (0/1) tensor 
            # that 1 for non-activated entities and 0 for activated entities
            act_signal = (hidden.sum(-1) == 0).detach().int()
            hidden = self.dropout(hidden)
            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
            hidden = hidden.squeeze(0)
            hidden = hidden * (1-act_signal).unsqueeze(-1)
            h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
            
            if self.params.concatHidden: hidden_list.append(hidden)

        # readout
        if self.params.readout == 'linear':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = self.W_final(hidden).squeeze(-1)        
        elif self.params.readout == 'multiply':
            if self.params.concatHidden: hidden = torch.cat(hidden_list, dim=-1)
            scores = torch.sum(hidden * hidden[query_sub_idxs][batch_idxs], dim=-1)
        
        # re-indexing
        scores_all = torch.zeros((n, self.loader.n_ent)).cuda()
        scores_all[batch_idxs, abs_idxs] = scores

        return scores_all
```

### 2. load_data.py (Original)
```python
import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
import pickle as pkl                 
import networkx as nx
import time
from tqdm import tqdm
from torch.utils.data import Dataset

class DataLoader(Dataset):
    def __init__(self, args, mode='train'):
        self.args = args
        self.mode = mode
        self.task_dir = task_dir = args.data_path
        with open(os.path.join(task_dir, 'entities.txt')) as f:
            self.entity2id = dict()
            n_ent = 0
            for line in f:
                entity = line.strip()
                self.entity2id[entity] = n_ent
                n_ent += 1
        with open(os.path.join(task_dir, 'relations.txt')) as f:
            self.relation2id = dict()
            n_rel = 0
            for line in f:
                relation = line.strip()
                self.relation2id[relation] = n_rel
                n_rel += 1

        self.n_ent = n_ent
        self.n_rel = n_rel
        self.filters = defaultdict(lambda:set())
        self.fact_triple  = self.read_triples('facts.txt')
        self.train_triple = self.read_triples('train.txt')
        self.valid_triple = self.read_triples('valid.txt')
        self.test_triple  = self.read_triples('test.txt')
        self.all_triple = np.concatenate([np.array(self.fact_triple), np.array(self.train_triple)], axis=0)

        # add inverse
        self.fact_data  = self.double_triple(self.fact_triple)
        self.train_data = np.array(self.double_triple(self.train_triple))
        self.valid_data = self.double_triple(self.valid_triple)
        self.test_data  = self.double_triple(self.test_triple)
        self.idd_data = np.concatenate([np.expand_dims(np.arange(self.n_ent),1), 2*self.n_rel*np.ones((self.n_ent, 1)), np.expand_dims(np.arange(self.n_ent),1)], 1)
            
        self.shuffle_train()
        self.valid_q, self.valid_a = self.load_query(self.valid_data)
        self.test_q,  self.test_a  = self.load_query(self.test_data)
        self.n_train = len(self.train_data)
        self.n_valid = len(self.valid_q)
        self.n_test  = len(self.test_q)

        for filt in self.filters:
            self.filters[filt] = list(self.filters[filt])
            
        if mode == 'train':
            self.len = len(self.train_data)
        elif mode == 'valid':
            self.len = len(self.valid_q)
        else:
            self.len = len(self.test_q)
                
    def addSampler(self, sampler):
        self.sampler = sampler
        self.getOneSubgraph = self.sampler.getOneSubgraph
        self.getBatchSubgraph = self.sampler.getBatchSubgraph
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        # indexing
        if self.mode == 'train':
            sub, rel, obj = self.train_data[idx]
            sub = torch.LongTensor([sub]).unsqueeze(0)
            rel = torch.LongTensor([rel]).unsqueeze(0)
            obj = torch.LongTensor([obj]).unsqueeze(0)
        else:
            if self.mode == 'valid':
                query, answer = self.valid_q, self.valid_a
            elif self.mode == 'test':
                query, answer = self.test_q, self.test_a
            sub, rel = query[idx]
            sub, rel = torch.LongTensor([sub]), torch.LongTensor([rel])
            obj = torch.zeros((self.n_ent)).long()
            obj[answer[idx]] = 1
                    
        # subgraph sampling
        subgraph = self.getOneSubgraph(int(sub))
        return sub, rel, obj, subgraph
        
    def collate_fn(self, data):
        subs = torch.stack([_[0] for _ in data], dim=0)
        rels = torch.stack([_[1] for _ in data], dim=0)
        objs = torch.stack([_[2] for _ in data], dim=0)
        subgraph_list = [_[3] for _ in data]
        batch_subgraph = self.getBatchSubgraph(subgraph_list)
        
        batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = batch_subgraph
        return subs, rels, objs, batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges

    def read_triples(self, filename):
        triples = []
        with open(os.path.join(self.task_dir, filename)) as f:
            for line in f:
                h, r, t = line.strip().split()
                h, r, t = self.entity2id[h], self.relation2id[r], self.entity2id[t]
                triples.append([h,r,t])
                self.filters[(h,r)].add(t)
                self.filters[(t,r+self.n_rel)].add(h)
        return triples

    def double_triple(self, triples):
        new_triples = []
        for triple in triples:
            h, r, t = triple
            new_triples.append([t, r+self.n_rel, h]) 
        return list(triples) + new_triples

    def load_query(self, triples):
        triples.sort(key=lambda x:(x[0], x[1]))
        trip_hr = defaultdict(lambda:list())

        for trip in triples:
            h, r, t = trip
            trip_hr[(h,r)].append(t)
        
        queries = []
        answers = []
        for key in trip_hr:
            queries.append(key)
            answers.append(np.array(trip_hr[key]))
        return queries, answers
    
    def shuffle_train(self):
        all_triple = self.all_triple
        n_all = len(all_triple)
        rand_idx = np.random.permutation(n_all)
        all_triple = all_triple[rand_idx]
        bar = int(n_all * self.args.fact_ratio)
        self.fact_data = np.array(self.double_triple(all_triple[:bar].tolist()))
        self.train_data = np.array(self.double_triple(all_triple[bar:].tolist()))
        
        if self.args.remove_1hop_edges:
            print('==> removing 1-hop links...')
            tmp_index = np.ones((self.n_ent, self.n_ent))
            tmp_index[self.train_data[:, 0], self.train_data[:, 2]] = 0
            save_facts = tmp_index[self.fact_data[:, 0], self.fact_data[:, 2]].astype(bool)
            self.fact_data = self.fact_data[save_facts]
            print('==> done')

        self.n_train = len(self.train_data)
        self.len = len(self.train_data)        
        
        n_all = len(self.train_data)
        rand_idx = np.random.permutation(n_all)
        self.train_data = self.train_data[rand_idx]
```

### 3. PPR_sampler.py (Original)
```python
import os
import torch
from scipy.sparse import csr_matrix
import numpy as np
from collections import defaultdict
import pickle as pkl                 
import networkx as nx
import time
from tqdm import tqdm
from utils import checkPath
import copy
from functools import partial

class pprSampler(object):
    def __init__(self, n_ent, n_rel, topk, topm, homoEdges, edge_index, data_path, split='train', args=None):
        ''' 
            args:
            topk: number of sampled nodes for one head entity 
            edge_index: list of triples [(h,r,t)]
            data_path: path to save the ppr/subgraphs files
        '''
        print('==> initializing ppr sampler...')
        self.args = args
        self.n_ent = n_ent
        self.n_samp_ent = args.n_samp_ent
        self.n_rel = n_rel
        self.topk = topk
        self.topm = topm
        self.edge_index = edge_index
        self.data_folder = data_path
        self.homoEdges = homoEdges
        self.homoTrainGraph = self.triplesToNxGraph(self.homoEdges)
        self.ppr_savePath = os.path.join(self.data_folder, f'ppr_scores/')
        checkPath(self.ppr_savePath)
        print('==> checking ppr scores for each entity...')
        
        for h in tqdm(range(self.n_ent), ncols=50, leave=False):
            ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(h)}.pkl')
            if os.path.exists(ent_ppr_savePath):
                pass
            else:
                h_ppr_scores = self.generatePPRScoresForOneEntity(h)
                pkl.dump(h_ppr_scores, open(ent_ppr_savePath, 'wb'))
        print('finished.')
        
        heads, edges = [h for (h,r,t) in edge_index], list(range(len(edge_index)))
        print(len(heads), len(edges), max(heads), self.n_ent)
        self.sparseTrainMatrix = csr_matrix((edges, (heads, edges)), shape=(self.n_ent, len(edge_index)))

        self.edge_index = torch.LongTensor(self.edge_index)

        del self.homoEdges
        del self.homoTrainGraph
        print('==> finish sampler initilization.')

    def updateEdges(self, edge_index):
        heads, edges = [h for (h,r,t) in edge_index], list(range(len(edge_index)))
        self.sparseTrainMatrix = csr_matrix((edges, (heads, edges)), shape=(self.n_ent, len(edge_index)))
        self.edge_index = torch.LongTensor(edge_index)
    
    def getPPRscores(self, ent):
        ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(ent)}.pkl')
        scores = pkl.load(open(ent_ppr_savePath, 'rb'))
        return scores
        
    def generatePPRScoresForOneEntity(self, h, method='nx'):
        if method == 'nx':
            scores = nx.pagerank(self.homoTrainGraph, personalization={h: 1})
        return scores
    
    def triplesToNxGraph(self, edges):
        graph = nx.Graph()
        nodes = list(range(self.n_ent))
        graph.add_nodes_from(nodes)        
        graph.add_edges_from(edges)
        return graph
    
    def sampleSubgraph(self, ent: int, cand=None):    
        ppr_scores = np.array(list(self.getPPRscores(ent).values()))
        
        if cand != None and self.topk < self.n_ent:
            tmp_ppr_scores = copy.deepcopy(ppr_scores)
            tmp_ppr_scores[cand] = 1e8
            topk_nodes = sorted(list(set([ent] + np.argsort(tmp_ppr_scores)[::-1][:self.topk].tolist())))
        else:
            if self.topk < self.n_ent:    
                topk_nodes = sorted(list(set([ent] + np.argsort(ppr_scores)[::-1][:self.topk].tolist())))
            else:
                topk_nodes = list(range(self.n_ent))

        selectd_edges = self.sparseTrainMatrix[topk_nodes, :]   
        _, tmp_edge_index = selectd_edges.nonzero()
        
        edges = self.edge_index[tmp_edge_index]
        topk_nodes = torch.LongTensor(topk_nodes)
        
        mask = torch.isin(edges[:,2], topk_nodes)
        sampled_edges = edges[mask, :]
        
        edge_num = int(sampled_edges.shape[0])
        if self.topm > 0 and edge_num > self.topm:
            heads, tails = sampled_edges[:,0], sampled_edges[:,2]
            edge_weights = ppr_scores[heads] + ppr_scores[tails]
            edge_weights = torch.Tensor(edge_weights)
            index = torch.topk(edge_weights, self.topm).indices
            sampled_edges = sampled_edges[index]
        
        node_index = torch.zeros(self.n_ent).long()
        node_index[topk_nodes] = torch.arange(len(topk_nodes))
              
        if self.args.add_manual_edges:
            add_edges_head2tails = torch.zeros((len(topk_nodes), 3)).long()
            add_edges_head2tails[:, 0] = ent
            add_edges_head2tails[:, 1] = 2*self.n_rel + 1
            add_edges_head2tails[:, 2] = topk_nodes
            add_edges_tails2head = torch.zeros((len(topk_nodes), 3)).long()
            add_edges_tails2head[:, 0] = topk_nodes
            add_edges_tails2head[:, 1] = 2*self.n_rel + 2
            add_edges_tails2head[:, 2] = ent
            sampled_edges = torch.cat([sampled_edges, add_edges_head2tails, add_edges_tails2head], dim=0)
        
        return topk_nodes, node_index, sampled_edges

    def getOneSubgraph(self, head: int, cand=None):
        topk_nodes, node_index, sampled_edges = self.sampleSubgraph(head, cand) 
        return [head, topk_nodes, node_index, sampled_edges]
        
    def getBatchSubgraph(self, subgraph_list: list):  
        batchsize = len(subgraph_list)
        ent_delta_values = [0]
        batch_sampled_edges = []
        batch_idxs, abs_idxs = [], []
        query_sub_idxs = []
        edge_batch_idxs = []

        for batch_idx in range(batchsize):       
            sub, topk_nodes, node_index, sampled_edges = subgraph_list[batch_idx]
            num_nodes = len(topk_nodes)
            ent_delta = sum(ent_delta_values)

            sampled_edges[:,0] = node_index[sampled_edges[:,0]] + ent_delta
            sampled_edges[:,2] = node_index[sampled_edges[:,2]] + ent_delta
            batch_sampled_edges.append(sampled_edges)
            edge_batch_idxs += [batch_idx] * int(sampled_edges.shape[0])

            ent_delta_values.append(num_nodes)
            batch_idxs += [batch_idx] * num_nodes
            abs_idxs += topk_nodes.tolist()
            query_sub_idxs.append(int(node_index[sub]) + ent_delta)
        
        batch_idxs = torch.LongTensor(batch_idxs)
        abs_idxs = torch.LongTensor(abs_idxs)
        batch_sampled_edges = torch.cat(batch_sampled_edges, dim=0)
        edge_batch_idxs = torch.LongTensor(edge_batch_idxs)
        query_sub_idxs = torch.LongTensor(query_sub_idxs)
        
        return batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges
```

### 4. train_auto.py (Original)
```python
import os
import argparse
import torch
import time
import numpy as np
from load_data import DataLoader
from base_model import BaseModel
from utils import *
from PPR_sampler import pprSampler

parser = argparse.ArgumentParser(description="Parser for the one-shot-subgraph framework")
parser.add_argument('--data_path', type=str, default='data/WN18RR/')
parser.add_argument('--seed', type=str, default=1234)
parser.add_argument('--topk', type=float, default=0.1) 
parser.add_argument('--topm', type=float, default=-1) 
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--fact_ratio', type=float, default=0.75)
parser.add_argument('--val_num', type=int, default=-1) 
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--layer', type=int, default=6)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--cpu', type=int, default=1)
parser.add_argument('--weight', type=str, default='')
parser.add_argument('--add_manual_edges', action='store_true')
parser.add_argument('--remove_1hop_edges', action='store_true')
parser.add_argument('--only_eval', action='store_true')
parser.add_argument('--not_shuffle_train', action='store_true')
args = parser.parse_args()

class Options(object):
    pass

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(8, args.cpu))
    torch.multiprocessing.set_sharing_strategy('file_system')
    
    dataset = args.data_path
    dataset = dataset.split('/')
    if len(dataset[-1]) > 0:
        dataset = dataset[-1]
    else:
        dataset = dataset[-2]
    
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(os.path.join(results_dir, dataset)):
        os.makedirs(os.path.join(results_dir, dataset))

    opts = args
    time = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
    opts.perf_file = os.path.join(results_dir,  dataset + '/' + time + '.txt')
    gpu = args.gpu
    torch.cuda.set_device(gpu)
    print('==> gpu:', gpu)
    opts.n_batch = opts.n_tbatch = int(args.batchsize)
    with open(opts.perf_file, 'a+') as f:
        f.write(str(opts))
    
    loader = DataLoader(args, mode='train')
    val_loader = DataLoader(args, mode='valid')
    test_loader = DataLoader(args, mode='test')
    opts.n_ent = loader.n_ent
    opts.n_rel = loader.n_rel
    
    args.n_samp_ent = int(args.topk * loader.n_ent)
    args.n_samp_edge = int(args.topm * len(loader.fact_data)) if args.topm > 0  else -1
    print(f'==> #sampled entities:{args.n_samp_ent}, #sampled edges:{args.n_samp_edge}')
    
    test_data = loader.double_triple(loader.all_triple)
    test_homo_edges = list(set([(h,t) for (h,r,t) in test_data]))
    test_data = np.concatenate([np.array(test_data), loader.idd_data], 0)
    test_sampler = pprSampler(loader.n_ent, loader.n_rel, args.n_samp_ent, args.n_samp_edge,
        test_homo_edges, test_data, args.data_path, split='test', args=args)

    del test_homo_edges
        
    fact_homo_edges = list(set([(h,t) for (h,r,t) in loader.fact_data]))
    fact_data = np.concatenate([np.array(loader.fact_data), loader.idd_data], 0)
    train_sampler = pprSampler(loader.n_ent, loader.n_rel, args.n_samp_ent, args.n_samp_edge,
        fact_homo_edges, fact_data, args.data_path, split='train', args=args)
        
    del fact_homo_edges
        
    loader.addSampler(train_sampler)
    val_loader.addSampler(test_sampler)
    test_loader.addSampler(test_sampler)
    
    checkPath('./results/')
    checkPath(f'./results/{dataset}/')
    checkPath(f'{args.data_path}/saveModel/')
            
    def run_model(params):       
        print('==> start training...')    
        print(params)
        args.lr = params['lr']
        args.decay_rate = params['decay_rate']
        args.lamb = params['lamb']
        args.hidden_dim = params['hidden_dim']
        args.attn_dim = params['attn_dim']
        args.n_layer = args.layer = params['n_layer']
        args.dropout = params['dropout']
        args.act = params['act']
        args.initializer = params['initializer']
        args.concatHidden = params['concatHidden']
        args.shortcut = params['shortcut']
        args.readout = params['readout']
        
        model = BaseModel(args, loaders=(loader, val_loader, test_loader), samplers=(train_sampler, test_sampler))
        
        if args.weight != '': 
            model.loadModel(args.weight)
            
        if args.only_eval:
            valid_mrr, out_str = model.evaluate(verbose=True, rank_CR=False)
            print(out_str)
            exit()

        best_mrr, best_test_mrr, bearing = 0, 0, 0
        for epoch in range(args.epoch):
            mrr, out_str = model.train_batch()
            
            with open(opts.perf_file, 'a+') as f:
                f.write(out_str)
                
            if mrr > best_mrr:
                best_mrr = mrr
                best_str = out_str
                print(str(epoch) + '\t' + best_str)
                bearing = 0
                
                BestMetricStr = f'ValMRR_{str(mrr)[:5]}'
                model.saveModelToFiles(args, BestMetricStr, deleteLastFile=False)
            else:
                bearing += 1
                
            if bearing >= 20: 
                print(f'early stopping at {epoch+1} epoch.')
                break
        
        print(best_str)
        return best_mrr
    
    if dataset == 'WN18RR':
        params = {'lr': 0.0001, 'hidden_dim': 256, 'attn_dim': 8, 'n_layer': 8, 'act': 'idd', 'initializer': 'relation', 'concatHidden': False, 'shortcut': True, 'readout': 'multiply', 'decay_rate': 0.8662400068095666, 'lamb': 0.00039154537550520227, 'dropout': 0.004323645605227445}
    elif dataset == 'nell':
        params = {'lr': 0.0011, 'hidden_dim': 128, 'attn_dim': 64, 'n_layer': 8, 'act': 'relu', 'initializer': 'relation', 'concatHidden': False, 'shortcut': False, 'readout': 'linear', 'decay_rate': 0.9938, 'lamb': 0.000089, 'dropout': 0.0193}
    elif dataset == 'YAGO':
        params = {'lr': 0.001, 'hidden_dim': 64, 'attn_dim': 2, 'n_layer': 8, 'act': 'relu', 'initializer': 'binary', 'concatHidden': True, 'shortcut': False, 'readout': 'linear', 'decay_rate': 0.9429713470775948, 'lamb': 0.000946516892415447, 'dropout': 0.19456805575101324}
    else:
        exit()
        
    run_model(params)
```

### 5. base_model.py (Original)
```python
import os
import torch
import numpy as np
import time
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from model import *
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import torch.nn.functional as F
import copy

class BaseModel(object):
    def __init__(self, args, loaders, samplers):
        self.args = args
        loader, val_loader, test_loader = loaders
        self.loader = loader
        self.model = GNN_auto(args, loader)
        self.model.cuda()
        self.n_ent = loader.n_ent
        self.n_samp_ent = args.n_samp_ent
        self.n_rel = loader.n_rel
        self.train_sampler, self.test_sampler = samplers
        self.trainLoader = DataLoader(loader, batch_size=args.n_batch, num_workers=args.cpu, collate_fn=loader.collate_fn, shuffle=False, prefetch_factor=args.cpu, pin_memory=True)
        self.valLoader = DataLoader(val_loader, batch_size=args.n_tbatch, num_workers=args.cpu, collate_fn=val_loader.collate_fn, shuffle=False, prefetch_factor=args.cpu, pin_memory=True)
        self.testLoader = DataLoader(test_loader, batch_size=args.n_tbatch, num_workers=args.cpu, collate_fn=test_loader.collate_fn, shuffle=False, prefetch_factor=args.cpu, pin_memory=True)
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, min_lr=args.lr/20, verbose=True)
        self.smooth = 1e-5
        self.t_time = 0
        self.mean_rank_dict = {}
        
    def saveModelToFiles(self, args, best_metric, deleteLastFile=True):
        if args.val_num == -1:
            savePath = f'{self.args.data_path}/saveModel/topk_{self.args.topk}_layer_{self.args.layer}_{best_metric}.pt'
        else:
            savePath = f'{self.args.data_path}/saveModel/topk_{self.args.topk}_layer_{self.args.layer}_valNum_{self.args.val_num}_{best_metric}.pt'
            
        print(f'Save checkpoint to : {savePath}')
        torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_mrr':best_metric,
                }, savePath)
        
    def loadModel(self, filePath):
        print(f'Load weight from {filePath}')
        assert os.path.exists(filePath)
        checkpoint = torch.load(filePath, map_location=torch.device(f'cuda:{self.args.gpu}'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer = Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.lamb)

    def prepareData(self, batch_data):
        subs, rels, objs, batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges = batch_data
        subgraph_data = [batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs.cuda(), batch_sampled_edges.cuda()]
        subs = subs.cuda().flatten()
        rels = rels.cuda().flatten()
        objs = objs.cuda()
        return subs, rels, objs, subgraph_data
        
    def train_batch(self,):        
        epoch_loss = 0
        reach_tails_list = []
        t_time = time.time()
        self.model.train()
        
        for batch_data in tqdm(self.trainLoader, ncols=50, leave=False):                      
            subs, rels, objs, subgraph_data = self.prepareData(batch_data)
            
            self.model.zero_grad()
            scores = self.model(subs, rels, subgraph_data)
            
            pos_scores = scores[[torch.arange(len(scores)).cuda(), objs.flatten()]]
            max_n = torch.max(scores, 1, keepdim=True)[0]
            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1))) 

            loss.backward()
            self.optimizer.step()

            reach_tails = (pos_scores == 0).detach().int().reshape(-1).cpu().tolist()
            reach_tails_list += reach_tails
            epoch_loss += loss.item()
            
        self.t_time += time.time() - t_time
        
        valid_mrr, out_str = self.evaluate()    
        self.scheduler.step(valid_mrr)
        
        if self.args.not_shuffle_train:
            pass
        else:
            self.loader.shuffle_train()
            fact_data = np.concatenate([np.array(self.loader.fact_data), self.loader.idd_data], 0)
            self.train_sampler.updateEdges(fact_data)
        
        return valid_mrr, out_str
    
    @torch.no_grad()
    def evaluate(self, eval_val=True, eval_test=True, verbose=False, rank_CR=False, mean_rank=False):
        ranking = []
        self.model.eval()
        i_time = time.time()
        
        if eval_val:
            val_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.valLoader, ncols=50, leave=False):      
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                
                scores = self.model(subs, rels, subgraph_data, mode='valid').data.cpu().numpy()

                subs = subs.cpu().numpy()
                rels = rels.cpu().numpy()
                objs = objs.cpu().numpy()
                filters = []
                for i in range(len(subs)):
                    filt = self.loader.filters[(subs[i], rels[i])]
                    filt_1hot = np.zeros((self.n_ent, ))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot)
                filters = np.array(filters)
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks
                
                if mean_rank: 
                    mean_ranks = cal_ranks_mean(scores, objs, filters)
                    mean_rank_list += mean_ranks

                ans = np.nonzero(objs)
                ans_score = scores[ans].reshape(-1)
                reach_tails = (ans_score == 0).astype(int).tolist() # (0/1)
                val_reach_tails_list += reach_tails

            ranking = np.array(ranking)
            v_mrr, v_h1, v_h10 = cal_performance(ranking)
            
            if rank_CR:
                target_rank = torch.Tensor(ranking).reshape(-1)
                rank_thre = [int(i/100 * self.loader.n_ent) for i in range(1,101)]
                rank_CR = []
                for thre in rank_thre:
                    ratio = torch.sum((target_rank <= thre).int()) / len(target_rank)
                    rank_CR.append(float(ratio))
                print('Val set:\n', rank_CR)
                
            if mean_rank: self.mean_rank_dict['val'] = copy.deepcopy(mean_rank_list)
                
        else:
            v_mrr, v_h1, v_h10 = -1, -1, -1
        
        if eval_test:
            ranking = []
            test_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.testLoader, ncols=50, leave=False):        
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                
                scores = self.model(subs, rels, subgraph_data, mode='test').data.cpu().numpy()

                subs = subs.cpu().numpy()
                rels = rels.cpu().numpy()
                objs = objs.cpu().numpy()
                filters = []
                for i in range(len(subs)):
                    filt = self.loader.filters[(subs[i], rels[i])]
                    filt_1hot = np.zeros((self.n_ent, ))
                    filt_1hot[np.array(filt)] = 1
                    filters.append(filt_1hot)
                filters = np.array(filters)
                ranks = cal_ranks(scores, objs, filters)
                ranking += ranks

                if mean_rank: 
                    mean_ranks = cal_ranks_mean(scores, objs, filters)
                    mean_rank_list += mean_ranks
                    
                ans = np.nonzero(objs)
                ans_score = scores[ans].reshape(-1)
                reach_tails = (ans_score == 0).astype(int).tolist() # (0/1)
                test_reach_tails_list += reach_tails

            ranking = np.array(ranking)
            t_mrr, t_h1, t_h10 = cal_performance(ranking)
            
            if rank_CR:
                target_rank = torch.Tensor(ranking).reshape(-1)
                rank_thre = [int(i/100 * self.loader.n_ent) for i in range(1,101)]
                rank_CR = []
                for thre in rank_thre:
                    ratio = torch.sum((target_rank <= thre).int()) / len(target_rank)
                    rank_CR.append(float(ratio))
                print('Test set:\n', rank_CR)
                
            if mean_rank: self.mean_rank_dict['test'] = copy.deepcopy(mean_rank_list)
            
        else:
            t_mrr, t_h1, t_h10 = -1, -1, -1
            
        i_time = time.time() - i_time
        out_str = '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n'%(v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.t_time, i_time)
        return v_mrr, out_str
```

---

## II. CHI TIẾT CÁC THAY ĐỔI VÀ KHỐI MÃ DIFF (DIFF CHUNKS)

Dưới đây là chi tiết các thay đổi từ mức biến số đến mức hàm kèm mã `git diff` so với code gốc ở trên:

### 1. PPR_sampler.py (Sửa đổi lấy mẫu Subgraph)
*   **Song song hóa tính toán PPR (PPR Score Parallelization) & Đọc tệp RAM (Pre-loading)**
```diff
@@ -14,6 +14,94 @@
 import os
 +import multiprocessing
 +import concurrent.futures
 +import glob
 +from utils import get_hop_distances
 +
 +
 +_global_homo_graph = None
 +
 +def _init_worker(graph):
 +    global _global_homo_graph
 +    _global_homo_graph = graph
 +
 +def _compute_and_save_ppr_scores(h, ppr_save_path):
 +    global _global_homo_graph
 +    import pickle as pkl
 +    import tempfile
 +    
 +    final_path = os.path.join(ppr_save_path, f'{int(h)}.pkl')
 +    if os.path.exists(final_path) and os.path.getsize(final_path) > 1000:
 +        return
 +        
 +    scores = nx.pagerank(_global_homo_graph, alpha=0.85, personalization={h: 1})
 +    
 +    with tempfile.NamedTemporaryFile(dir=ppr_save_path, delete=False) as tf:
 +        pkl.dump(scores, tf)
 +        temp_name = tf.name
 +        
 +    os.replace(temp_name, final_path)
 +
 +def _load_one_ppr(args):
 +    path, n_ent = args
 +    import pickle as pkl
 +    import os
 +    h = int(os.path.basename(path).split('.')[0])
 +    with open(path, 'rb') as f:
 +        scores = pkl.load(f)
 +    row = np.zeros(n_ent, dtype=np.float32)
 +    for k, v in scores.items():
 +        row[k] = v
 +    return h, row
  
  class pprSampler(object):
      def __init__(self, args, edge_index, n_ent, n_rel, mode='train'):
@@ -48,16 +136,16 @@
          else:
              # run nx pagerank in parallel using multiprocessing pool
              print("==> Computing PPR scores in parallel...")
 -            for h in tqdm(range(self.n_ent), ncols=50):
 -                scores = nx.pagerank(graph, alpha=0.85, personalization={h:1})
 -                pkl.dump(scores, open(os.path.join(self.ppr_savePath, f'{int(h)}.pkl'), 'wb'))
 +            num_workers = min(64, os.cpu_count() - 4)
 +            entities_to_compute = []
 +            for h in range(self.n_ent):
 +                final_path = os.path.join(self.ppr_savePath, f'{int(h)}.pkl')
 +                if not (os.path.exists(final_path) and os.path.getsize(final_path) > 1000):
 +                    entities_to_compute.append(h)
 +            
 +            if len(entities_to_compute) > 0:
 +                with multiprocessing.Pool(processes=num_workers, initializer=_init_worker, initargs=(graph,)) as pool:
 +                    list(tqdm(pool.imap_unordered(
 +                        partial(_compute_and_save_ppr_scores, ppr_save_path=self.ppr_savePath),
 +                        entities_to_compute
 +                    ), total=len(entities_to_compute), desc="PPR calculation"))
 +
 +        # Pre-load PPR scores into memory if dataset is small to save I/O overhead
 +        if self.n_ent <= 50000:
 +            print("==> Pre-loading PPR scores into memory...")
 +            self.all_ppr_scores = np.zeros((self.n_ent, self.n_ent), dtype=np.float32)
 +            ppr_files = glob.glob(os.path.join(self.ppr_savePath, '*.pkl'))
 +            load_args = [(f, self.n_ent) for f in ppr_files]
 +            num_workers = min(32, os.cpu_count())
 +            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
 +                results = list(tqdm(executor.map(_load_one_ppr, load_args), total=len(load_args), desc="Pre-loading PPR"))
 +                for h, row in results:
 +                    self.all_ppr_scores[h] = row
 +            self.use_in_memory_ppr = True
 +        else:
 +            self.use_in_memory_ppr = False
 ```
 
 *   **Tích hợp Learned Pruning (MLP) & Lấy mẫu lai (Hybrid Sampler)**
 ```diff
 @@ -71,6 +159,40 @@
 +        if hasattr(self.args, 'use_learned_pruning') and self.args.use_learned_pruning:
 +            print("==> Initializing MLP Pruning components in sampler...")
 +            self.adj = defaultdict(list)
 +            for h, t in homoEdges:
 +                self.adj[int(h)].append(int(t))
 +                self.adj[int(t)].append(int(h))
 +
 +            self.degrees = torch.zeros(self.n_ent)
 +            for i in range(self.n_ent):
 +                self.degrees[i] = len(self.adj[i])
 +
 +            self.direct_edges_set = set()
 +            for h, r, t in edge_index:
 +                self.direct_edges_set.add((int(h), int(r), int(t)))
 +
 +            tail_freq = np.zeros((self.n_ent, 2 * self.n_rel + 2), dtype=np.float32)
 +            for h, r, t in edge_index:
 +                tail_freq[int(t), int(r)] += 1.0
 +            rel_total = tail_freq.sum(axis=0, keepdims=True)
 +            self.tail_freq_norm = torch.tensor(tail_freq / (rel_total + 1e-8))
 +
 +            rel_counts = np.zeros((self.n_ent, 2 * self.n_rel + 2), dtype=np.float32)
 +            for h, r, t in edge_index:
 +                rel_counts[int(h), int(r)] += 1.0
 +            row_sums = rel_counts.sum(axis=1, keepdims=True)
 +            self.rel_dist = torch.tensor(rel_counts / (row_sums + 1e-8))
 +
 +            from learned_pruning import PruningMLP
 +            self.pruning_model = PruningMLP(in_dim=7, hidden=64)
 +            if hasattr(self.args, 'pruning_model_path') and self.args.pruning_model_path:
 +                print(f"==> Loading learned pruning model from: {self.args.pruning_model_path}")
 +                self.pruning_model.load_state_dict(torch.load(self.args.pruning_model_path, map_location='cpu'))
 +            self.pruning_model.eval()
   
       def updateEdges(self, edge_index):
 @@ -108,22 +233,90 @@
 -    def sampleSubgraph(self, ent: int, cand=None):    
 +    def build_features_for_inference(self, u, q, candidate_ids, ppr_scores):
 +        N = candidate_ids.numel()
 +        cids_cpu = candidate_ids.cpu()
 +        ppr_log = torch.log(ppr_scores + 1e-8)
 +        order = torch.argsort(-ppr_scores)
 +        ranks = torch.empty(N, device=ppr_scores.device)
 +        ranks[order] = torch.arange(N, dtype=torch.float32, device=ppr_scores.device)
 +        ppr_rank_pct = 1.0 - ranks / max(N - 1, 1)
 +        deg_log = torch.log1p(self.degrees[cids_cpu]).to(ppr_scores.device)
 +        hop_dist = torch.full((N,), 4.0, device=ppr_scores.device)
 +        bfs_dists = get_hop_distances(self.adj, u, max_hops=3)
 +        for i, cid in enumerate(cids_cpu.tolist()):
 +            if cid in bfs_dists:
 +                hop_dist[i] = float(bfs_dists[cid])
 +        is_direct = torch.zeros(N, device=ppr_scores.device)
 +        for i, cid in enumerate(cids_cpu.tolist()):
 +            if (u, q, cid) in self.direct_edges_set:
 +                is_direct[i] = 1.0
 +        tail_freq_q = self.tail_freq_norm[cids_cpu, q].to(ppr_scores.device)
 +        rel_match = self.rel_dist[cids_cpu, q].to(ppr_scores.device)
 +        feats = torch.stack([
 +            ppr_log, ppr_rank_pct, deg_log, hop_dist,
 +            is_direct, tail_freq_q, rel_match
 +        ], dim=1)
 +        feats = (feats - feats.mean(0, keepdim=True)) / (feats.std(0, keepdim=True) + 1e-6)
 +        return feats
 +
 +    def sampleSubgraph(self, ent: int, rel: int = None, cand=None):    
          # sample subgraph to get the edges
 -        ppr_scores = np.array(list(self.getPPRscores(ent).values()))
 +        if hasattr(self, 'use_in_memory_ppr') and self.use_in_memory_ppr:
 +            ppr_scores = self.all_ppr_scores[ent]
 +        else:
 +            ppr_scores = np.array(list(self.getPPRscores(ent).values()))
          
 -        # gurantee the candidates are sampled
 -        if cand != None and self.topk < self.n_ent:
 -            tmp_ppr_scores = copy.deepcopy(ppr_scores)
 -            tmp_ppr_scores[cand] = 1e8
 -            topk_nodes = sorted(list(set([ent] + np.argsort(tmp_ppr_scores)[::-1][:self.topk].tolist())))
 +        # Use learned pruning if enabled
 +        if hasattr(self.args, 'use_learned_pruning') and self.args.use_learned_pruning and self.topk < self.n_ent:
 +            if rel is None:
 +                raise ValueError("Learned pruning requires query relation 'rel' during sampling!")
 +            
 +            pool_size = int(0.10 * self.n_ent)
 +            pool_size = max(pool_size, self.topk * 3)
 +            pool_size = min(pool_size, self.n_ent)
 +            candidate_ids_np = np.argsort(ppr_scores)[::-1][:pool_size].copy()
 +            candidate_ids = torch.tensor(candidate_ids_np)
 +            ppr_scores_subset = torch.tensor(ppr_scores[candidate_ids_np])
 +
 +            feats = self.build_features_for_inference(ent, rel, candidate_ids, ppr_scores_subset)
 +            with torch.no_grad():
 +                mlp_scores = self.pruning_model(feats)
 +
 +            k_mlp = min(self.topk // 2, candidate_ids.numel())
 +            _, topk_idx = torch.topk(mlp_scores, k_mlp)
 +            selected_nodes_mlp = candidate_ids[topk_idx].tolist()
 +
 +            k_ppr = self.topk - len(selected_nodes_mlp)
 +            selected_nodes_ppr = np.argsort(ppr_scores)[::-1][:k_ppr].tolist()
 +
 +            topk_nodes = sorted(list(set([ent] + selected_nodes_mlp + selected_nodes_ppr)))
          else:
 -            # topk sampling
 -            if self.topk < self.n_ent:    
 -                topk_nodes = sorted(list(set([ent] + np.argsort(ppr_scores)[::-1][:self.topk].tolist())))
 +            # guarantee the candidates are sampled
 +            if cand != None and self.topk < self.n_ent:
 +                tmp_ppr_scores = copy.deepcopy(ppr_scores)
 +                tmp_ppr_scores[cand] = 1e8
 +                topk_nodes = sorted(list(set([ent] + np.argsort(tmp_ppr_scores)[::-1][:self.topk].tolist())))
              else:
 -                # no sampling
 -                topk_nodes = list(range(self.n_ent))
 +                # topk sampling
 +                if self.topk < self.n_ent:    
 +                    topk_nodes = sorted(list(set([ent] + np.argsort(ppr_scores)[::-1][:self.topk].tolist())))
 +                else:
 +                    # no sampling
 +                    topk_nodes = list(range(self.n_ent))
  ```
 
  ### 2. load_data.py (Truyền ID truy vấn)
  ```diff
  @@ -88,7 +88,11 @@ class DataLoader(Dataset):
               obj[answer[idx]] = 1
                       
           # subgraph sampling
  -        subgraph = self.getOneSubgraph(int(sub))
  +        if self.mode == 'train':
  +            sub_id, rel_id = self.train_data[idx][0], self.train_data[idx][1]
  +        else:
  +            sub_id, rel_id = int(sub.item()), int(rel.item())
  +        subgraph = self.getOneSubgraph(sub_id, rel_id)
           return sub, rel, obj, subgraph
  ```
 
  ### 3. model.py (Tối ưu hóa GNNLayer chiếu & Gradient Checkpointing & Dynamic Embedding)
  ```diff
  @@ -1,16 +1,19 @@
   import torch
   import torch.nn as nn
   from torch_scatter import scatter
  +import torch.nn.functional as F
  +from torch.utils.checkpoint import checkpoint
   
   class GNNLayer(torch.nn.Module):
  -    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x):
  +    def __init__(self, in_dim, out_dim, attn_dim, n_rel, act=lambda x:x, add_manual_edges=False):
           super(GNNLayer, self).__init__()
           self.n_rel = n_rel
           self.in_dim = in_dim
           self.out_dim = out_dim
           self.attn_dim = attn_dim
           self.act = act
  -        self.rela_embed = nn.Embedding(2*n_rel+1, in_dim)
  +        n_embed = 2*n_rel+3 if add_manual_edges else 2*n_rel+1
  +        self.rela_embed = nn.Embedding(n_embed, in_dim)
           self.Ws_attn = nn.Linear(in_dim, attn_dim, bias=False)
           self.Wr_attn = nn.Linear(in_dim, attn_dim, bias=False)
           self.Wqr_attn = nn.Linear(in_dim, attn_dim)
  @@ -22,13 +25,19 @@ class GNNLayer(torch.nn.Module):
           sub = edges[:,0]
           rel = edges[:,1]
           obj = edges[:,2]
  -        hs = hidden[sub]
  -        hr = self.rela_embed(rel) # relation embedding of each edge
  -        h_qr = self.rela_embed(q_rel)[r_idx] # use batch_idx to get the query relation
  +        
  +        # Mathematically equivalent projection optimizations to save massive GPU memory
  +        ws_proj = self.Ws_attn(hidden)[sub]
  +        wr_weight = self.Wr_attn(self.rela_embed.weight)
  +        wr_proj = F.embedding(rel, wr_weight)
  +        wqr_proj = self.Wqr_attn(self.rela_embed(q_rel))[r_idx]
           
           # message aggregation
  +        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(ws_proj + wr_proj + wqr_proj)))
  +        
  +        hs = hidden[sub]
  +        hr = self.rela_embed(rel)
           message = hs * hr
  -        alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
           message = alpha * message        
           message_agg = scatter(message, index=obj, dim=0, dim_size=n_node, reduce='sum') #ori
  @@ -39,6 +48,24 @@ class GNNLayer(torch.nn.Module):
           
           return hidden_new
   
  +class PropagationCell(nn.Module):
  +    def __init__(self, gnn_layer, gate, dropout, shortcut):
  +        super(PropagationCell, self).__init__()
  +        self.gnn_layer = gnn_layer
  +        self.gate = gate
  +        self.dropout = dropout
  +        self.shortcut = shortcut
  +
  +    def forward(self, hidden, h0, q_sub, q_rel, edge_batch_idxs, batch_sampled_edges, n_node):
  +        hidden = self.gnn_layer(q_sub, q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node, shortcut=self.shortcut)
  +        act_signal = (hidden.sum(-1) == 0).detach().int()
  +        hidden = self.dropout(hidden)
  +        hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
  +        hidden = hidden.squeeze(0)
  +        hidden = hidden * (1-act_signal).unsqueeze(-1)
  +        h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
  +        return hidden, h0
  ```
  
  *   **Lan truyền GRU qua Checkpointing**
  ```diff
  @@ -85,18 +119,17 @@ class GNN_auto(torch.nn.Module):
           
           # propagation
           for i in range(self.n_layer):
  -            # forward
  -            hidden = self.gnn_layers[i](q_sub, q_rel, edge_batch_idxs, hidden, batch_sampled_edges, n_node,
  -                                        shortcut=self.params.shortcut)
  -            
  -            # act_signal is a binary (0/1) tensor 
  -            # that 1 for non-activated entities and 0 for activated entities
  -            act_signal = (hidden.sum(-1) == 0).detach().int()
  -            hidden = self.dropout(hidden)
  -            hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
  -            hidden = hidden.squeeze(0)
  -            hidden = hidden * (1-act_signal).unsqueeze(-1)
  -            h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
  +            if self.training:
  +                # Use gradient checkpointing to save VRAM during training
  +                hidden, h0 = checkpoint(
  +                    self.cells[i],
  +                    hidden, h0, q_sub, q_rel, edge_batch_idxs, batch_sampled_edges, n_node,
  +                    use_reentrant=False
  +                )
  +            else:
  +                hidden, h0 = self.cells[i](
  +                    hidden, h0, q_sub, q_rel, edge_batch_idxs, batch_sampled_edges, n_node
  +                )
  ```

### 4. base_model.py (Tích hợp AMP FP16, logs chi tiết & fix warning PyTorch 2.6+)
```diff
diff --git a/base_model.py b/base_model.py
index 2aadd70..7048b13 100644
--- a/base_model.py
+++ b/base_model.py
@@ -12,6 +12,10 @@ from collections import defaultdict
 import torch.nn.functional as F
 import copy
 
+def worker_init_fn(worker_id):
+    import os
+    os.environ["CUDA_VISIBLE_DEVICES"] = ""
+
 class BaseModel(object):
     def __init__(self, args, loaders, samplers):
         self.args = args
@@ -23,20 +27,32 @@ class BaseModel(object):
         self.n_samp_ent = args.n_samp_ent
         self.n_rel = loader.n_rel
         self.train_sampler, self.test_sampler = samplers
-        self.trainLoader = DataLoader(loader, batch_size=args.n_batch, num_workers=args.cpu, collate_fn=loader.collate_fn, shuffle=False, prefetch_factor=args.cpu, pin_memory=True)
-        self.valLoader = DataLoader(val_loader, batch_size=args.n_tbatch, num_workers=args.cpu, collate_fn=val_loader.collate_fn, shuffle=False, prefetch_factor=args.cpu, pin_memory=True)
-        self.testLoader = DataLoader(test_loader, batch_size=args.n_tbatch, num_workers=args.cpu, collate_fn=test_loader.collate_fn, shuffle=False, prefetch_factor=args.cpu, pin_memory=True)
+        prefetch = args.cpu if args.cpu > 0 else None
+        self.trainLoader = DataLoader(loader, batch_size=args.n_batch, num_workers=args.cpu, collate_fn=loader.collate_fn, shuffle=False, prefetch_factor=prefetch, pin_memory=False, worker_init_fn=worker_init_fn)
+        self.valLoader = DataLoader(val_loader, batch_size=args.n_tbatch, num_workers=args.cpu, collate_fn=val_loader.collate_fn, shuffle=False, prefetch_factor=prefetch, pin_memory=False, worker_init_fn=worker_init_fn)
+        self.testLoader = DataLoader(test_loader, batch_size=args.n_tbatch, num_workers=args.cpu, collate_fn=test_loader.collate_fn, shuffle=False, prefetch_factor=prefetch, pin_memory=False, worker_init_fn=worker_init_fn)
         self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
-        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, min_lr=args.lr/20, verbose=True)
+        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, min_lr=args.lr/20)
         self.smooth = 1e-5
-        self.t_time = 0
+        self.train_time = 0.0
+        self.t_time = 0.0
         self.mean_rank_dict = {}
-        
+        self.scaler = torch.amp.GradScaler('cuda')
+
+    def _cuda_peak_memory_mb(self):
+        if torch.cuda.is_available():
+            return torch.cuda.max_memory_allocated() / (1024 ** 2)
+        return 0.0
+
+    def _reset_cuda_peak_memory(self):
+        if torch.cuda.is_available():
+            torch.cuda.reset_peak_memory_stats()
+
     def saveModelToFiles(self, args, best_metric, deleteLastFile=True):
+        suffix = '_mlp' if (hasattr(args, 'use_learned_pruning') and args.use_learned_pruning) else ''
         if args.val_num == -1:
-            savePath = f'{self.args.data_path}/saveModel/topk_{self.args.topk}_layer_{self.args.layer}_{best_metric}.pt'
+            savePath = f'{self.args.data_path}/saveModel/topk_{self.args.topk}_layer_{self.args.layer}_{best_metric}_seed{self.args.seed}{suffix}.pt'
         else:
-            savePath = f'{self.args.data_path}/saveModel/topk_{self.args.topk}_layer_{self.args.layer}_valNum_{self.args.val_num}_{best_metric}.pt'
+            savePath = f'{self.args.data_path}/saveModel/topk_{self.args.topk}_layer_{self.args.layer}_valNum_{self.args.val_num}_{best_metric}_seed{self.args.seed}{suffix}.pt'
@@ -64,39 +80,42 @@ class BaseModel(object):
     def train_batch(self,):        
         epoch_loss = 0
         reach_tails_list = []
-        t_time = time.time()
+        self._reset_cuda_peak_memory()
+        if torch.cuda.is_available():
+            torch.cuda.synchronize()
+        t_time = time.perf_counter()
         self.model.train()
         
         for batch_data in tqdm(self.trainLoader, ncols=50, leave=False):                      
             # prepare data    
             subs, rels, objs, subgraph_data = self.prepareData(batch_data)
             
-            # forward
+            # forward with autocast
             self.model.zero_grad()
-            scores = self.model(subs, rels, subgraph_data)
-            
-            # loss calculation
-            pos_scores = scores[[torch.arange(len(scores)).cuda(), objs.flatten()]]
-            max_n = torch.max(scores, 1, keepdim=True)[0]
-            loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1))) 
-
-            # loss backward
-            loss.backward()
-            self.optimizer.step()
+            with torch.amp.autocast('cuda'):
+                scores = self.model(subs, rels, subgraph_data)
+                
+                # loss calculation
+                pos_scores = scores[torch.arange(len(scores)).cuda(), objs.flatten()]
+                max_n = torch.max(scores, 1, keepdim=True)[0]
+                loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1))) 
 
-            # avoid NaN
-            # for p in self.model.parameters():
-            #     X = p.data.clone()
-            #     flag = X != X
-            #     X[flag] = np.random.random()
-            #     p.data.copy_(X)
+            # loss backward with scaler
+            self.scaler.scale(loss).backward()
+            self.scaler.step(self.optimizer)
+            self.scaler.update()
 
             # cover tail entity or not
             reach_tails = (pos_scores == 0).detach().int().reshape(-1).cpu().tolist()
             reach_tails_list += reach_tails
             epoch_loss += loss.item()
             
-        self.t_time += time.time() - t_time
+        if torch.cuda.is_available():
+            torch.cuda.synchronize()
+        train_latency_ms = (time.perf_counter() - t_time) * 1000.0
+        train_peak_mem_mb = self._cuda_peak_memory_mb()
+        self.train_time += train_latency_ms / 1000.0
+        self.t_time = self.train_time
         
         # evaluate on val/test set
         valid_mrr, out_str = self.evaluate()    
@@ -110,13 +129,19 @@ class BaseModel(object):
             fact_data = np.concatenate([np.array(self.loader.fact_data), self.loader.idd_data], 0)
             self.train_sampler.updateEdges(fact_data)
         
-        return valid_mrr, out_str
+        return valid_mrr, f'[TRAIN] latency_ms:{train_latency_ms:.2f} peak_gpu_mem_mb:{train_peak_mem_mb:.2f}\t' + out_str
     
     @torch.no_grad()
     def evaluate(self, eval_val=True, eval_test=True, verbose=False, rank_CR=False, mean_rank=False):
         ranking = []
         self.model.eval()
-        i_time = time.time()
+        self._reset_cuda_peak_memory()
+        if torch.cuda.is_available():
+            torch.cuda.synchronize()
+        i_time = time.perf_counter()
+        data_prep_ms = 0.0
+        forward_ms = 0.0
+        ranking_ms = 0.0
         
         # eval on val set
         if eval_val:
@@ -124,12 +149,22 @@ class BaseModel(object):
             if mean_rank: mean_rank_list = []
             for batch_data in tqdm(self.valLoader, ncols=50, leave=False):      
                 # prepare data            
+                prep_t0 = time.perf_counter()
                 subs, rels, objs, subgraph_data = self.prepareData(batch_data)
+                data_prep_ms += (time.perf_counter() - prep_t0) * 1000.0
                 
-                # forward
-                scores = self.model(subs, rels, subgraph_data, mode='valid').data.cpu().numpy()
+                # forward with autocast
+                if torch.cuda.is_available():
+                    torch.cuda.synchronize()
+                fwd_t0 = time.perf_counter()
+                with torch.amp.autocast('cuda'):
+                    scores = self.model(subs, rels, subgraph_data, mode='valid').float().data.cpu().numpy()
+                if torch.cuda.is_available():
+                    torch.cuda.synchronize()
+                forward_ms += (time.perf_counter() - fwd_t0) * 1000.0
 
                 # calculate rank
+                rank_t0 = time.perf_counter()
                 subs = subs.cpu().numpy()
                 rels = rels.cpu().numpy()
                 objs = objs.cpu().numpy()
@@ -152,6 +187,7 @@ class BaseModel(object):
                 ans_score = scores[ans].reshape(-1)
                 reach_tails = (ans_score == 0).astype(int).tolist() # (0/1)
                 val_reach_tails_list += reach_tails
+                ranking_ms += (time.perf_counter() - rank_t0) * 1000.0
 
             ranking = np.array(ranking)
             v_mrr, v_h1, v_h10 = cal_performance(ranking)
@@ -179,12 +215,22 @@ class BaseModel(object):
             if mean_rank: mean_rank_list = []
             for batch_data in tqdm(self.testLoader, ncols=50, leave=False):        
                 # prepare data            
+                prep_t0 = time.perf_counter()
                 subs, rels, objs, subgraph_data = self.prepareData(batch_data)
+                data_prep_ms += (time.perf_counter() - prep_t0) * 1000.0
                 
-                # forward
-                scores = self.model(subs, rels, subgraph_data, mode='test').data.cpu().numpy()
+                # forward with autocast
+                if torch.cuda.is_available():
+                    torch.cuda.synchronize()
+                fwd_t0 = time.perf_counter()
+                with torch.amp.autocast('cuda'):
+                    scores = self.model(subs, rels, subgraph_data, mode='test').float().data.cpu().numpy()
+                if torch.cuda.is_available():
+                    torch.cuda.synchronize()
+                forward_ms += (time.perf_counter() - fwd_t0) * 1000.0
 
                 # calculate rank
+                rank_t0 = time.perf_counter()
                 subs = subs.cpu().numpy()
                 rels = rels.cpu().numpy()
                 objs = objs.cpu().numpy()
@@ -207,6 +253,7 @@ class BaseModel(object):
                 ans_score = scores[ans].reshape(-1)
                 reach_tails = (ans_score == 0).astype(int).tolist() # (0/1)
                 test_reach_tails_list += reach_tails
+                ranking_ms += (time.perf_counter() - rank_t0) * 1000.0
 
             ranking = np.array(ranking)
             t_mrr, t_h1, t_h10 = cal_performance(ranking)
@@ -227,6 +274,9 @@ class BaseModel(object):
         else:
             t_mrr, t_h1, t_h10 = -1, -1, -1
             
-        i_time = time.time() - i_time
-        out_str = '[VALID] MRR:%.4f H@1:%.4f H@10:%.4f\t [TEST] MRR:%.4f H@1:%.4f H@10:%.4f \t[TIME] train:%.4f inference:%.4f\n'%(v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.t_time, i_time)
+        if torch.cuda.is_available():
+            torch.cuda.synchronize()
+        eval_latency_ms = (time.perf_counter() - i_time) * 1000.0
+        eval_peak_mem_mb = self._cuda_peak_memory_mb()
+        out_str = '[VALID] MRR:%.6f H@1:%.6f H@10:%.6f\t [TEST] MRR:%.6f H@1:%.6f H@10:%.6f \t[TIME] train:%.4f inference:%.4f \t[LATENCY] eval_total_ms:%.2f data_prep_ms:%.2f forward_ms:%.2f ranking_ms:%.2f \t[PEAK_GPU_MEM] %.2fMB\n'%(v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.train_time, eval_latency_ms / 1000.0, eval_latency_ms, data_prep_ms, forward_ms, ranking_ms, eval_peak_mem_mb)
         return v_mrr, out_str
```

### 5. train_auto.py (Sửa đổi kiểu Seed & Tự động Git Push & Tham số mới)
```diff
diff --git a/train_auto.py b/train_auto.py
index 71cfa17..02e85be 100644
--- a/train_auto.py
+++ b/train_auto.py
@@ -8,9 +8,37 @@ from base_model import BaseModel
 from utils import *
 from PPR_sampler import pprSampler
 
+def git_push_update(message="Auto-commit checkpoints and logs"):
+    import subprocess
+    try:
+        res = subprocess.run(["git", "remote"], capture_output=True, text=True)
+        if not res.stdout.strip():
+            print("==> Git remote not found. Skipping auto-push.")
+            return
+            
+        subprocess.run(["git", "add", "data/WN18RR/results/", "data/nell/results/", "data/YAGO/results/", "data/WN18RR/saveModel/", "data/nell/saveModel/", "data/YAGO/saveModel/", "changes_summary.md", "README.md", "model.py", "base_model.py", "PPR_sampler.py", "train_auto.py", "search_auto.py", ".gitignore"], capture_output=True)
+        
+        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
+        if not status.stdout.strip():
+            return 
+            
+        subprocess.run(["git", "commit", "-m", message], capture_output=True)
+        push_res = subprocess.run(["git", "push"], capture_output=True, text=True)
+        if push_res.returncode == 0:
+            print("==> Successfully pushed checkpoints and logs to GitHub!")
+        else:
+            print("==> Git push failed. Please configure credentials.")
+    except Exception as e:
+        print(f"==> Error during git auto-push: {e}")
+
 parser = argparse.ArgumentParser(description="Parser for the one-shot-subgraph framework")
 parser.add_argument('--data_path', type=str, default='data/WN18RR/')
-parser.add_argument('--seed', type=str, default=1234)
+parser.add_argument('--seed', type=int, default=1234)
 parser.add_argument('--topk', type=float, default=0.1) 
 parser.add_argument('--topm', type=float, default=-1) 
 parser.add_argument('--gpu', type=int, default=0)
@@ -19,12 +47,14 @@ parser.add_argument('--val_num', type=int, default=-1)
 parser.add_argument('--epoch', type=int, default=200)
 parser.add_argument('--layer', type=int, default=6)
 parser.add_argument('--batchsize', type=int, default=16)
-parser.add_argument('--cpu', type=int, default=1)
+parser.add_argument('--cpu', type=int, default=8)
 parser.add_argument('--weight', type=str, default='')
 parser.add_argument('--add_manual_edges', action='store_true')
 parser.add_argument('--remove_1hop_edges', action='store_true')
 parser.add_argument('--only_eval', action='store_true')
 parser.add_argument('--not_shuffle_train', action='store_true')
+parser.add_argument('--use_learned_pruning', action='store_true')
+parser.add_argument('--pruning_model_path', type=str, default='')
 args = parser.parse_args()
 
 class Options(object):
@@ -43,15 +73,13 @@ if __name__ == '__main__':
     else:
         dataset = dataset[-2]
     
-    results_dir = 'results'
+    results_dir = os.path.join(args.data_path, 'results')
     if not os.path.exists(results_dir):
         os.makedirs(results_dir)
-    if not os.path.exists(os.path.join(results_dir, dataset)):
-        os.makedirs(os.path.join(results_dir, dataset))
 
     opts = args
-    time = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
-    opts.perf_file = os.path.join(results_dir,  dataset + '/' + time + '.txt')
+    time_str = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
+    opts.perf_file = os.path.join(results_dir, time_str + '.txt')
     gpu = args.gpu
     torch.cuda.set_device(gpu)
     print('==> gpu:', gpu)
@@ -94,12 +122,17 @@ if __name__ == '__main__':
     test_loader.addSampler(test_sampler)
     
-    checkPath('./results/')
-    checkPath(f'./results/{dataset}/')
+    checkPath(results_dir)
     checkPath(f'{args.data_path}/saveModel/')
             
     def run_model(params):       
         print('==> start training...')    
+        if args.weight != '':
+            import re
+            m = re.search(r'layer_(\d+)', args.weight)
+            if m:
+                params['n_layer'] = int(m.group(1))
+                print(f"==> Automatically set layer={params['n_layer']} from weight path.")
         print(params)
         args.lr = params['lr']
         args.decay_rate = params['decay_rate']
@@ -144,6 +177,7 @@ if __name__ == '__main__':
                 BestMetricStr = f'ValMRR_{str(mrr)[:5]}'
                 model.saveModelToFiles(args, BestMetricStr, deleteLastFile=False)
+                git_push_update(f"Saved best checkpoint ValMRR_{str(mrr)[:5]} at epoch {epoch}")
             else:
                 bearing += 1
                 
@@ -153,6 +187,7 @@ if __name__ == '__main__':
                 break
         
         print(best_str)
+        git_push_update(f"Training finished. Best ValMRR: {best_mrr}")
         return best_mrr
```

### 6. search_auto.py (Sửa đổi kiểu Seed & Sắp xếp lại đường dẫn lưu kết quả)
```diff
diff --git a/search_auto.py b/search_auto.py
index d0c8021..ae50e3b 100644
--- a/search_auto.py
+++ b/search_auto.py
@@ -30,7 +30,7 @@ HPO_search_space = {
 
 parser = argparse.ArgumentParser(description="Parser")
 parser.add_argument('--data_path', type=str, default='data/WN18RR/')
-parser.add_argument('--seed', type=str, default=1234)
+parser.add_argument('--seed', type=int, default=1234)
 parser.add_argument('--topk', type=float, default=0.1) 
 parser.add_argument('--topm', type=float, default=-1) 
 parser.add_argument('--gpu', type=int, default=0)
@@ -38,7 +38,7 @@ parser.add_argument('--fact_ratio', type=float, default=0.75)
 parser.add_argument('--val_num', type=int, default=-1)
 parser.add_argument('--epoch', type=int, default=200)
 parser.add_argument('--batchsize', type=int, default=16)
-parser.add_argument('--cpu', type=int, default=1)
+parser.add_argument('--cpu', type=int, default=8)
 parser.add_argument('--weight', type=str, default='')
 parser.add_argument('--add_manual_edges', action='store_true')
 parser.add_argument('--remove_1hop_edges', action='store_true')
@@ -63,18 +63,15 @@ if __name__ == '__main__':
     args.dataset = dataset
     
-    checkPath('./results/')
-    checkPath(f'./results/{dataset}/')
+    checkPath(results_dir)
     checkPath(f'{args.data_path}/saveModel/')
     
-    results_dir = 'results'
+    results_dir = os.path.join(args.data_path, 'results')
     if not os.path.exists(results_dir):
         os.makedirs(results_dir)
-    if not os.path.exists(os.path.join(results_dir, dataset)):
-        os.makedirs(os.path.join(results_dir, dataset))
             
-    time = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
-    args.perf_file = os.path.join(results_dir,  dataset + '/' + time + '.txt')
+    time_str = str(time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()))
+    args.perf_file = os.path.join(results_dir, time_str + '.txt')
     gpu = args.gpu
     torch.cuda.set_device(gpu)
     print('==> gpu:', gpu)
@@ -114,7 +111,7 @@ if __name__ == '__main__':
     loader.addSampler(train_sampler)
     val_loader.addSampler(test_sampler)
     test_loader.addSampler(test_sampler)
-    HPO_save_path = f'./results/{dataset}/search_log.pkl'
+    HPO_save_path = os.path.join(results_dir, 'search_log.pkl')
```

---

## III. NỘI DUNG ĐẦY ĐỦ CÁC TỆP TIN MỚI THÊM (NEW FILES)

### 1. learned_pruning.py
```python
"""
learned_pruning.py
====================
Learned pruning cho PIVOT (Tuan 9 trong ke hoach).

Y TUONG
-------
Paper goc chon node cho subgraph BANG MOT CONG THUC CO DINH, khong hoc:
        V_s = TopK(V, ppr_score, K = budget * |V|)
PPR khong biet gi ve relation q cu the cua tung query -- no chi nhin
cau truc do thi quanh u. Vi vay PPR co the bo lo nhung entity it duyet
qua (PPR thap) nhung lai chinh xac la cau tra loi dung cho mot relation
hiem.

PIVOT thay THUAT TOAN XEP HANG (khong thay viec sinh candidate) bang
mot MLP nho, query-aware:

    PPR (chi de sinh candidate POOL, vi du top 20% theo PPR)
        -> trich dac trung cho moi candidate
        -> MLP nho cho diem lai
        -> giu top-K theo diem MLP (K = budget thuc te, vi du 1/5/10%)

PPR van duoc dung (re, khong can hoc, lam "luoi an toan" sinh candidate
pool), nhung BUOC CAT CUOI CUNG gio la hoc duoc va biet ve query.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def build_candidate_features(
    candidate_ids,      # LongTensor [N]   id cua N candidate entity
    ppr_scores,         # FloatTensor [N]  diem PPR cua tung candidate (tu u)
    node_degree,        # FloatTensor [N]  degree cua node trong toan KG
    hop_distance,       # FloatTensor [N]  khoang cach BFS tu u (so hop)
    rel_match_score,    # FloatTensor [N]  do "khop" giua relation q va cac
                        #                  relation ke can candidate nay
):
    feats = torch.stack(
        [ppr_scores, torch.log1p(node_degree), hop_distance.float(), rel_match_score],
        dim=1,
    )
    feats = (feats - feats.mean(0, keepdim=True)) / (feats.std(0, keepdim=True) + 1e-6)
    return feats

class PruningMLP(nn.Module):
    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feats):
        return self.net(feats).squeeze(-1)

def listwise_bce_loss(scores, true_tail_mask):
    return F.binary_cross_entropy_with_logits(scores, true_tail_mask.float())

def pairwise_hinge_loss(scores, true_tail_idx, margin=1.0, n_negatives=20):
    true_score = scores[true_tail_idx]
    neg_idx = torch.randperm(scores.numel())[:n_negatives]
    neg_idx = neg_idx[neg_idx != true_tail_idx]
    if neg_idx.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)
    neg_scores = scores[neg_idx]
    loss = F.relu(margin - (true_score - neg_scores)).mean()
    return loss

def train_pruning_model(model, optimizer, query_batches, n_epochs=20, use_pairwise=False):
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in query_batches:
            optimizer.zero_grad()
            scores = model(batch["features"])
            if use_pairwise:
                loss = pairwise_hinge_loss(scores, batch["true_tail_idx"])
            else:
                mask = torch.zeros_like(scores)
                mask[batch["true_tail_idx"]] = 1.0
                loss = listwise_bce_loss(scores, mask)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[learned_pruning] epoch {epoch + 1}/{n_epochs}  loss={total_loss:.4f}")
    return model

@torch.no_grad()
def prune_candidates(model, candidate_ids, features, budget_k, use_learned=True,
                      ppr_scores=None):
    if use_learned:
        model.eval()
        scores = model(features)
    else:
        assert ppr_scores is not None
        scores = ppr_scores

    k = min(budget_k, candidate_ids.numel())
    _, topk_idx = torch.topk(scores, k)
    return candidate_ids[topk_idx]

def recall_at_k_after_pruning(model, eval_queries, budget_k, use_learned=True):
    hits = 0
    for q in eval_queries:
        kept_ids = prune_candidates(
            model, q["candidate_ids"], q["features"], budget_k,
            use_learned=use_learned,
            ppr_scores=q["features"][:, 0] if not use_learned else None,
        )
        true_id = q["candidate_ids"][q["true_tail_idx"]]
        if true_id in kept_ids:
            hits += 1
    return hits / len(eval_queries)
```

### 2. run_learned_pruning_wn18rr.py
```python
"""
run_learned_pruning_wn18rr.py (Multi-seed & PPR Pool Coverage)
========================================================================
Huấn luyện và đánh giá mô hình PruningMLP với các hạt giống (seeds) khác nhau.
"""

import os
import time
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm

from load_data import DataLoader
from PPR_sampler import pprSampler
from learned_pruning import PruningMLP, prune_candidates

log_dir = "./data/WN18RR/budget_results"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "pruning_mlp_v2.log")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PruningMLP_v3")

def get_hop_distances(adj, u, max_hops=3):
    dist = {u: 0}
    queue = [u]
    head = 0
    while head < len(queue):
        curr = queue[head]
        head += 1
        curr_dist = dist[curr]
        if curr_dist >= max_hops:
            continue
        for neighbor in adj[curr]:
            if neighbor not in dist:
                dist[neighbor] = curr_dist + 1
                queue.append(neighbor)
    return dist

def sample_negatives(ppr_scores, true_tail_idx, n_hard=30, n_random=20):
    N = ppr_scores.numel()
    sorted_idx = torch.argsort(ppr_scores, descending=True)
    mask = sorted_idx != true_tail_idx
    sorted_idx = sorted_idx[mask]

    n_hard_actual = min(n_hard, sorted_idx.numel())
    hard_neg = sorted_idx[:n_hard_actual]

    remaining = sorted_idx[n_hard_actual:]
    n_random_actual = min(n_random, remaining.numel())
    if n_random_actual > 0:
        perm = torch.randperm(remaining.numel(), device=remaining.device)[:n_random_actual]
        random_neg = remaining[perm]
        neg_indices = torch.cat([hard_neg, random_neg])
    else:
        neg_indices = hard_neg
    return neg_indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/WN18RR/')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--topk', type=float, default=0.1)
    parser.add_argument('--topm', type=float, default=-1)
    parser.add_argument('--cpu', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lambda_rank', type=float, default=0.4)
    parser.add_argument('--n_hard_neg', type=int, default=30)
    parser.add_argument('--n_random_neg', type=int, default=20)
    parser.add_argument('--hinge_margin', type=float, default=1.0)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)

    logger.info(f"Arguments: {args}")
    logger.info("=" * 70)
    logger.info("SPEC COMPLIANCE: 7 features, 64->32 architecture, hard-neg mining, multi-seed evaluation")
    logger.info("=" * 70)

    logger.info("Loading WN18RR dataset...")
    class MockArgs:
        pass
    loader_args = MockArgs()
    loader_args.data_path = args.data_path
    loader_args.cpu = args.cpu
    loader_args.topk = args.topk
    loader_args.topm = args.topm
    loader_args.fact_ratio = 0.75
    loader_args.remove_1hop_edges = False

    loader = DataLoader(loader_args, mode='train')
    val_loader = DataLoader(loader_args, mode='valid')

    n_ent = loader.n_ent
    n_rel = loader.n_rel
    logger.info(f"KG stats: n_ent={n_ent}, n_rel={n_rel}")

    logger.info("Initializing PPR Sampler (Pre-loading scores)...")
    loader_args.n_samp_ent = int(args.topk * n_ent)
    loader_args.n_samp_edge = -1
    loader_args.add_manual_edges = False

    fact_homo_edges = list(set([(h, t) for (h, r, t) in loader.fact_data]))
    fact_data = np.concatenate([np.array(loader.fact_data), loader.idd_data], 0)
    train_sampler = pprSampler(n_ent, n_rel, loader_args.n_samp_ent, loader_args.n_samp_edge,
                               fact_homo_edges, fact_data, args.data_path, split='train', args=loader_args)

    logger.info("Building KG adjacency structure...")
    adj = defaultdict(list)
    for h, t in fact_homo_edges:
        adj[h].append(t)
        adj[t].append(h)

    degrees = torch.zeros(n_ent)
    for i in range(n_ent):
        degrees[i] = len(adj[i])

    direct_edges_set = set()
    for h, r, t in loader.fact_data:
        direct_edges_set.add((h, r, t))

    logger.info("Computing tail_freq_for_q statistics...")
    tail_freq = np.zeros((n_ent, 2 * n_rel + 2), dtype=np.float32)
    for h, r, t in loader.fact_data:
        tail_freq[t, r] += 1.0
    rel_total = tail_freq.sum(axis=0, keepdims=True)
    tail_freq_norm = torch.tensor(tail_freq / (rel_total + 1e-8))

    logger.info("Computing rel_match_score statistics...")
    rel_counts = np.zeros((n_ent, 2 * n_rel + 2), dtype=np.float32)
    for h, r, t in loader.fact_data:
        rel_counts[h, r] += 1.0
    row_sums = rel_counts.sum(axis=1, keepdims=True)
    rel_dist = torch.tensor(rel_counts / (row_sums + 1e-8))

    def build_features(u, q, candidate_ids, ppr_scores):
        N = candidate_ids.numel()
        cids_cpu = candidate_ids.cpu()
        ppr_log = torch.log(ppr_scores + 1e-8)
        order = torch.argsort(-ppr_scores)
        ranks = torch.empty(N, device=ppr_scores.device)
        ranks[order] = torch.arange(N, dtype=torch.float32, device=ppr_scores.device)
        ppr_rank_pct = 1.0 - ranks / max(N - 1, 1)
        deg_log = torch.log1p(degrees[cids_cpu]).to(ppr_scores.device)

        hop_dist = torch.full((N,), 4.0, device=ppr_scores.device)
        bfs_dists = get_hop_distances(adj, u, max_hops=3)
        for i, cid in enumerate(cids_cpu.tolist()):
            if cid in bfs_dists:
                hop_dist[i] = float(bfs_dists[cid])

        is_direct = torch.zeros(N, device=ppr_scores.device)
        for i, cid in enumerate(cids_cpu.tolist()):
            if (u, q, cid) in direct_edges_set:
                is_direct[i] = 1.0

        tail_freq_q = tail_freq_norm[cids_cpu, q].to(ppr_scores.device)
        rel_match = rel_dist[cids_cpu, q].to(ppr_scores.device)

        feats = torch.stack([
            ppr_log, ppr_rank_pct, deg_log, hop_dist,
            is_direct, tail_freq_q, rel_match
        ], dim=1)
        feats = (feats - feats.mean(0, keepdim=True)) / (feats.std(0, keepdim=True) + 1e-6)
        return feats

    logger.info("Collecting candidate pools and extracting features...")
    def collect_query_data(triples_data, desc, max_queries=3000):
        collected = []
        count = 0
        covered_count = 0
        for h, r, t in tqdm(triples_data, desc=desc, ncols=80):
            ppr_scores_all = train_sampler.all_ppr_scores[h]
            candidate_ids_np = np.argsort(ppr_scores_all)[::-1][:loader_args.n_samp_ent].copy()

            cand_list = candidate_ids_np.tolist()
            is_covered = (t in cand_list)
            if is_covered:
                true_tail_idx = cand_list.index(t)
                candidate_ids = torch.tensor(candidate_ids_np)
                covered_count += 1
            else:
                candidate_ids = torch.tensor(cand_list + [t])
                true_tail_idx = candidate_ids.numel() - 1

            ppr_scores = torch.tensor(train_sampler.all_ppr_scores[h, candidate_ids.tolist()]).cuda()
            candidate_ids = candidate_ids.cuda()
            feats = build_features(h, r, candidate_ids, ppr_scores)

            collected.append({
                "candidate_ids": candidate_ids,
                "features": feats,
                "true_tail_idx": true_tail_idx,
                "ppr_scores": ppr_scores,
                "is_covered": is_covered
            })
            count += 1
            if count >= max_queries:
                break
        coverage = covered_count / count if count > 0 else 0.0
        logger.info(f"{desc} | PPR Candidate Pool (size={loader_args.n_samp_ent}) Coverage: {coverage*100:.2f}% ({covered_count}/{count})")
        return collected, coverage

    train_queries, train_coverage = collect_query_data(loader.train_data, "Collecting Train Queries", max_queries=4000)
    valid_queries, valid_coverage = collect_query_data(val_loader.valid_data, "Collecting Valid Queries", max_queries=1000)

    seeds = [42, 123, 1234]
    budgets = [10, 50, 100, 200, 500]
    seed_results = {}

    for seed in seeds:
        logger.info("=" * 60)
        logger.info(f"RUNNING WITH SEED: {seed}")
        logger.info("=" * 60)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        in_dim = 7
        model = PruningMLP(in_dim=in_dim, hidden=64).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )

        best_recall = 0.0
        patience_counter = 0
        best_model_path = os.path.join(log_dir, f"pruning_mlp_v2_best_seed_{seed}.pt")

        for epoch in range(args.epochs):
            model.train()
            total_loss, total_bce, total_hinge = 0.0, 0.0, 0.0
            indices = np.arange(len(train_queries))
            np.random.shuffle(indices)

            for idx in indices:
                q = train_queries[idx]
                true_idx = q["true_tail_idx"]
                ppr_scores = q["ppr_scores"]
                features = q["features"]

                neg_indices = sample_negatives(
                    ppr_scores, true_idx,
                    n_hard=args.n_hard_neg, n_random=args.n_random_neg
                )

                subset_idx = torch.cat([torch.tensor([true_idx], device=features.device), neg_indices])
                subset_feats = features[subset_idx]
                subset_true_idx = 0

                optimizer.zero_grad()
                scores = model(subset_feats)

                n_neg = neg_indices.numel()
                pos_weight = torch.tensor([min(float(n_neg), 50.0)], device=scores.device)
                labels = torch.zeros_like(scores)
                labels[subset_true_idx] = 1.0
                bce_loss = F.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weight)

                true_score = scores[subset_true_idx]
                neg_scores = scores[1:]
                hinge_loss = F.relu(args.hinge_margin - (true_score - neg_scores)).mean()

                loss = bce_loss + args.lambda_rank * hinge_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_bce += bce_loss.item()
                total_hinge += hinge_loss.item()

            avg_loss = total_loss / len(train_queries)
            avg_bce = total_bce / len(train_queries)
            avg_hinge = total_hinge / len(train_queries)

            model.eval()
            hits = 0
            with torch.no_grad():
                for q in valid_queries:
                    if not q["is_covered"]:
                        continue
                    kept = prune_candidates(model, q["candidate_ids"], q["features"], 100, use_learned=True)
                    true_id = q["candidate_ids"][q["true_tail_idx"]]
                    if true_id in kept:
                        hits += 1
            val_recall = hits / len(valid_queries)
            scheduler.step(val_recall)

            logger.info(
                f"Seed {seed:4d} | Epoch {epoch+1:2d}/{args.epochs}  |  "
                f"Loss: {avg_loss:.4f} (BCE: {avg_bce:.4f}, Hinge: {avg_hinge:.4f})  |  "
                f"Val Realistic R@100: {val_recall:.4f}  |  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_recall > best_recall:
                best_recall = val_recall
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path)
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    logger.info(f"Early stopping for seed {seed} at epoch {epoch+1}")
                    break

        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Seed {seed} | Best Model loaded (Realistic R@100 = {best_recall:.4f})")

        seed_results[seed] = {}
        model.eval()
        with torch.no_grad():
            for k in budgets:
                hits_oracle_mlp, hits_real_mlp = 0, 0
                hits_oracle_ppr, hits_real_ppr = 0, 0

                for q in valid_queries:
                    kept_mlp = prune_candidates(model, q["candidate_ids"], q["features"], k, use_learned=True)
                    true_id = q["candidate_ids"][q["true_tail_idx"]]
                    if true_id in kept_mlp:
                        hits_oracle_mlp += 1
                        if q["is_covered"]:
                            hits_real_mlp += 1

                    kept_ppr = prune_candidates(None, q["candidate_ids"], q["features"], k,
                                                use_learned=False, ppr_scores=q["ppr_scores"])
                    if true_id in kept_ppr:
                        hits_oracle_ppr += 1
                        if q["is_covered"]:
                            hits_real_ppr += 1

                seed_results[seed][k] = {
                    "oracle_mlp": hits_oracle_mlp / len(valid_queries),
                    "realistic_mlp": hits_real_mlp / len(valid_queries),
                    "oracle_ppr": hits_oracle_ppr / len(valid_queries),
                    "realistic_ppr": hits_real_ppr / len(valid_queries),
                }

    logger.info("=" * 70)
    logger.info("                    AGGREGATED MULTI-SEED ANALYSIS                 ")
    logger.info("=" * 70)

    rows = []
    for k in budgets:
        o_mlp_vals = [seed_results[s][k]["oracle_mlp"] for s in seeds]
        r_mlp_vals = [seed_results[s][k]["realistic_mlp"] for s in seeds]
        o_ppr_vals = [seed_results[s][k]["oracle_ppr"] for s in seeds]
        r_ppr_vals = [seed_results[s][k]["realistic_ppr"] for s in seeds]

        o_diff_vals = [o_mlp - o_ppr for o_mlp, o_ppr in zip(o_mlp_vals, o_ppr_vals)]
        r_diff_vals = [r_mlp - r_ppr for r_mlp, r_ppr in zip(r_mlp_vals, r_ppr_vals)]

        rows.append({
            "Budget": k,
            "Oracle MLP Mean": np.mean(o_mlp_vals), "Oracle MLP Std": np.std(o_mlp_vals),
            "Oracle PPR Mean": np.mean(o_ppr_vals), "Oracle PPR Std": np.std(o_ppr_vals),
            "Oracle Delta": np.mean(o_diff_vals),
            "Realistic MLP Mean": np.mean(r_mlp_vals), "Realistic MLP Std": np.std(r_mlp_vals),
            "Realistic PPR Mean": np.mean(r_ppr_vals), "Realistic PPR Std": np.std(r_ppr_vals),
            "Realistic Delta": np.mean(r_diff_vals)
        })

    df_agg = pd.DataFrame(rows)
    csv_out = os.path.join(log_dir, "pruning_mlp_aggregated_summary.csv")
    df_agg.to_csv(csv_out, index=False)

    logger.info(f"Aggregated summary saved to: {csv_out}")
    logger.info(f"PPR Candidate Pool Upper Bound (Validation Coverage): {valid_coverage*100:.2f}%")
    logger.info("-" * 80)
    logger.info("1. REALISTIC EVALUATION")
    for r in rows:
        diff_sign = "+" if r["Realistic Delta"] >= 0 else ""
        logger.info(f"{r['Budget']:8d}  |  {r['Realistic MLP Mean']:.4f} ± {r['Realistic MLP Std']:.4f}       |  {r['Realistic PPR Mean']:.4f} ± {r['Realistic PPR Std']:.4f}        |  {diff_sign}{r['Realistic Delta']:.4f}")
    logger.info("-" * 80)
    logger.info("2. ORACLE EVALUATION")
    for r in rows:
        diff_sign = "+" if r["Oracle Delta"] >= 0 else ""
        logger.info(f"{r['Budget']:8d}  |  {r['Oracle MLP Mean']:.4f} ± {r['Oracle MLP Std']:.4f}       |  {r['Oracle PPR Mean']:.4f} ± {r['Oracle PPR Std']:.4f}        |  {diff_sign}{r['Oracle Delta']:.4f}")
    logger.info("=" * 80)

if __name__ == '__main__':
    main()
```

### 3. budgeted_protocol.py
```python
import argparse
import re
import sys
import subprocess
import time
from pathlib import Path
import pandas as pd

PARSE_REGEX = re.compile(
    r"\[TEST\]\s+MRR:([\d.]+)\s+H@1:([\d.]+)\s+H@10:([\d.]+).*?"
    r"\[LATENCY\]\s+eval_total_ms:([\d.]+)\s+data_prep_ms:([\d.]+)\s+forward_ms:([\d.]+)\s+ranking_ms:([\d.]+).*?"
    r"\[PEAK_GPU_MEM\]\s+([\d.]+)MB",
    re.DOTALL,
)

def run_one(data_path, weight, gpu, topk, batchsize=16, extra_args=None,
            train_script="train_auto.py"):
    cmd = [
        sys.executable, train_script,
        "--data_path", data_path,
        "--batchsize", str(batchsize),
        "--only_eval",
        "--gpu", str(gpu),
        "--topk", str(topk),
        "--topm", "-1",
        "--weight", weight,
    ]
    if extra_args:
        cmd += extra_args

    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall_s = time.time() - t0

    out = proc.stdout + proc.stderr
    m = PARSE_REGEX.search(out)
    if not m:
        raise RuntimeError(
            f"Khong parse duoc output cho topk={topk}.\n"
            f"--- 1500 ky tu cuoi stdout/stderr ---\n{out[-1500:]}"
        )

    mrr, h1, h10, eval_ms, data_prep_ms, forward_ms, ranking_ms, peak_mem = map(float, m.groups())
    return {
        "budget": topk,
        "MRR": mrr,
        "Hits@1": h1,
        "Hits@10": h10,
        "eval_total_ms": eval_ms,
        "data_prep_ms": data_prep_ms,
        "forward_ms": forward_ms,
        "ranking_ms": ranking_ms,
        "peak_gpu_mem_mb": peak_mem,
        "wall_clock_s": wall_s,
    }

def run_budget_sweep(data_path, weight, gpu, budgets, n_queries, seeds=(0,),
                      batchsize=16, out_csv="budget_results.csv",
                      train_script="train_auto.py", use_mlp_pruning=False,
                      dataset_name="WN18RR"):
    rows = []
    for seed in seeds:
        extra = ["--seed", str(seed)] if seed is not None else []
        if use_mlp_pruning:
            pruning_model_path = f"data/{dataset_name}/budget_results/pruning_mlp_v2_best_seed_{seed}.pt"
            extra += ["--use_learned_pruning", "--pruning_model_path", pruning_model_path]
        for b in budgets:
            print(f"[budgeted_protocol] dang chay budget={b} seed={seed} ...")
            r = run_one(data_path, weight, gpu, b, batchsize, extra, train_script)
            r["seed"] = seed
            r["throughput_qps"] = n_queries / (r["eval_total_ms"] / 1000.0)
            r["latency_per_query_ms"] = r["eval_total_ms"] / n_queries
            rows.append(r)
            print(
                f"  -> MRR={r['MRR']:.4f}  "
                f"latency/q={r['latency_per_query_ms']:.2f}ms  "
                f"peak_mem={r['peak_gpu_mem_mb']:.0f}MB"
            )

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nDa luu raw results -> {out_csv}")
    return df

def summarize(df, out_csv="budget_summary.csv", dataset_name="WN18RR", method_name="PPR-only"):
    agg = (
        df.groupby("budget")
        .agg(
            MRR_mean=("MRR", "mean"),
            MRR_std=("MRR", "std"),
            Hits1_mean=("Hits@1", "mean"),
            Hits10_mean=("Hits@10", "mean"),
            eval_total_mean=("eval_total_ms", "mean"),
            data_prep_mean=("data_prep_ms", "mean"),
            forward_mean=("forward_ms", "mean"),
            ranking_mean=("ranking_ms", "mean"),
            latency_mean=("latency_per_query_ms", "mean"),
            latency_std=("latency_per_query_ms", "std"),
            throughput_mean=("throughput_qps", "mean"),
            peak_mem_mean=("peak_gpu_mem_mb", "mean"),
        )
        .reset_index()
        .sort_values("budget")
    )
    
    formatted = agg.copy()
    formatted["Test MRR (Mean ± Std)"] = agg.apply(
        lambda r: f"{r['MRR_mean']:.6f} ± {r['MRR_std']:.6f}" if not pd.isna(r['MRR_std']) else f"{r['MRR_mean']:.6f} ± 0.000000",
        axis=1
    )
    
    formatted = formatted.rename(columns={
        "budget": "Budget",
        "Hits1_mean": "H@1 (Mean)",
        "Hits10_mean": "H@10 (Mean)",
        "eval_total_mean": "eval_total (ms)",
        "data_prep_mean": "data_prep (ms)",
        "forward_mean": "forward (ms)",
        "ranking_mean": "ranking (ms)",
        "latency_mean": "Latency / query (ms)",
        "throughput_mean": "Throughput (q/s)",
        "peak_mem_mean": "Peak GPU Mem (MB)",
    })
    formatted.insert(0, "Dataset", dataset_name)
    formatted.insert(1, "Phương pháp", method_name)
    
    cols_to_keep = [
        "Dataset", "Phương pháp", "Budget", "Test MRR (Mean ± Std)",
        "H@1 (Mean)", "H@10 (Mean)", "eval_total (ms)", "data_prep (ms)",
        "forward (ms)", "ranking (ms)", "Latency / query (ms)",
        "Throughput (q/s)", "Peak GPU Mem (MB)"
    ]
    formatted = formatted[cols_to_keep]
    formatted.to_csv(out_csv, index=False)
    print(formatted.to_string(index=False))
    return agg

def plot_pareto(summary_df, out_png="pareto_frontier.png"):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(
        summary_df["latency_mean"], summary_df["MRR_mean"],
        marker="o", linewidth=1.5,
    )
    for _, row in summary_df.iterrows():
        ax.annotate(
            f"{int(row['budget'] * 100)}%",
            (row["latency_mean"], row["MRR_mean"]),
            textcoords="offset points", xytext=(6, 4), fontsize=9,
        )
    ax.set_xlabel("Latency per query (ms)")
    ax.set_ylabel("MRR")
    ax.set_title("Accuracy-Latency Pareto frontier")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PIVOT budgeted inference benchmark")
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--weight", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n_queries", type=int, required=True)
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.20])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--batchsize", type=int, default=16)
    ap.add_argument("--outdir", default="./budget_results")
    ap.add_argument("--dataset", default="WN18RR")
    ap.add_argument("--method", default="PPR-only")
    ap.add_argument("--train_script", default="train_auto.py")
    ap.add_argument("--use_mlp_pruning", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = run_budget_sweep(
        args.data_path, args.weight, args.gpu, args.budgets,
        args.n_queries, args.seeds, args.batchsize,
        out_csv=str(outdir / "raw_results.csv"),
        train_script=args.train_script,
        use_mlp_pruning=args.use_mlp_pruning,
        dataset_name=args.dataset,
    )
    summary = summarize(df, out_csv=str(outdir / "summary.csv"), dataset_name=args.dataset, method_name=args.method)
    plot_pareto(summary, out_png=str(outdir / "pareto_frontier.png"))
```

---

## IV. TÓM TẮT SO SÁNH CẤU TRÚC TẬP TIN (FILE SYSTEM STRUCTURE SUMMARY)

Dưới đây là sơ đồ so sánh trực quan cấu trúc thư mục dự án ban đầu (Paper) và sau khi nâng cấp tối ưu hóa (PIVOT):

```mermaid
graph TD
    subgraph Repo Gốc (AndrewZhou924/one-shot-subgraph)
        direction TB
        orig1[model.py]
        orig2[load_data.py]
        orig3[PPR_sampler.py]
        orig4[train_auto.py]
        orig5[base_model.py]
    end

    subgraph codebase tối ưu (PIVOT)
        direction TB
        opt1[model.py: gradient checkpointing & dynamic embedding size]
        opt2[load_data.py: query relation propagation]
        opt3[PPR_sampler.py: multiprocessing PPR & loading cache & hybrid sampler]
        opt4[train_auto.py: seed int & git push automation]
        opt5[base_model.py: AMP training & detailed profiling logs]
        
        new1[learned_pruning.py: PruningMLP & loss functions]
        new2[run_learned_pruning_wn18rr.py: HPO & eval script]
        new3[budgeted_protocol.py: Accuracy-Latency Sweep & Pareto frontier]
        new4[changes_summary.md: Chat transfer synchronization document]
    end

    orig1 -.-> opt1
    orig2 -.-> opt2
    orig3 -.-> opt3
    orig4 -.-> opt4
    orig5 -.-> opt5
```

---

## IV. HOÀN NGUYÊN MÃ NGUỒN (REVERT FOR BASELINE) — 2026-07-01

Phiên làm việc này hoàn nguyên **model.py** và **base_model.py** về code gốc của paper để huấn luyện NELL995 làm **mốc cơ sở (baseline)** nhằm so sánh trước/sau cải tiến.

### Mục đích
- Lấy kết quả huấn luyện NELL995 không có cải tiến làm baseline.
- Sau khi có kết quả baseline, sẽ áp dụng lại 2 cải tiến dưới đây và so sánh.

### Các cải tiến đã TẠM THỜI hoàn nguyên (sẽ tái áp dụng sau)

#### 1. model.py — Hoàn nguyên Gradient Checkpointing
```diff
- # PropagationCell wrapper + checkpoint() call
+ # Original inline propagation loop (no PropagationCell)
  for i in range(self.n_layer):
-     if self.training:
-         hidden, h0 = checkpoint(self.cells[i], hidden, h0, ...)
-     else:
-         hidden, h0 = self.cells[i](hidden, h0, ...)
+     hidden = self.gnn_layers[i](q_sub, q_rel, edge_batch_idxs, hidden, ...)
+     act_signal = (hidden.sum(-1) == 0).detach().int()
+     hidden = self.dropout(hidden)
+     hidden, h0 = self.gate(hidden.unsqueeze(0), h0)
+     hidden = hidden.squeeze(0)
+     hidden = hidden * (1-act_signal).unsqueeze(-1)
+     h0 = h0 * (1-act_signal).unsqueeze(-1).unsqueeze(0)
```

#### 2. model.py — Hoàn nguyên Attention Projection Optimization
```diff
- # Optimized: project full embedding matrix first, then index
- ws_proj = self.Ws_attn(hidden)[sub]
- wr_weight = self.Wr_attn(self.rela_embed.weight)
- wr_proj = F.embedding(rel, wr_weight)
- wqr_proj = self.Wqr_attn(self.rela_embed(q_rel))[r_idx]
- alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(ws_proj + wr_proj + wqr_proj)))
- hs = hidden[sub]
- hr = self.rela_embed(rel)
+ # Original: index first, then project
+ hs = hidden[sub]
+ hr = self.rela_embed(rel)
+ h_qr = self.rela_embed(q_rel)[r_idx]
+ alpha = torch.sigmoid(self.w_alpha(nn.ReLU()(self.Ws_attn(hs) + self.Wr_attn(hr) + self.Wqr_attn(h_qr))))
```

#### 3. base_model.py — Hoàn nguyên AMP FP16 + Profiling logs
```diff
- # AMP autocast + GradScaler
- with torch.amp.autocast('cuda'):
-     scores = self.model(...)
- self.scaler.scale(loss).backward()
- self.scaler.step(self.optimizer)
+ # Original: plain forward + backward
+ scores = self.model(...)
+ loss.backward()
+ self.optimizer.step()

- # Removed: detailed latency/peak-memory logging
- # Removed: worker_init_fn
- # Removed: pin_memory=False
+ # Restored: pin_memory=True, original DataLoader settings
```

---

## V. TỐI ƯU HÓA BỘ NHỚ ĐỆM PPR (PPR MEMORY CACHING OPTIMIZATION) — 2026-07-02

### Thay đổi trong `PPR_sampler.py`
Tăng ngưỡng tối đa để nạp toàn bộ điểm PPR vào bộ nhớ RAM (`use_in_memory_ppr`) từ **50,000** lên **80,000**:
```diff
-        if self.n_ent <= 50000:
+        if self.n_ent <= 80000:
```

### Mục đích & Tác động
- **Mục đích**: Hỗ trợ tập dữ liệu **NELL995** (`n_ent = 74,536`) được nạp trực tiếp toàn bộ điểm PPR vào RAM lúc khởi tạo (trước đây chỉ áp dụng cho WN18RR).
- **Tác động**: Tránh việc đọc file pickle `.pkl` từ đĩa và parse sang numpy array liên tục cho mỗi mẫu huấn luyện. Kết hợp với tăng `--cpu` của DataLoader, tốc độ chạy 1 epoch huấn luyện dự kiến giảm từ **50 phút** xuống còn **3-5 phút** mà không làm tăng đáng kể mức tiêu thụ RAM thực tế nhờ cơ chế CoW (Copy-on-Write) của Linux.

### Chia sẻ tài nguyên nạp sẵn (Shared PPR Caching across Samplers)
- **Thay đổi**: Thêm tham số `preloaded_ppr` vào `pprSampler.__init__` và truyền từ `test_sampler` sang `train_sampler` trong `train_auto.py`:
```diff
# Trong PPR_sampler.py
-    def __init__(self, n_ent:int, n_rel:int, topk:int, topm:int, homoEdges:list, edge_index:list, data_path:str, split='train', args=None):
+    def __init__(self, n_ent:int, n_rel:int, topk:int, topm:int, homoEdges:list, edge_index:list, data_path:str, split='train', args=None, preloaded_ppr=None):

# Trong train_auto.py
+    preloaded_ppr = getattr(test_sampler, 'all_ppr_scores', None)
     train_sampler = pprSampler(loader.n_ent, loader.n_rel, args.n_samp_ent, args.n_samp_edge,
-        fact_homo_edges, fact_data, args.data_path, split='train', args=args)
+        fact_homo_edges, fact_data, args.data_path, split='train', args=args, preloaded_ppr=preloaded_ppr)
```
- **Tác động**: Rút ngắn thời gian khởi tạo đi một nửa (giảm 20 phút tải file pickle lặp lại không cần thiết khi khởi tạo train sampler).

