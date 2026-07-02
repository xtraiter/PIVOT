import networkx as nx
import pickle as pkl
import time
import copy
import numpy as np
import torch
import os
import logging
import copy
from tqdm import tqdm
from scipy.sparse import csr_matrix, coo_matrix
from collections import defaultdict

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

def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return

_global_homo_graph = None

def _init_worker(graph):
    global _global_homo_graph
    _global_homo_graph = graph

def _compute_and_save_ppr_scores(h, ppr_save_path):
    import os
    import pickle as pkl
    import networkx as nx
    import tempfile
    ent_ppr_savePath = os.path.join(ppr_save_path, f'{int(h)}.pkl')
    if os.path.exists(ent_ppr_savePath) and os.path.getsize(ent_ppr_savePath) > 1000:
        return
    scores = nx.pagerank(_global_homo_graph, personalization={h: 1})
    
    # Atomic write to prevent corruption if process is killed
    dir_name = os.path.dirname(ent_ppr_savePath)
    with tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix=".tmp") as temp_file:
        pkl.dump(scores, temp_file)
        temp_file_path = temp_file.name
    os.replace(temp_file_path, ent_ppr_savePath)

def _load_one_ppr(h, ppr_save_path):
    import os
    import pickle as pkl
    ent_ppr_savePath = os.path.join(ppr_save_path, f'{int(h)}.pkl')
    if os.path.exists(ent_ppr_savePath):
        try:
            with open(ent_ppr_savePath, 'rb') as f:
                scores = pkl.load(f)
            return h, scores
        except Exception:
            return h, None
    return h, None

class pprSampler():
    def __init__(self, n_ent:int, n_rel:int, topk:int, topm:int, homoEdges:list, edge_index:list, data_path:str, split='train', args=None):
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
        
        import multiprocessing
        from functools import partial

        # Determine number of worker processes
        num_workers = max(1, multiprocessing.cpu_count() - 4) # leave some CPUs free
        num_workers = min(64, num_workers) # avoid excessive process overhead
        print(f'==> Using {num_workers} parallel workers to generate PPR scores...')

        # Filter entities that don't have ppr file yet (or are corrupted/empty)
        entities_to_compute = []
        for h in range(self.n_ent):
            ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(h)}.pkl')
            if not os.path.exists(ent_ppr_savePath) or os.path.getsize(ent_ppr_savePath) < 1000:
                entities_to_compute.append(h)

        if len(entities_to_compute) > 0:
            worker_func = partial(_compute_and_save_ppr_scores, ppr_save_path=self.ppr_savePath)
            with multiprocessing.Pool(
                processes=num_workers,
                initializer=_init_worker,
                initargs=(self.homoTrainGraph,)
            ) as pool:
                list(tqdm(pool.imap_unordered(worker_func, entities_to_compute), 
                          total=len(entities_to_compute), ncols=50, leave=False))
        print('finished.')
        
        # build head to edges with sparse matrix
        heads, edges = [h for (h,r,t) in edge_index], list(range(len(edge_index)))
        print(len(heads), len(edges), max(heads), self.n_ent)
        self.sparseTrainMatrix = csr_matrix((edges, (heads, edges)), shape=(self.n_ent, len(edge_index)))

        # change data type
        self.edge_index = torch.LongTensor(self.edge_index)

        # clean cache
        del self.homoEdges
        del self.homoTrainGraph
        
        self.ppr_cache = {}
        # Pre-load all PPR scores in memory to avoid extremely slow disk I/O / unpickling during training
        if self.n_ent <= 50000:
            if not hasattr(pprSampler, '_global_ppr_scores') or pprSampler._global_ppr_scores is None:
                print(f"==> Pre-loading all {self.n_ent} PPR scores into memory...")
                ppr_scores_matrix = np.zeros((self.n_ent, self.n_ent), dtype=np.float32)
                
                import multiprocessing
                from functools import partial
                num_loader_workers = min(32, multiprocessing.cpu_count())
                worker_func = partial(_load_one_ppr, ppr_save_path=self.ppr_savePath)
                
                with multiprocessing.Pool(processes=num_loader_workers) as pool:
                    for h, scores in tqdm(pool.imap_unordered(worker_func, range(self.n_ent), chunksize=200), total=self.n_ent, desc="Loading PPR scores", ncols=50, leave=False):
                        if scores is not None:
                            for k, v in scores.items():
                                ppr_scores_matrix[h, k] = v
                pprSampler._global_ppr_scores = ppr_scores_matrix
            else:
                print(f"==> Re-using pre-loaded PPR scores from memory cache for split '{split}'")
                
            self.all_ppr_scores = pprSampler._global_ppr_scores
            self.use_in_memory_ppr = True
        else:
            self.use_in_memory_ppr = False

        # build sparse tensor self.PPR_W for matrix-computation PPR
        '''
        tmp_degree, tmp_adj = torch.zeros(self.n_ent, self.n_ent), torch.zeros(self.n_ent, self.n_ent)
        tmp_adj[self.edge_index[:,0], self.edge_index[:,2]] = 1
        tmp_degree = torch.diag(1 / torch.sum(tmp_adj, dim=1))
        self.PPR_W = torch.eye(self.n_ent) + torch.matmul(tmp_degree, tmp_adj)
        self.PPR_W = self.PPR_W.cuda()
        del tmp_adj; del tmp_degree
        '''
        
        if hasattr(self.args, 'use_learned_pruning') and self.args.use_learned_pruning:
            print("==> Initializing MLP Pruning components in sampler...")
            self.adj = defaultdict(list)
            for h, t in homoEdges:
                self.adj[int(h)].append(int(t))
                self.adj[int(t)].append(int(h))

            self.degrees = torch.zeros(self.n_ent)
            for i in range(self.n_ent):
                self.degrees[i] = len(self.adj[i])

            self.direct_edges_set = set()
            for h, r, t in edge_index:
                self.direct_edges_set.add((int(h), int(r), int(t)))

            tail_freq = np.zeros((self.n_ent, 2 * self.n_rel + 2), dtype=np.float32)
            for h, r, t in edge_index:
                tail_freq[int(t), int(r)] += 1.0
            rel_total = tail_freq.sum(axis=0, keepdims=True)
            self.tail_freq_norm = torch.tensor(tail_freq / (rel_total + 1e-8))

            rel_counts = np.zeros((self.n_ent, 2 * self.n_rel + 2), dtype=np.float32)
            for h, r, t in edge_index:
                rel_counts[int(h), int(r)] += 1.0
            row_sums = rel_counts.sum(axis=1, keepdims=True)
            self.rel_dist = torch.tensor(rel_counts / (row_sums + 1e-8))

            from learned_pruning import PruningMLP
            self.pruning_model = PruningMLP(in_dim=7, hidden=64)
            if hasattr(self.args, 'pruning_model_path') and self.args.pruning_model_path:
                print(f"==> Loading learned pruning model from: {self.args.pruning_model_path}")
                self.pruning_model.load_state_dict(torch.load(self.args.pruning_model_path, map_location='cpu'))
            self.pruning_model.eval()

        print('==> finish sampler initilization.')

    def updateEdges(self, edge_index):
        # co-operate with shuffle_train
        heads, edges = [h for (h,r,t) in edge_index], list(range(len(edge_index)))
        self.sparseTrainMatrix = csr_matrix((edges, (heads, edges)), shape=(self.n_ent, len(edge_index)))
        self.edge_index = torch.LongTensor(edge_index)
    
    def getPPRscores(self, ent):
        if ent in self.ppr_cache:
            return self.ppr_cache[ent]
        ent_ppr_savePath = os.path.join(self.ppr_savePath, f'{int(ent)}.pkl')
        scores = pkl.load(open(ent_ppr_savePath, 'rb'))
        self.ppr_cache[ent] = scores
        return scores
        
    def generatePPRScoresForOneEntity(self, h, method='nx'):
        if method == 'nx':
            '''
            nx.pagerank(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-06, nstart=None, weight='weight', dangling=None)
            '''
            scores = nx.pagerank(self.homoTrainGraph, personalization={h: 1})
        elif method == 'matrix':
            alpha, iteration = 0.85, 100
            scores = torch.zeros(1, self.n_ent).cuda()
            s = torch.zeros(1, self.n_ent).cuda()
            s[0, h] = 1
            for i in range(iteration):
                scores = alpha * s + (1 - alpha) * torch.matmul(scores, self.PPR_W)            
            scores = scores.cpu().reshape(-1).numpy()
        return scores
    
    def triplesToNxGraph(self, edges):
        ''' edges is the list of [(h,t)] '''
        graph = nx.Graph()
        nodes = list(range(self.n_ent))
        graph.add_nodes_from(nodes)        
        graph.add_edges_from(edges)
        return graph
    
    def build_features_for_inference(self, u, q, candidate_ids, ppr_scores):
        N = candidate_ids.numel()
        cids_cpu = candidate_ids.cpu()

        ppr_log = torch.log(ppr_scores + 1e-8)

        order = torch.argsort(-ppr_scores)
        ranks = torch.empty(N, device=ppr_scores.device)
        ranks[order] = torch.arange(N, dtype=torch.float32, device=ppr_scores.device)
        ppr_rank_pct = 1.0 - ranks / max(N - 1, 1)

        deg_log = torch.log1p(self.degrees[cids_cpu]).to(ppr_scores.device)

        hop_dist = torch.full((N,), 4.0, device=ppr_scores.device)
        bfs_dists = get_hop_distances(self.adj, u, max_hops=3)
        for i, cid in enumerate(cids_cpu.tolist()):
            if cid in bfs_dists:
                hop_dist[i] = float(bfs_dists[cid])

        is_direct = torch.zeros(N, device=ppr_scores.device)
        for i, cid in enumerate(cids_cpu.tolist()):
            if (u, q, cid) in self.direct_edges_set:
                is_direct[i] = 1.0

        tail_freq_q = self.tail_freq_norm[cids_cpu, q].to(ppr_scores.device)
        rel_match = self.rel_dist[cids_cpu, q].to(ppr_scores.device)

        feats = torch.stack([
            ppr_log, ppr_rank_pct, deg_log, hop_dist,
            is_direct, tail_freq_q, rel_match
        ], dim=1)

        feats = (feats - feats.mean(0, keepdim=True)) / (feats.std(0, keepdim=True) + 1e-6)
        return feats

    def sampleSubgraph(self, ent: int, rel: int = None, cand=None):    
        # sample subgraph to get the edges
        if hasattr(self, 'use_in_memory_ppr') and self.use_in_memory_ppr:
            ppr_scores = self.all_ppr_scores[ent]
        else:
            ppr_scores = np.array(list(self.getPPRscores(ent).values()))
        
        # Use learned pruning if enabled
        if hasattr(self.args, 'use_learned_pruning') and self.args.use_learned_pruning and self.topk < self.n_ent:
            if rel is None:
                raise ValueError("Learned pruning requires query relation 'rel' during sampling!")
            
            # Determine candidate pool size (must be larger than target budget topk)
            pool_size = int(0.10 * self.n_ent)
            pool_size = max(pool_size, self.topk * 3)
            pool_size = min(pool_size, self.n_ent)

            # Extract candidate pool from top-pool_size PPR entities
            candidate_ids_np = np.argsort(ppr_scores)[::-1][:pool_size].copy()
            candidate_ids = torch.tensor(candidate_ids_np)
            ppr_scores_subset = torch.tensor(ppr_scores[candidate_ids_np])

            feats = self.build_features_for_inference(ent, rel, candidate_ids, ppr_scores_subset)
            with torch.no_grad():
                mlp_scores = self.pruning_model(feats)

            # Hybrid Subgraph Selection (Lựa chọn A): 50% MLP + 50% PPR
            k_mlp = min(self.topk // 2, candidate_ids.numel())
            _, topk_idx = torch.topk(mlp_scores, k_mlp)
            selected_nodes_mlp = candidate_ids[topk_idx].tolist()

            # Select top PPR nodes to preserve dense relational reasoning paths
            k_ppr = self.topk - len(selected_nodes_mlp)
            selected_nodes_ppr = np.argsort(ppr_scores)[::-1][:k_ppr].tolist()

            topk_nodes = sorted(list(set([ent] + selected_nodes_mlp + selected_nodes_ppr)))
        else:
            # guarantee the candidates are sampled
            if cand != None and self.topk < self.n_ent:
                tmp_ppr_scores = copy.deepcopy(ppr_scores)
                tmp_ppr_scores[cand] = 1e8
                topk_nodes = sorted(list(set([ent] + np.argsort(tmp_ppr_scores)[::-1][:self.topk].tolist())))
            else:
                # topk sampling
                if self.topk < self.n_ent:    
                    topk_nodes = sorted(list(set([ent] + np.argsort(ppr_scores)[::-1][:self.topk].tolist())))
                else:
                    # no sampling
                    topk_nodes = list(range(self.n_ent))

        # get candididate edges
        selectd_edges = self.sparseTrainMatrix[topk_nodes, :]	
        _, tmp_edge_index = selectd_edges.nonzero()
        
        # (h,r,t)
        edges = self.edge_index[tmp_edge_index]
        topk_nodes = torch.LongTensor(topk_nodes)
        
        # edge sampling
        mask = torch.isin(edges[:,2], topk_nodes)
        
        # [n_edges, 3]
        sampled_edges = edges[mask, :]
        
        # edge sampling (topm edges for each subgraph)
        edge_num = int(sampled_edges.shape[0])
        # NOTE: if self.topm== 0, then skip edge sampling 
        if self.topm > 0 and edge_num > self.topm:
            # ppr weight
            heads, tails = sampled_edges[:,0], sampled_edges[:,2]
            edge_weights = ppr_scores[heads] + ppr_scores[tails]
            edge_weights = torch.Tensor(edge_weights)
            index = torch.topk(edge_weights, self.topm).indices
            sampled_edges = sampled_edges[index]
        
        # get node indexing map
        node_index = torch.zeros(self.n_ent).long()
        node_index[topk_nodes] = torch.arange(len(topk_nodes))
              
        # connect head to all tails 
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

    def getOneSubgraph(self, head: int, rel: int = None, cand=None):
        topk_nodes, node_index, sampled_edges = self.sampleSubgraph(head, rel, cand) 
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
        
        # [n_batch_ent]
        batch_idxs = torch.LongTensor(batch_idxs)
        # [n_batch_ent]
        abs_idxs = torch.LongTensor(abs_idxs)
        # [n_batch_edges, 3]
        batch_sampled_edges = torch.cat(batch_sampled_edges, dim=0)
        # [n_batch_edges]
        edge_batch_idxs = torch.LongTensor(edge_batch_idxs)
        # [n_batch]
        query_sub_idxs = torch.LongTensor(query_sub_idxs)
        
        return batch_idxs, abs_idxs, query_sub_idxs, edge_batch_idxs, batch_sampled_edges