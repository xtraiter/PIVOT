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

def worker_init_fn(worker_id):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

class BaseModel(object):
    def __init__(self, args, loaders, samplers):
        self.args = args
        loader, val_loader, test_loader = loaders
        self.loader = loader
        self.model = GNN_auto(args, loader)
        self.model.cuda()
        if hasattr(args, 'compile') and args.compile:
            print("==> Setting float32 matmul precision to 'high'...")
            torch.set_float32_matmul_precision('high')
            print("==> Compiling GNN model with torch.compile...")
            self.model = torch.compile(self.model)

        self.n_ent = loader.n_ent
        self.n_samp_ent = args.n_samp_ent
        self.n_rel = loader.n_rel
        self.train_sampler, self.test_sampler = samplers
        prefetch = 2 if args.cpu > 0 else None
        self.trainLoader = DataLoader(loader, batch_size=args.n_batch, num_workers=args.cpu, collate_fn=loader.collate_fn, shuffle=False, prefetch_factor=prefetch, pin_memory=False, worker_init_fn=worker_init_fn)
        self.valLoader = DataLoader(val_loader, batch_size=args.n_tbatch, num_workers=args.cpu, collate_fn=val_loader.collate_fn, shuffle=False, prefetch_factor=prefetch, pin_memory=False, worker_init_fn=worker_init_fn)
        self.testLoader = DataLoader(test_loader, batch_size=args.n_tbatch, num_workers=args.cpu, collate_fn=test_loader.collate_fn, shuffle=False, prefetch_factor=prefetch, pin_memory=False, worker_init_fn=worker_init_fn)
        self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.lamb)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=2, min_lr=args.lr/20)
        self.smooth = 1e-5
        self.train_time = 0.0
        self.t_time = 0.0
        self.mean_rank_dict = {}
        self.scaler = torch.amp.GradScaler('cuda')

    def _cuda_peak_memory_mb(self):
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
        return 0.0

    def _reset_cuda_peak_memory(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _post_hoc_rerank(self, scores, subs, rels, subgraph_data):
        if hasattr(self.args, 'rerank_alpha') and self.args.rerank_alpha >= 0.0 and hasattr(self.args, 'use_learned_pruning') and self.args.use_learned_pruning:
            batch_idxs_cpu = subgraph_data[0].cpu()
            abs_idxs_cpu = subgraph_data[1].cpu()
            rels_cpu = rels.cpu()
            subs_cpu = subs.cpu()
            
            mlp_scores_full = torch.zeros_like(torch.from_numpy(scores))  # [n_query, n_ent]
            
            for i in range(len(rels_cpu)):
                mask = (batch_idxs_cpu == i)
                entity_ids = abs_idxs_cpu[mask]
                if len(entity_ids) == 0:
                    continue
                q_rel = int(rels_cpu[i])
                q_sub = int(subs_cpu[i])
                
                if hasattr(self.test_sampler, 'all_ppr_scores') and self.test_sampler.all_ppr_scores is not None:
                    ppr_scores_i = torch.tensor(
                        self.test_sampler.all_ppr_scores[q_sub, entity_ids.numpy()]
                    )
                else:
                    ppr_dict = self.test_sampler.getPPRscores(q_sub)
                    ppr_scores_i = torch.tensor([ppr_dict.get(int(eid), 0.0) for eid in entity_ids])
                feats = self.test_sampler.build_features_for_inference(
                    q_sub, q_rel, entity_ids, ppr_scores_i
                )
                
                with torch.no_grad():
                    mlp_s = self.test_sampler.pruning_model(feats).cpu()
                
                # Normalize to [0, 1]
                mlp_s = (mlp_s - mlp_s.min()) / (mlp_s.max() - mlp_s.min() + 1e-8)
                mlp_scores_full[i, entity_ids] = mlp_s
            
            alpha = self.args.rerank_alpha
            scores_blended = (1 - alpha) * scores + alpha * mlp_scores_full.numpy()
            
            if hasattr(self.args, 'dump_scores') and self.args.dump_scores != '':
                if not hasattr(self, 'dump_gnn_list'):
                    self.dump_gnn_list = []
                    self.dump_mlp_list = []
                self.dump_gnn_list.append(scores.copy())
                self.dump_mlp_list.append(mlp_scores_full.numpy())
            
            scores = scores_blended
        return scores

    def saveModelToFiles(self, args, best_metric, deleteLastFile=True):
        suffix = '_mlp' if (hasattr(args, 'use_learned_pruning') and args.use_learned_pruning) else ''
        if args.val_num == -1:
            savePath = f'{self.args.data_path}/saveModel/topk_{self.args.topk}_layer_{self.args.layer}_{best_metric}_seed{self.args.seed}{suffix}.pt'
        else:
            savePath = f'{self.args.data_path}/saveModel/topk_{self.args.topk}_layer_{self.args.layer}_valNum_{self.args.val_num}_{best_metric}_seed{self.args.seed}{suffix}.pt'
            
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
        if hasattr(self.model, '_orig_mod'):
            self.model._orig_mod.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        # re-build optimizter
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
        self._reset_cuda_peak_memory()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t_time = time.perf_counter()
        self.model.train()
        
        for batch_data in tqdm(self.trainLoader, ncols=50, leave=False):                      
            # prepare data    
            subs, rels, objs, subgraph_data = self.prepareData(batch_data)
            
            # forward with autocast
            self.model.zero_grad()
            if hasattr(self.args, 'no_amp') and self.args.no_amp:
                scores = self.model(subs, rels, subgraph_data)
                
                # loss calculation
                pos_scores = scores[torch.arange(len(scores)).cuda(), objs.flatten()]
                max_n = torch.max(scores, 1, keepdim=True)[0]
                loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1)))
                
                loss.backward()
                self.optimizer.step()
            else:
                with torch.amp.autocast('cuda'):
                    scores = self.model(subs, rels, subgraph_data)
                    
                    # loss calculation
                    pos_scores = scores[torch.arange(len(scores)).cuda(), objs.flatten()]
                    max_n = torch.max(scores, 1, keepdim=True)[0]
                    loss = torch.sum(- pos_scores + max_n + torch.log(torch.sum(torch.exp(scores - max_n),1))) 
                
                # loss backward with scaler
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # cover tail entity or not
            reach_tails = (pos_scores == 0).detach().int().reshape(-1).cpu().tolist()
            reach_tails_list += reach_tails
            epoch_loss += loss.item()
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_latency_ms = (time.perf_counter() - t_time) * 1000.0
        train_peak_mem_mb = self._cuda_peak_memory_mb()
        self.train_time += train_latency_ms / 1000.0
        self.t_time = self.train_time
        
        # evaluate on val/test set
        valid_mrr, out_str = self.evaluate()    
        self.scheduler.step(valid_mrr)
        
        # shuffle train set
        if not self.args.not_shuffle_train:
            self.loader.shuffle_train()
            fact_data = np.concatenate([np.array(self.loader.fact_data), self.loader.idd_data], 0)
            self.train_sampler.updateEdges(fact_data)
        
        return valid_mrr, f'[TRAIN] latency_ms:{train_latency_ms:.2f} peak_gpu_mem_mb:{train_peak_mem_mb:.2f}\t' + out_str
    
    @torch.no_grad()
    def evaluate(self, eval_val=True, eval_test=True, verbose=False, rank_CR=False, mean_rank=False, return_ranks=False):
        ranking = []
        val_ranking_ret = []
        test_ranking_ret = []
        self.model.eval()
        self._reset_cuda_peak_memory()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        i_time = time.perf_counter()
        data_prep_ms = 0.0
        forward_ms = 0.0
        ranking_ms = 0.0
        
        # eval on val set
        if eval_val:
            val_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.valLoader, ncols=50, leave=False):      
                # prepare data            
                prep_t0 = time.perf_counter()
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                data_prep_ms += (time.perf_counter() - prep_t0) * 1000.0
                
                # forward with autocast
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                fwd_t0 = time.perf_counter()
                if hasattr(self.args, 'no_amp') and self.args.no_amp:
                    scores = self.model(subs, rels, subgraph_data, mode='valid').float().data.cpu().numpy()
                else:
                    with torch.amp.autocast('cuda'):
                        scores = self.model(subs, rels, subgraph_data, mode='valid').float().data.cpu().numpy()
                scores = self._post_hoc_rerank(scores, subs, rels, subgraph_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_ms += (time.perf_counter() - fwd_t0) * 1000.0

                # calculate rank
                rank_t0 = time.perf_counter()
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
                if return_ranks:
                    val_ranking_ret += ranks
                    
                if hasattr(self.args, 'dump_scores') and self.args.dump_scores != '':
                    if not hasattr(self, 'dump_objs_list'):
                        self.dump_objs_list = []
                        self.dump_filters_list = []
                    self.dump_objs_list.append(objs)
                    filt_idx = [np.where(f == 1)[0] for f in filters]
                    self.dump_filters_list.extend(filt_idx)
                
                if mean_rank: 
                    mean_ranks = cal_ranks_mean(scores, objs, filters)
                    mean_rank_list += mean_ranks

                # cover tails or not
                ans = np.nonzero(objs)
                ans_score = scores[ans].reshape(-1)
                reach_tails = (ans_score == 0).astype(int).tolist() # (0/1)
                val_reach_tails_list += reach_tails
                ranking_ms += (time.perf_counter() - rank_t0) * 1000.0

            ranking = np.array(ranking)
            v_mrr, v_h1, v_h10 = cal_performance(ranking)
            # print(f'[val]  covering tail ratio: {len(val_reach_tails_list)}, {1 - sum(val_reach_tails_list) / len(val_reach_tails_list)}')
            
            if rank_CR:
                target_rank = torch.Tensor(ranking).reshape(-1)
                rank_thre = [int(i/100 * self.loader.n_ent) for i in range(1,101)]
                rank_CR = []
                for thre in rank_thre:
                    ratio = torch.sum((target_rank <= thre).int()) / len(target_rank)
                    rank_CR.append(float(ratio))
                print('Val set:\n', rank_CR)
                
            # save mean rank
            if mean_rank: self.mean_rank_dict['val'] = copy.deepcopy(mean_rank_list)
                
        else:
            v_mrr, v_h1, v_h10 = -1, -1, -1
        
        # eval on test set
        if eval_test:
            ranking = []
            test_reach_tails_list = []
            if mean_rank: mean_rank_list = []
            for batch_data in tqdm(self.testLoader, ncols=50, leave=False):        
                # prepare data            
                prep_t0 = time.perf_counter()
                subs, rels, objs, subgraph_data = self.prepareData(batch_data)
                data_prep_ms += (time.perf_counter() - prep_t0) * 1000.0
                
                # forward with autocast
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                fwd_t0 = time.perf_counter()
                if hasattr(self.args, 'no_amp') and self.args.no_amp:
                    scores = self.model(subs, rels, subgraph_data, mode='test').float().data.cpu().numpy()
                else:
                    with torch.amp.autocast('cuda'):
                        scores = self.model(subs, rels, subgraph_data, mode='test').float().data.cpu().numpy()
                scores = self._post_hoc_rerank(scores, subs, rels, subgraph_data)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                forward_ms += (time.perf_counter() - fwd_t0) * 1000.0

                # calculate rank
                rank_t0 = time.perf_counter()
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
                if return_ranks:
                    test_ranking_ret += ranks
                    
                if hasattr(self.args, 'dump_scores') and self.args.dump_scores != '':
                    if not hasattr(self, 'dump_objs_list'):
                        self.dump_objs_list = []
                        self.dump_filters_list = []
                    self.dump_objs_list.append(objs)
                    # Convert one-hot filters back to list of indices to save space
                    filt_idx = [np.where(f == 1)[0] for f in filters]
                    self.dump_filters_list.extend(filt_idx)

                if mean_rank: 
                    mean_ranks = cal_ranks_mean(scores, objs, filters)
                    mean_rank_list += mean_ranks
                    
                # cover tails or not
                ans = np.nonzero(objs)
                ans_score = scores[ans].reshape(-1)
                reach_tails = (ans_score == 0).astype(int).tolist() # (0/1)
                test_reach_tails_list += reach_tails
                ranking_ms += (time.perf_counter() - rank_t0) * 1000.0

            ranking = np.array(ranking)
            t_mrr, t_h1, t_h10 = cal_performance(ranking)
            # print(f'[test] covering tail ratio: {len(test_reach_tails_list)}, {1 - sum(test_reach_tails_list) / len(test_reach_tails_list)}')
            
            if rank_CR:
                target_rank = torch.Tensor(ranking).reshape(-1)
                rank_thre = [int(i/100 * self.loader.n_ent) for i in range(1,101)]
                rank_CR = []
                for thre in rank_thre:
                    ratio = torch.sum((target_rank <= thre).int()) / len(target_rank)
                    rank_CR.append(float(ratio))
                print('Test set:\n', rank_CR)
                
            # save mean rank
            if mean_rank: self.mean_rank_dict['test'] = copy.deepcopy(mean_rank_list)
            
        else:
            t_mrr, t_h1, t_h10 = -1, -1, -1
            
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        eval_latency_ms = (time.perf_counter() - i_time) * 1000.0
        eval_peak_mem_mb = self._cuda_peak_memory_mb()
        out_str = '[VALID] MRR:%.6f H@1:%.6f H@10:%.6f\t [TEST] MRR:%.6f H@1:%.6f H@10:%.6f \t[TIME] train:%.4f inference:%.4f \t[LATENCY] eval_total_ms:%.2f data_prep_ms:%.2f forward_ms:%.2f ranking_ms:%.2f \t[PEAK_GPU_MEM] %.2fMB\n'%(v_mrr, v_h1, v_h10, t_mrr, t_h1, t_h10, self.train_time, eval_latency_ms / 1000.0, eval_latency_ms, data_prep_ms, forward_ms, ranking_ms, eval_peak_mem_mb)
        if hasattr(self.args, 'dump_scores') and self.args.dump_scores != '' and hasattr(self, 'dump_gnn_list'):
            gnn_all = np.vstack(self.dump_gnn_list)
            mlp_all = np.vstack(self.dump_mlp_list)
            objs_all = np.concatenate(self.dump_objs_list)
            filters_all = np.array(self.dump_filters_list, dtype=object)
            os.makedirs(os.path.dirname(self.args.dump_scores), exist_ok=True)
            np.savez(self.args.dump_scores, gnn=gnn_all, mlp=mlp_all, objs=objs_all, filters=filters_all)
            print(f"==> Dumped GNN/MLP scores to {self.args.dump_scores}")

        if return_ranks:
            return v_mrr, out_str, val_ranking_ret, test_ranking_ret
        return v_mrr, out_str