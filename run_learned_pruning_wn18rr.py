"""
run_learned_pruning_wn18rr.py  (v3 — Full Multi-seed & PPR Pool Coverage)
========================================================================
Sửa đổi toàn diện theo spec và yêu cầu của người dùng:
  1. 7 features (thêm tail_freq_for_q — nhóm C)
  2. Kiến trúc 64→32 (spec v2), không phải 32→16
  3. Hard negative mining: 30 hard + 20 random per query,
     loss kết hợp BCE(pos_weight) + λ * pairwise_hinge
  4. Đo lường PPR Pool Coverage (Tỷ lệ bao phủ của tập candidate làm trần lý thuyết).
  5. Hỗ trợ Multi-seed (42, 123, 1234), huấn luyện và đánh giá trên từng seed,
     sau đó tính toán và hiển thị Mean ± Std.
  6. Tách biệt hai góc nhìn đánh giá:
     - Realistic Recall: coi các query không được bao phủ bởi PPR pool là miss (đúng thực tế inference).
     - Oracle Recall: tự động chèn true tail vào pool nếu bị PPR bỏ lỡ (đánh giá năng lực xếp hạng thuần túy).

Logs chi tiết ra: data/WN18RR/budget_results/pruning_mlp_v2.log
Bảng csv tổng hợp lưu ra: data/WN18RR/budget_results/pruning_mlp_aggregated_summary.csv
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

# ======================================================================
# 1. Setup Logging
# ======================================================================
log_dir = "./data/WN18RR/budget_results"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "pruning_mlp_v2.log")

# Clear existing log handlers to avoid duplication
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


# ======================================================================
# BFS for hop distances
# ======================================================================
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


# ======================================================================
# Hard Negative Sampling (spec: 30 hard + 20 random)
# ======================================================================
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

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Build PPR Sampler (Done once to save time)
    # ------------------------------------------------------------------
    logger.info("Initializing PPR Sampler (Pre-loading scores)...")
    loader_args.n_samp_ent = int(args.topk * n_ent)
    loader_args.n_samp_edge = -1
    loader_args.add_manual_edges = False

    fact_homo_edges = list(set([(h, t) for (h, r, t) in loader.fact_data]))
    fact_data = np.concatenate([np.array(loader.fact_data), loader.idd_data], 0)
    train_sampler = pprSampler(n_ent, n_rel, loader_args.n_samp_ent, loader_args.n_samp_edge,
                               fact_homo_edges, fact_data, args.data_path, split='train', args=loader_args)

    # ------------------------------------------------------------------
    # Build KG adjacency, degrees, direct edges, relation stats
    # ------------------------------------------------------------------
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

    # --- Feature #5: tail_freq_for_q ---
    logger.info("Computing tail_freq_for_q statistics...")
    tail_freq = np.zeros((n_ent, 2 * n_rel + 2), dtype=np.float32)
    for h, r, t in loader.fact_data:
        tail_freq[t, r] += 1.0
    rel_total = tail_freq.sum(axis=0, keepdims=True)
    tail_freq_norm = torch.tensor(tail_freq / (rel_total + 1e-8))

    # --- Feature #6: rel_match_score ---
    logger.info("Computing rel_match_score statistics...")
    rel_counts = np.zeros((n_ent, 2 * n_rel + 2), dtype=np.float32)
    for h, r, t in loader.fact_data:
        rel_counts[h, r] += 1.0
    row_sums = rel_counts.sum(axis=1, keepdims=True)
    rel_dist = torch.tensor(rel_counts / (row_sums + 1e-8))

    # ------------------------------------------------------------------
    # Feature Builder (7 features)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Collect query data with PPR candidate pool coverage measurement
    # ------------------------------------------------------------------
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
                # Oracle mode: append it to the candidate pool
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

    # ------------------------------------------------------------------
    # Loop over multiple seeds for robust evaluation
    # ------------------------------------------------------------------
    seeds = [42, 123, 1234]
    budgets = [10, 50, 100, 200, 500]

    # Store results for aggregation
    # format: {seed: {k: {'oracle_mlp': x, 'realistic_mlp': y, 'oracle_ppr': z, 'realistic_ppr': w}}}
    seed_results = {}

    for seed in seeds:
        logger.info("=" * 60)
        logger.info(f"RUNNING WITH SEED: {seed}")
        logger.info("=" * 60)
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 7 input, hidden=64 -> 32 (spec v2)
        in_dim = 7
        model = PruningMLP(in_dim=in_dim, hidden=64).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )

        best_recall = 0.0
        patience_counter = 0
        best_model_path = os.path.join(log_dir, f"pruning_mlp_v2_best_seed_{seed}.pt")

        # Training loop
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            total_bce = 0.0
            total_hinge = 0.0
            
            # Shuffle using numpy random (seeded)
            indices = np.arange(len(train_queries))
            np.random.shuffle(indices)

            for idx in indices:
                q = train_queries[idx]
                true_idx = q["true_tail_idx"]
                ppr_scores = q["ppr_scores"]
                features = q["features"]

                # Hard negative sampling
                neg_indices = sample_negatives(
                    ppr_scores, true_idx,
                    n_hard=args.n_hard_neg, n_random=args.n_random_neg
                )

                # Subset: true tail + sampled negatives
                subset_idx = torch.cat([torch.tensor([true_idx], device=features.device), neg_indices])
                subset_feats = features[subset_idx]
                subset_true_idx = 0

                optimizer.zero_grad()
                scores = model(subset_feats)

                # BCE with pos_weight
                n_neg = neg_indices.numel()
                pos_weight = torch.tensor([min(float(n_neg), 50.0)], device=scores.device)
                labels = torch.zeros_like(scores)
                labels[subset_true_idx] = 1.0
                bce_loss = F.binary_cross_entropy_with_logits(scores, labels, pos_weight=pos_weight)

                # Pairwise hinge
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

            # Evaluate Realistic Recall@100 on valid set for early stopping
            model.eval()
            hits = 0
            with torch.no_grad():
                for q in valid_queries:
                    # Only check if covered
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

        # Load best model for this seed and evaluate
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Seed {seed} | Best Model loaded (Realistic R@100 = {best_recall:.4f})")

        seed_results[seed] = {}
        model.eval()
        with torch.no_grad():
            for k in budgets:
                hits_oracle_mlp = 0
                hits_real_mlp = 0
                
                hits_oracle_ppr = 0
                hits_real_ppr = 0

                for q in valid_queries:
                    # 1. Evaluate MLP
                    kept_mlp = prune_candidates(model, q["candidate_ids"], q["features"], k, use_learned=True)
                    true_id = q["candidate_ids"][q["true_tail_idx"]]
                    
                    if true_id in kept_mlp:
                        hits_oracle_mlp += 1
                        if q["is_covered"]:
                            hits_real_mlp += 1

                    # 2. Evaluate PPR baseline
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
                
                logger.info(
                    f"Seed {seed:4d} | Budget K={k:3d} | "
                    f"MLP (Oracle/Real): {seed_results[seed][k]['oracle_mlp']:.4f}/{seed_results[seed][k]['realistic_mlp']:.4f} | "
                    f"PPR (Oracle/Real): {seed_results[seed][k]['oracle_ppr']:.4f}/{seed_results[seed][k]['realistic_ppr']:.4f}"
                )

    # ------------------------------------------------------------------
    # Aggregate and Compute Mean ± Std across seeds
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("                    AGGREGATED MULTI-SEED ANALYSIS                 ")
    logger.info("=" * 70)

    rows = []
    for k in budgets:
        # Extract values for budget K across seeds
        o_mlp_vals = [seed_results[s][k]["oracle_mlp"] for s in seeds]
        r_mlp_vals = [seed_results[s][k]["realistic_mlp"] for s in seeds]
        o_ppr_vals = [seed_results[s][k]["oracle_ppr"] for s in seeds]
        r_ppr_vals = [seed_results[s][k]["realistic_ppr"] for s in seeds]

        o_diff_vals = [o_mlp - o_ppr for o_mlp, o_ppr in zip(o_mlp_vals, o_ppr_vals)]
        r_diff_vals = [r_mlp - r_ppr for r_mlp, r_ppr in zip(r_mlp_vals, r_ppr_vals)]

        # Stats
        o_mlp_mean, o_mlp_std = np.mean(o_mlp_vals), np.std(o_mlp_vals)
        r_mlp_mean, r_mlp_std = np.mean(r_mlp_vals), np.std(r_mlp_vals)
        o_ppr_mean, o_ppr_std = np.mean(o_ppr_vals), np.std(o_ppr_vals)
        r_ppr_mean, r_ppr_std = np.mean(r_ppr_vals), np.std(r_ppr_vals)

        o_diff_mean = np.mean(o_diff_vals)
        r_diff_mean = np.mean(r_diff_vals)

        rows.append({
            "Budget": k,
            # Oracle
            "Oracle MLP Mean": o_mlp_mean, "Oracle MLP Std": o_mlp_std,
            "Oracle PPR Mean": o_ppr_mean, "Oracle PPR Std": o_ppr_std,
            "Oracle Delta": o_diff_mean,
            # Realistic
            "Realistic MLP Mean": r_mlp_mean, "Realistic MLP Std": r_mlp_std,
            "Realistic PPR Mean": r_ppr_mean, "Realistic PPR Std": r_ppr_std,
            "Realistic Delta": r_diff_mean
        })

    df_agg = pd.DataFrame(rows)
    # Save CSV summary
    csv_out = os.path.join(log_dir, "pruning_mlp_aggregated_summary.csv")
    df_agg.to_csv(csv_out, index=False)
    logger.info(f"Aggregated summary saved to: {csv_out}")

    # Output detailed report to log
    logger.info(f"PPR Candidate Pool Upper Bound (Validation Coverage): {valid_coverage*100:.2f}%")
    logger.info("-" * 80)
    logger.info("1. REALISTIC EVALUATION (Coi query ngoài candidate pool là miss - Đúng thực tế)")
    logger.info("-" * 80)
    logger.info("Budget K  |  PIVOT (MLP) Recall (Mean±Std)  |  PPR-only Recall (Mean±Std)  |  Δ Mean")
    logger.info("----------|---------------------------------|------------------------------|----------")
    for r in rows:
        diff_sign = "+" if r["Realistic Delta"] >= 0 else ""
        logger.info(
            f"{r['Budget']:8d}  |  "
            f"{r['Realistic MLP Mean']:.4f} ± {r['Realistic MLP Std']:.4f}       |  "
            f"{r['Realistic PPR Mean']:.4f} ± {r['Realistic PPR Std']:.4f}        |  "
            f"{diff_sign}{r['Realistic Delta']:.4f}"
        )
    logger.info("-" * 80)

    logger.info("2. ORACLE EVALUATION (Chèn true tail vào pool nếu bị PPR bỏ lỡ)")
    logger.info("-" * 80)
    logger.info("Budget K  |  PIVOT (MLP) Recall (Mean±Std)  |  PPR-only Recall (Mean±Std)  |  Δ Mean")
    logger.info("----------|---------------------------------|------------------------------|----------")
    for r in rows:
        diff_sign = "+" if r["Oracle Delta"] >= 0 else ""
        logger.info(
            f"{r['Budget']:8d}  |  "
            f"{r['Oracle MLP Mean']:.4f} ± {r['Oracle MLP Std']:.4f}       |  "
            f"{r['Oracle PPR Mean']:.4f} ± {r['Oracle PPR Std']:.4f}        |  "
            f"{diff_sign}{r['Oracle Delta']:.4f}"
        )
    logger.info("=" * 80)
    logger.info("Done.")


if __name__ == '__main__':
    main()
