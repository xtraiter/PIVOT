"""learned_pruning.py - Module MLP Pruning cua PIVOT.

PPR sinh candidate pool (top-10% theo PPR); MLP nho, query-aware cham diem lai
tung candidate tu 7 dac trung: ppr_log, ppr_rank_pct, deg_log, hop_dist,
is_direct, tail_freq_q, rel_match. Diem MLP dung o hai che do test-time:
  - PIVOT-Rerank : tron diem voi GNN o buoc xep hang cuoi.
  - Hybrid+Rerank: chon 50% node subgraph theo diem MLP.
Huan luyen / danh gia: experiments/run_learned_pruning_*.py.
"""

import torch
import torch.nn as nn


class PruningMLP(nn.Module):
    """MLP 3 tang: in_dim -> hidden -> hidden//2 -> 1 (2.625 tham so, in_dim=7)."""

    def __init__(self, in_dim=7, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, feats):
        return self.net(feats).squeeze(-1)


@torch.no_grad()
def prune_candidates(model, candidate_ids, features, budget_k, use_learned=True,
                     ppr_scores=None):
    """Giu top-k candidate theo diem MLP (use_learned=True) hoac theo PPR."""
    if use_learned:
        model.eval()
        scores = model(features)
    else:
        assert ppr_scores is not None
        scores = ppr_scores
    k = min(budget_k, candidate_ids.numel())
    _, topk_idx = torch.topk(scores, k)
    return candidate_ids[topk_idx]
