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

ABLATION: dat use_learned=False de quay ve xep hang theo PPR thuan tuy
(day chinh la yeu cau "tat learned component" trong ke hoach cua ban).

CACH TICH HOP VAO REPO CUA BAN
--------------------------------
File nay viet doc lap, co the chay thu ngay (xem block __main__ o cuoi,
dung du lieu gia lap) TRUOC KHI ban can ghep vao pipeline thuc.

De ghep vao thuc te, ban can sua 3 cho duoc danh dau "ADAPT" ben duoi:
  1. build_candidate_features()  -- lay dung bien tu PPR_sampler.py cua
     ban (vi du self.n_ent, self.PPR_W, do thi self.homoTrainGraph...)
  2. train_pruning_model()       -- lay candidate pool + true tail tu
     vong lap train hien co trong train_auto.py / base_model.py
  3. prune_candidates()          -- goi ham nay THAY CHO TopK(...) o
     Step-2 cua Algorithm 1 trong paper (PPR_sampler.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. Trich dac trung cho moi candidate
# ----------------------------------------------------------------------
# ADAPT: thay placeholder nay bang du lieu thuc tu PPR_sampler.py /
# load_data.py. Shape ky vong duoc ghi ro trong docstring.

def build_candidate_features(
    candidate_ids,      # LongTensor [N]   id cua N candidate entity
    ppr_scores,         # FloatTensor [N]  diem PPR cua tung candidate (tu u)
    node_degree,        # FloatTensor [N]  degree cua node trong toan KG
    hop_distance,       # FloatTensor [N]  khoang cach BFS tu u (so hop)
    rel_match_score,    # FloatTensor [N]  do "khop" giua relation q va cac
                        #                  relation ke can candidate nay
):
    """
    Tra ve ma tran dac trung [N, 4]. Muon them dac trung khac (vd in/out
    degree rieng, embedding similarity...) thi them cot va doi in_dim cua
    PruningMLP cho khop.
    """
    feats = torch.stack(
        [ppr_scores, torch.log1p(node_degree), hop_distance.float(), rel_match_score],
        dim=1,
    )
    # chuan hoa theo batch -- giup MLP hoc on dinh hon nhieu vi cac dac
    # trung nay co thang do rat khac nhau (PPR ~0-1, degree co the ~10000)
    feats = (feats - feats.mean(0, keepdim=True)) / (feats.std(0, keepdim=True) + 1e-6)
    return feats


# ----------------------------------------------------------------------
# 2. Model pruning -- co tinh nho, vi muc dich la KHONG them mot model
#    nang nhu GNN chinh, chi la mot bo loc re
# ----------------------------------------------------------------------

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
        # feats: [N, in_dim] -> scores: [N]
        return self.net(feats).squeeze(-1)


# ----------------------------------------------------------------------
# 3. Loss: "giu true tail trong top-K"
# ----------------------------------------------------------------------
# 2 lua chon hop ly -- bat dau bang (a), de debug hon.

def listwise_bce_loss(scores, true_tail_mask):
    """
    (a) BCE don gian tren tung candidate: true tail label=1, con lai=0.
    scores, true_tail_mask: deu shape [N] cho MOT query.
    Day la cung kieu loss paper goc dung cho predictor (L_cls), nen de
    bao ve khi giai trinh voi giao vien / hoi dong.
    """
    return F.binary_cross_entropy_with_logits(scores, true_tail_mask.float())


def pairwise_hinge_loss(scores, true_tail_idx, margin=1.0, n_negatives=20):
    """
    (b) Pairwise ranking loss: day diem cua true tail cao hon mot mau
    negative ngau nhien it nhat `margin`. Thuong cho ranh gioi top-K ro
    hon BCE khi candidate pool lon va mat can.
    """
    true_score = scores[true_tail_idx]
    neg_idx = torch.randperm(scores.numel())[:n_negatives]
    neg_idx = neg_idx[neg_idx != true_tail_idx]
    if neg_idx.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)
    neg_scores = scores[neg_idx]
    loss = F.relu(margin - (true_score - neg_scores)).mean()
    return loss


def train_pruning_model(model, optimizer, query_batches, n_epochs=20, use_pairwise=False):
    """
    query_batches: iterable, moi phan tu la 1 dict cho 1 query luc train:
        {
          "features": FloatTensor [N, in_dim],  (tu build_candidate_features)
          "true_tail_idx": int,                 chi so cua tail dung trong
                                                 candidate_ids/features
        }
    ADAPT: xay iterable nay tu vong lap train hien co trong train_auto.py
    -- ban DA tinh candidate pool tu PPR o do, chi can truyen them no
    (va tail dung v) vao day thay vi bo qua.
    """
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


# ----------------------------------------------------------------------
# 4. Pruning luc inference -- ham nay THAY THE loi goi TopK(...) trong
#    Step-2 cua Algorithm 1 (PPR_sampler.py) cua paper goc
# ----------------------------------------------------------------------

@torch.no_grad()
def prune_candidates(model, candidate_ids, features, budget_k, use_learned=True,
                      ppr_scores=None):
    """
    Tra ve danh sach candidate id cuoi cung, kich thuoc budget_k.

    use_learned=True  -> xep hang theo diem MLP   (PIVOT, learned pruning)
    use_learned=False -> xep hang theo PPR thuan   (ablation: "PPR only")
    """
    if use_learned:
        model.eval()
        scores = model(features)
    else:
        assert ppr_scores is not None, "can ppr_scores cho ablation PPR-only"
        scores = ppr_scores

    k = min(budget_k, candidate_ids.numel())
    _, topk_idx = torch.topk(scores, k)
    return candidate_ids[topk_idx]


# ----------------------------------------------------------------------
# 5. Metric danh gia: "true tail con song sau khi prune khong?"
#    -- day la metric chinh cho deliverable Tuan 9
# ----------------------------------------------------------------------

def recall_at_k_after_pruning(model, eval_queries, budget_k, use_learned=True):
    """
    eval_queries: cung dinh dang voi query_batches cua train_pruning_model,
    them truong "candidate_ids" (LongTensor [N]).
    Tra ve ti le query ma true_tail_idx con song sau khi prune.

    So sanh use_learned=True vs False O CUNG budget_k -- neu learned
    pruning giu duoc nhieu true tail hon o cung budget, do chinh la ket
    qua "cung budget -> accuracy cao hon" ma ke hoach Tuan 9 yeu cau.
    """
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


if __name__ == "__main__":
    # =========================================================================
    # CACH SU DUNG (3 cap do)
    # =========================================================================
    #
    # Cap 1 — Smoke test doc lap (khong can data thuc):
    #   python3 learned_pruning.py                        # learned (default)
    #   python3 learned_pruning.py --mode ppr             # PPR-only (ablation)
    #   python3 learned_pruning.py --mode compare         # so sanh ca 2 cung luc
    #
    # Cap 2 — Tich hop vao pipeline hien co (train_auto.py / PPR_sampler.py):
    #   Goi prune_candidates(..., use_learned=True)  thay cho TopK PPR    [PIVOT]
    #   Goi prune_candidates(..., use_learned=False) de giu nguyen PPR    [paper goc]
    #
    # Cap 3 — Benchmark so sanh (dung voi budgeted_protocol.py):
    #   Chay budgeted_protocol.py 2 lan voi 2 checkpoint khac nhau:
    #     - checkpoint PPR-only  (tu train_auto.py hien tai)
    #     - checkpoint PIVOT     (sau khi huan luyen them learned_pruning)
    # =========================================================================

    import argparse
    ap = argparse.ArgumentParser(description="Learned pruning smoke test")
    ap.add_argument(
        "--mode", choices=["learned", "ppr", "compare"], default="compare",
        help=(
            "learned  = chi chay PIVOT (MLP re-ranking)\n"
            "ppr      = chi chay PPR-only (paper goc, ablation)\n"
            "compare  = chay ca 2 va in ket qua so sanh [default]"
        ),
    )
    ap.add_argument("--budget_k", type=int, default=10, help="So candidate giu lai")
    ap.add_argument("--n_candidates", type=int, default=200, help="Kich thuoc candidate pool")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    N = args.n_candidates
    budget_k = args.budget_k

    # ----- Sinh du lieu gia lap -----
    candidate_ids = torch.arange(N)
    ppr = torch.rand(N) ** 2
    degree = torch.randint(1, 500, (N,)).float()
    hop = torch.randint(0, 4, (N,)).float()
    relmatch = torch.rand(N)
    feats = build_candidate_features(candidate_ids, ppr, degree, hop, relmatch)

    # "True tail" = entity PPR thap nhung rel_match cao
    # -> day chinh la truong hop PPR-only se bo lo
    true_idx = int(torch.argmax(relmatch - ppr))
    true_id = candidate_ids[true_idx].item()

    print(f"==> N={N} candidates | budget_k={budget_k} | true_tail_id={true_id}")
    print(f"    (true tail: ppr={ppr[true_idx]:.4f}, rel_match={relmatch[true_idx]:.4f})")

    # ----- Huan luyen MLP (chi can khi dung mode learned hoac compare) -----
    if args.mode in ("learned", "compare"):
        model = PruningMLP(in_dim=feats.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        fake_batches = [{"features": feats, "true_tail_idx": true_idx}] * 50
        print("\n==> Huan luyen PruningMLP...")
        train_pruning_model(model, opt, fake_batches, n_epochs=15)
    else:
        model = None  # PPR-only khong can model

    # ----- Inference -----
    print(f"\n==> Pruning voi budget_k={budget_k}:")

    if args.mode == "learned":
        kept = prune_candidates(model, candidate_ids, feats, budget_k,
                                use_learned=True)
        print(f"  [PIVOT - learned] giu: {kept.tolist()}")
        print(f"  -> co giu true tail ({true_id}): {true_id in kept.tolist()}")

    elif args.mode == "ppr":
        kept = prune_candidates(model, candidate_ids, feats, budget_k,
                                use_learned=False, ppr_scores=ppr)
        print(f"  [PPR-only baseline] giu: {kept.tolist()}")
        print(f"  -> co giu true tail ({true_id}): {true_id in kept.tolist()}")

    else:  # compare
        kept_learned = prune_candidates(model, candidate_ids, feats, budget_k,
                                        use_learned=True)
        kept_ppr = prune_candidates(model, candidate_ids, feats, budget_k,
                                    use_learned=False, ppr_scores=ppr)
        print(f"\n  Che do               | Co giu true tail ({true_id})?")
        print(f"  ---------------------|------------------------")
        print(f"  PIVOT (learned MLP)  | {'YES ✓' if true_id in kept_learned.tolist() else 'NO  ✗'}")
        print(f"  PPR-only (paper goc) | {'YES ✓' if true_id in kept_ppr.tolist() else 'NO  ✗'}")
        print(f"\n  PIVOT giu  : {kept_learned.tolist()}")
        print(f"  PPR giu    : {kept_ppr.tolist()}")
