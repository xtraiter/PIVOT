# PIVOT: Suy luận đồ thị tri thức theo ngân sách với tối ưu Pareto

**PIVOT** (Pareto-Improved subgraph eValuation for One-shot reasoning under budgeT) — mã nguồn khóa luận tốt nghiệp, xây dựng trên khung **one-shot subgraph reasoning** (Zhou et al., ICLR 2024) và bổ sung ba đóng góp:

1. **Budgeted Protocol** — giao thức đánh giá đồng thời độ chính xác (MRR/H@1/H@10) và chi phí (latency/query, peak VRAM) tại nhiều mức ngân sách subgraph θ ∈ {1%, 5%, 10%, 20%}, dùng **một checkpoint duy nhất** huấn luyện tại θ=10% (test-time extrapolation).
2. **Module MLP Pruning học được** (7 đặc trưng, 7→64→32→1, **2.625 tham số**) với hai chế độ triển khai trên cùng một bộ trọng số:
   - **PIVOT-Rerank** (chế độ chính, tối đa accuracy): `Final(v) = (1−α)·Score_GNN(v) + α·Score_MLP(v)`, α* chọn duy nhất trên tập Valid theo từng seed.
   - **Hybrid+Rerank** (chế độ tiết kiệm chi phí): trộn node subgraph 50% theo điểm MLP + 50% theo PPR lúc suy luận (không train lại GNN) → cùng accuracy, **−30% VRAM**.
3. **Khung Pareto + Budget Controller** — trích frontier (MRR, latency, VRAM) trên Valid và trả lời truy vấn ràng buộc (`argmax MRR s.t. latency ≤ T`, `argmin latency s.t. MRR ≥ X`).

Kèm theo: **Robustness Suite** (xóa cạnh ngẫu nhiên 5/10/20% + relation-specific) và **Feature Ablation hai tầng** (Tier-1 Recall / Tier-2 MRR, 7 biến thể × 3 seed × 2 dataset).

## Kết quả chính (FP32, mean ± std trên 3 seed {42, 123, 1234})

| Dataset | PPR-only (baseline tái lập) | PIVOT-Rerank | Hybrid+Rerank @θ=10% |
|:---|:---:|:---:|:---:|
| **WN18RR** | 0.5637 ± 0.0016 | **0.5685 ± 0.0021** (+0.0047) | 0.5684 ± 0.0015 · VRAM 1035 MB (−30%) |
| **NELL-995** | 0.5354 ± 0.0036 (θ=10%) | **0.5448 ± 0.0038** (+0.0093) | 0.5455 ± 0.0033 · VRAM 1740 MB (−32%) |

- Cải thiện **bền dưới nhiễu**: gap Rerank − PPR giữ ~+0.004 ở mọi mức xóa cạnh; cả ba chế độ suy giảm cùng tốc độ (retention 90.6 / 83.8 / 73.9 / 99.6%).
- Ablation hai-dataset: nhóm đặc trưng **cấu trúc** tạo toàn bộ mức tăng MRR ở khâu rerank; nhóm **ngữ nghĩa** quyết định chất lượng candidate selection (Recall NELL sụp 0.604→0.265 khi thiếu) và hiện thực hóa chế độ Hybrid.

Toàn bộ số liệu truy vết được về log trong repo: xem **[reports/PIVOT_reproduction_walkthrough.md](reports/PIVOT_reproduction_walkthrough.md)** (báo cáo đầy đủ Tuần 1–10 + ablation), CSV chốt tại `reports/csv_deliverables/`, log thô tại `reports/grid_t78_*/`, `reports/robustness_t10/`, `reports/artifacts/`.

## Cấu trúc repo

```
train_auto.py                 # train / eval GNN; cờ --rerank_alpha, --use_learned_pruning,
                              #   --eval_split {valid,test}, --no_amp, --dump_scores, --mlp_ablation_mode
PPR_sampler.py                # PPR sampling + Hybrid 50/50 + trích 7 đặc trưng lúc suy luận
learned_pruning.py            # PruningMLP (7→64→32→1)
run_learned_pruning_wn18rr.py # train MLP + Recall eval (hỗ trợ --ablation_mode v1..v4,l1..l3)
run_learned_pruning_nell.py   # bản NELL, có cache đặc trưng dùng chung cho ablation
budgeted_protocol.py          # sweep budget θ, đo MRR/latency/VRAM
run_grid_t78.sh               # chiến dịch grid 120 lượt (3 phương pháp × 4θ × 3 seed × 2 split)
build_pareto.py               # tổng hợp grid → frontier + Figure 1
pareto_optimizer.py           # BudgetController (truy vấn ràng buộc trên frontier)
make_perturbed_datasets_v2.py # sinh dataset nhiễu (del05/10/20, reldel) trên facts∪train
run_robustness_t10.sh         # chiến dịch robustness (sanity → run), resume-safe
build_robustness.py           # tổng hợp robustness → bảng + figure (có cảnh báo lẫn cache)
reports/                      # walkthrough, changes_summary, CSV chốt, figures, log minh chứng
data/                         # WN18RR (+4 bản nhiễu), NELL-995, YAGO3-10 (raw triples)
savedModels/, reports/artifacts/*/saveModel/   # checkpoint GNN per-seed (~12MB, commit làm minh chứng)
```

> **Lưu ý dung lượng:** cache điểm PPR (`**/ppr_scores/`, ~19GB cho WN18RR) **không** nằm trong repo — được tính tự động ở lần chạy đầu của mỗi dataset/cấu hình nhiễu.

## Môi trường

Python 3.10 · PyTorch 2.x + CUDA 12.x · `torch_scatter` · numpy, pandas, matplotlib, networkx, scipy, tqdm. GPU 16GB đủ cho mọi thí nghiệm (đã chạy trên RTX 5060 Ti / WSL2).

## Tái lập các thí nghiệm chính

**1) Baseline (tái lập paper gốc, mỗi seed ∈ {42,123,1234}):**
```bash
python3 train_auto.py --data_path ./data/WN18RR/ --seed 42 --topk 0.1 \
    --fact_ratio 0.95 --epoch 200 --batchsize 16 --gpu 0
```

**2) PIVOT-Rerank (eval-only tại α* theo Valid — 0.8/0.6/0.7 cho seed 42/123/1234):**
```bash
python3 train_auto.py --data_path ./data/WN18RR/ --only_eval --no_amp --eval_split test \
    --weight data/WN18RR/saveModel/<ckpt_seed42>.pt \
    --pruning_model_path data/WN18RR/budget_results/pruning_mlp_v2_best_seed_42.pt \
    --rerank_alpha 0.8 --topk 0.1 --batchsize 16 --seed 42
```

**3) Hybrid+Rerank (thêm cờ trộn node 50/50, không train lại):** như (2) và thêm `--use_learned_pruning`.

**4) Train MLP Pruning + ablation đặc trưng (Tier-1):**
```bash
python3 run_learned_pruning_wn18rr.py --ablation_mode v4   # v1|v2|v3|v4|l1|l2|l3
python3 run_learned_pruning_nell.py   --ablation_mode v4   # NELL: dùng cache đặc trưng chung
```
Tier-2 (MRR): dump điểm bằng một lượt forward với `--dump_scores <file.npz>` rồi quét α offline; số per-seed chốt tại `reports/artifacts/WN18RR/budget_results/ablation_tier2_results.csv`.

**5) Grid Pareto + Controller:**
```bash
bash run_grid_t78.sh wn18rr        # (INCLUDE_HYBRID=1 để thêm Hybrid)
python3 build_pareto.py --dir reports/grid_t78_wn18rr
python3 pareto_optimizer.py --cache_path reports/grid_t78_wn18rr/pareto_cache_wn18rr_v2.json --max_latency 12.0
```

**6) Robustness Suite:**
```bash
python3 make_perturbed_datasets_v2.py --data_path ./data/WN18RR --seed 42 --configs del05 del10 del20 reldel
bash run_robustness_t10.sh sanity   # PASS mới chạy tiếp
bash run_robustness_t10.sh run
python3 build_robustness.py --dir reports/robustness_t10 --clean_dir reports/grid_t78_wn18rr
```

## Ghi công & trích dẫn

Repo kế thừa và mở rộng mã nguồn chính thức của bài báo *"Less is More: One-shot Subgraph Reasoning on Large-scale Knowledge Graphs"* (ICLR 2024) — [arXiv:2403.10231](https://arxiv.org/abs/2403.10231), repo gốc [AndrewZhou924/one-shot-subgraph](https://github.com/AndrewZhou924/one-shot-subgraph). Mọi khác biệt so với mã gốc được ghi dạng diff đầy đủ trong [reports/changes_summary.md](reports/changes_summary.md).

```bibtex
@inproceedings{zhou2024less,
  title={Less is More: One-shot Subgraph Reasoning on Large-scale Knowledge Graphs},
  author={Zhou, Zhanke and Yao, Quanming and others},
  booktitle={ICLR},
  year={2024}
}
```
