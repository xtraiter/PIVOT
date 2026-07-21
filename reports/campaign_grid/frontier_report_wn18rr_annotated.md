# Frontier Report WN18RR — Tuần 7–8 (PIVOT KLTN)
# Tổng hợp toàn bộ kết quả Grid Search & Pareto Optimizer
# Dataset: WN18RR | Thực hiện: FP32 (--no_amp) | 3 seed: 42, 123, 1234

---

## Metadata

| Hạng mục | Giá trị |
|:---|:---|
| Dataset | WN18RR (40,943 entities, 11 relations, 87,140 train / 3,134 valid / 3,134 test) |
| Checkpoint | `reports/artifacts/WN18RR/saveModel/topk_0.1_layer_8_ValMRR_*_seed{42,123,1234}.pt` |
| Precision | FP32 (`--no_amp`), batchsize=16 |
| N_test_q | 3,134 × 2 = 6,268 (eval split: test-only) |
| Seeds | 42, 123, 1234 (3 seeds tất cả đều đủ) |
| Phương pháp | PPR-only (`ppr`), PIVOT-Rerank (`rerank`), Hybrid+Rerank (`hybrid`) |
| Budget θ | 1%, 5%, 10%, 20% |
| Tổng số lượt chạy | 72 (3 method × 4 budget × 3 seed × 2 split valid/test) |
| Log directory | `campaign_grid/grid_t78_wn18rr/` |
| Cache | `campaign_grid/grid_t78_wn18rr/pareto_cache_wn18rr_v2.json` |
| Figure | `campaign_grid/grid_t78_wn18rr/figure1_frontier_wn18rr.png` |

---

## 1. Bảng tổng hợp grid (mean±std, 3 seed, FP32) — WN18RR

Latency/query tính theo **test-only ÷ 6,268 truy vấn** (quy ước §1.5 của KLTN). Valid MRR dùng để chọn frontier và controller; Test MRR là số liệu báo cáo chính thức.

| Phương pháp | θ | Valid MRR (chọn) | Test MRR (báo cáo) | Latency/q (ms) | VRAM (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| Hybrid+Rerank | 1% | 0.5488 | 0.5481 ± 0.0037 | 6.66 | 117 |
| Hybrid+Rerank | 5% | 0.5661 | 0.5661 ± 0.0007 | 9.66 | 482 |
| Hybrid+Rerank | 10% | 0.5687 | 0.5684 ± 0.0015 | 13.18 | 1035 |
| Hybrid+Rerank | 20% | 0.5685 | 0.5674 ± 0.0026 | 21.60 | 2322 |
| PPR-only | 1% | 0.5435 | 0.5412 ± 0.0015 | 5.61 | 160 |
| PPR-only | 5% | 0.5645 | 0.5643 ± 0.0015 | 8.92 | 730 |
| PPR-only | 10% | 0.5651 | 0.5638 ± 0.0017 | 15.14 | 1487 |
| PPR-only | 20% | 0.5623 | 0.5600 ± 0.0015 | 25.16 | 3164 |
| PIVOT-Rerank | 1% | 0.5463 | 0.5446 ± 0.0015 | 7.10 | 160 |
| PIVOT-Rerank | 5% | 0.5681 | 0.5678 ± 0.0017 | 10.96 | 730 |
| PIVOT-Rerank | 10% | 0.5691 | 0.5685 ± 0.0021 | 17.71 | 1487 |
| PIVOT-Rerank | 20% | 0.5670 | 0.5654 ± 0.0013 | 27.92 | 3164 |

---

## 2. Frontier cơ sở (PPR-only) — deliverable T7-8 thuần

Frontier trích theo **Valid MRR** (không bị trội). Mỗi điểm vào frontier khi Valid MRR của nó cao hơn mọi điểm có latency nhỏ hơn.

| θ | Valid MRR | Test MRR | Latency/q (ms) |
|:---:|:---:|:---:|:---:|
| 1% | 0.5435 | 0.5412 ± 0.0015 | 5.61 |
| 5% | 0.5645 | 0.5643 ± 0.0015 | 8.92 |
| 10% | 0.5651 | 0.5638 ± 0.0017 | 15.14 |

> **Ghi chú frontier cơ sở:** Điểm θ=10% được chọn vào frontier theo **Valid MRR** (0.5651 > 0.5645 tại θ=5%). Nghịch đảo nhẹ valid–test tại điểm này (Test 0.5638 < 0.5643 của θ=5%) nằm trong 1 std và được giữ nguyên theo protocol. Frontier trích từ Valid là quyết định của pipeline; số Test báo cáo đúng thực tế.

---

## 3. Frontier đầy đủ — deliverable tích hợp T7-9

Trích frontier từ tất cả phương pháp (PPR + Rerank + Hybrid), sắp xếp tăng dần theo latency.

| Phương pháp | θ | Valid MRR | Test MRR | Latency/q (ms) | VRAM (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| PPR-only | 1% | 0.5435 | 0.5412 ± 0.0015 | 5.61 | 160 |
| Hybrid+Rerank | 1% | 0.5488 | 0.5481 ± 0.0037 | 6.66 | 117 |
| PPR-only | 5% | 0.5645 | 0.5643 ± 0.0015 | 8.92 | 730 |
| Hybrid+Rerank | 5% | 0.5661 | 0.5661 ± 0.0007 | 9.66 | 482 |
| PIVOT-Rerank | 5% | 0.5681 | 0.5678 ± 0.0017 | 10.96 | 730 |
| Hybrid+Rerank | 10% | 0.5687 | 0.5684 ± 0.0015 | 13.18 | 1035 |
| PIVOT-Rerank | 10% | 0.5691 | 0.5685 ± 0.0021 | 17.71 | 1487 |

> **Nhận xét frontier đầy đủ:** Hybrid+Rerank chiếm ưu thế VRAM đáng kể — tại θ=10%, Hybrid tiêu thụ 1035 MB so với 1487 MB của PPR-only/PIVOT-Rerank (giảm 30.4%). Frontier này cho thấy: để đạt Valid MRR ≥ 0.565 với chi phí VRAM tối thiểu, Hybrid+Rerank@5% là lựa chọn tối ưu (482 MB, 9.66 ms/q, Test 0.5661).

---

## 4. Demo BudgetController — WN18RR

Controller chọn theo **Valid MRR**, báo cáo **Test MRR** để đánh giá khách quan.

### Truy vấn 1: Latency ≤ 12 ms → Best Valid MRR

```
python3 pareto_optimizer.py --cache_path campaign_grid/grid_t78_wn18rr/pareto_cache_wn18rr_v2.json --max_latency 12.0
```

**Kết quả:** PIVOT-Rerank @ θ=5%
- Valid MRR: `0.5681` | Test MRR: `0.5678 ± 0.0017`
- Latency/q: `10.96 ms` | VRAM: `730 MB`

### Truy vấn 2: Min Valid MRR ≥ 0.565 → Min Latency

```
python3 pareto_optimizer.py --cache_path campaign_grid/grid_t78_wn18rr/pareto_cache_wn18rr_v2.json --min_mrr 0.565
```

**Kết quả:** Hybrid+Rerank @ θ=5%
- Valid MRR: `0.5661` | Test MRR: `0.5661 ± 0.0007`
- Latency/q: `9.66 ms` | VRAM: `482 MB`

**Phân tích:** Hai truy vấn trả về hai điểm **khác nhau** — PIVOT-Rerank cho accuracy tốt hơn (+0.0020 Valid MRR) với latency cao hơn 1.30 ms; Hybrid+Rerank là lựa chọn tối ưu khi ưu tiên tiết kiệm VRAM (−248 MB) và latency thấp. Đây là minh họa điển hình của trade-off trên frontier Pareto đầy đủ.

---

## 5. Cross-check với kết quả các tuần trước

| Checkpoint | Metric | Tuần 2-3 (training) | §7 Grid T7-8 | Chênh lệch |
|:---|:---|:---:|:---:|:---:|
| seed42, θ=10% | Test MRR | 0.5636 | 0.5638 ± 0.0017 (mean) | +0.0002 ✅ |
| seed123, θ=10% | Test MRR | 0.5648 | — (trong mean) | ≤ 0.003 ✅ |
| seed1234, θ=10% | Test MRR | 0.5648 | — (trong mean) | ≤ 0.003 ✅ |
| θ=10% (PPR) | VRAM | 1487 MB (Tuần 6) | 1487 MB | ✅ Nhất quán |
| θ=5% (PPR) | VRAM | 730 MB (Tuần 6) | 730 MB | ✅ Nhất quán |

> **Ghi chú latency:** Latency/q trong Tuần 6 (ví dụ: 60.81 ms tại θ=10%) tính theo quy ước `eval valid+test ÷ 3,134 triple`; §7 tính `test-only ÷ 6,268 truy vấn` → §7 nhỏ hơn ~4×. Cùng checkpoint, khác quy ước chia, nhất quán.

---

## 6. Nhận xét phân tích chuyên sâu

Phân tích Pareto trên WN18RR cho thấy sự vượt trội rõ rệt của các phương pháp tích hợp so với baseline PPR-only:

1. **PIVOT-Rerank** luôn nằm trên frontier ở vùng MRR cao nhất với latency chấp nhận được. Tại θ=10%, đây là điểm Pareto-optimal tổng thể: Test MRR 0.5685, latency 17.71 ms/q.

2. **Hybrid+Rerank** chiếm ưu thế tuyệt đối ở vùng tiết kiệm VRAM: θ=10% chỉ tốn 1035 MB vs 1487 MB của PPR/Rerank (giảm 30.4%) với Test MRR tương đương (0.5684 vs 0.5685).

3. **PPR-only** vẫn có vai trò ở vùng cực-tiết-kiệm (θ=1%, 5.61 ms/q, 160 MB) khi không cần tích hợp reranking nhưng chấp nhận mức accuracy thấp hơn ~2.7% Test MRR.

4. **Tính nhất quán nội bộ:** VRAM của PPR-only và PIVOT-Rerank khớp hoàn toàn (cùng subgraph topology trước GNN pass), trong khi Hybrid thấp hơn ~30% do MLP filter giảm candidate pool trước khi truyền tin.