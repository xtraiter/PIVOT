# Frontier Report NELL-995 — Tuần 7–8 (PIVOT KLTN)
# Tổng hợp toàn bộ kết quả Grid Search & Pareto Optimizer
# Dataset: NELL-995 | Thực hiện: FP32 (--no_amp) | 3 seed: 42, 123, 1234

---

## Metadata

| Hạng mục | Giá trị |
|:---|:---|
| Dataset | NELL-995 (75,492 entities, 200 relations, 149,678 train / 543 valid / 3,992 test) |
| Checkpoint | `reports/artifacts/nell/saveModel/topk_0.1_layer_8_ValMRR_*_seed{42,123,1234}.pt` |
| Precision | FP32 (`--no_amp`), batchsize=8 (quy ước paper NELL) |
| N_test_q | 3,992 × 2 = 7,984 (eval split: test-only) |
| Seeds | 42, 123, 1234 (3 seeds tất cả đều đủ) |
| Phương pháp | PPR-only (`ppr`), PIVOT-Rerank (`rerank`) — Hybrid: tùy chọn, chưa chạy |
| Budget θ | 1%, 5%, 10%, 20% |
| Tổng số lượt chạy | 48 (2 method × 4 budget × 3 seed × 2 split valid/test) |
| Log directory | `grid_t78_nell/` |
| Cache | `grid_t78_nell/pareto_cache_nell_v2.json` |
| Figure | `grid_t78_nell/figure1_frontier_nell.png` |

> ⚠️ **Ghi chú VRAM và latency:** Do batchsize=8 (thay vì bs=16 của Tuần 6), cột VRAM và latency trong bảng này nhỏ hơn ~2× so với `reports/artifacts/nell/budget_results/seed_*/summary.csv`. Số chuẩn hoá trong file này (bs=8) là tham chiếu chính thức của KLTN.

---

## 1. Bảng tổng hợp grid (mean±std, 3 seed, FP32) — NELL-995

Latency/query tính theo **test-only ÷ 7,984 truy vấn** (quy ước §1.5 của KLTN). Valid MRR dùng để chọn frontier và controller; Test MRR là số liệu báo cáo chính thức.

| Phương pháp | θ | Valid MRR (chọn) | Test MRR (báo cáo) | Latency/q (ms) | VRAM (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| PPR-only | 1% | 0.4840 | 0.5361 ± 0.0030 | 6.84 | 395 |
| PPR-only | 5% | 0.5012 | 0.5369 ± 0.0042 | 16.19 | 1715 |
| PPR-only | 10% | 0.5003 | 0.5354 ± 0.0036 | 27.12 | 2547 |
| PPR-only | 20% | 0.4959 | 0.5297 ± 0.0050 | 46.37 | 3573 |
| PIVOT-Rerank | 1% | 0.5357 | 0.5414 ± 0.0044 | 15.17 | 395 |
| PIVOT-Rerank | 5% | 0.5655 | 0.5468 ± 0.0042 | 27.77 | 1715 |
| PIVOT-Rerank | 10% | 0.5684 | 0.5448 ± 0.0038 | 41.78 | 2547 |
| PIVOT-Rerank | 20% | 0.5590 | 0.5384 ± 0.0060 | 70.96 | 3573 |

---

## 2. Frontier cơ sở (PPR-only) — deliverable T7-8 thuần

Frontier trích theo **Valid MRR** (không bị trội). Mỗi điểm vào frontier khi Valid MRR của nó cao hơn mọi điểm có latency nhỏ hơn.

| θ | Valid MRR | Test MRR | Latency/q (ms) |
|:---:|:---:|:---:|:---:|
| 1% | 0.4840 | 0.5361 ± 0.0030 | 6.84 |
| 5% | 0.5012 | 0.5369 ± 0.0042 | 16.19 |

> **Ghi chú frontier cơ sở:** Chỉ có θ=1% và θ=5% vào frontier vì θ=10% có Valid MRR 0.5003 < 0.5012 của θ=5% (bị trội). Điều này khác với WN18RR (θ=10% vào frontier) và phản ánh đặc điểm của NELL: subgraph thưa hơn, tăng budget vượt ngưỡng 5% không cải thiện thêm Valid MRR cho PPR-only.

---

## 3. Frontier đầy đủ — deliverable tích hợp T7-9

Trích frontier từ tất cả phương pháp (PPR + Rerank), sắp xếp tăng dần theo latency.

| Phương pháp | θ | Valid MRR | Test MRR | Latency/q (ms) | VRAM (MB) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| PPR-only | 1% | 0.4840 | 0.5361 ± 0.0030 | 6.84 | 395 |
| PIVOT-Rerank | 1% | 0.5357 | 0.5414 ± 0.0044 | 15.17 | 395 |
| PIVOT-Rerank | 5% | 0.5655 | 0.5468 ± 0.0042 | 27.77 | 1715 |
| PIVOT-Rerank | 10% | 0.5684 | 0.5448 ± 0.0038 | 41.78 | 2547 |

> **Nhận xét frontier đầy đủ:** PIVOT-Rerank chiếm 3 trong 4 điểm frontier — chứng tỏ reranking cải thiện đáng kể Valid MRR tại mọi mức budget từ θ=1% (+0.0517) đến θ=10% (+0.0681). Đây là gradient Valid MRR rất lớn so với WN18RR (+0.0028 đến +0.0040), phản ánh sự bổ trợ mạnh của bước reranking trên đồ thị NELL dày đặc hơn.

---

## 4. Demo BudgetController — NELL-995

Controller chọn theo **Valid MRR**, báo cáo **Test MRR** để đánh giá khách quan.

### Truy vấn 1: Latency ≤ 30 ms → Best Valid MRR

```
python3 pareto_optimizer.py --cache_path grid_t78_nell/pareto_cache_nell_v2.json --max_latency 30.0
```

**Kết quả:** PIVOT-Rerank @ θ=5%
- Valid MRR: `0.5655` | Test MRR: `0.5468 ± 0.0042`
- Latency/q: `27.77 ms` | VRAM: `1715 MB`

### Truy vấn 2: Min Valid MRR ≥ 0.568 → Min Latency

```
python3 pareto_optimizer.py --cache_path grid_t78_nell/pareto_cache_nell_v2.json --min_mrr 0.568
```

**Kết quả:** PIVOT-Rerank @ θ=10%
- Valid MRR: `0.5684` | Test MRR: `0.5448 ± 0.0038`
- Latency/q: `41.78 ms` | VRAM: `2547 MB`

**Phân tích:** Hai truy vấn trả về hai điểm khác nhau (θ=5% và θ=10%) — minh họa đúng tinh thần trade-off của BudgetController.

> ⚠️ **Valid/Test gap — minh họa cụ thể qua demo:** Đáng chú ý, cấu hình được chọn bởi truy vấn "accuracy-constrained" (θ=10%, Valid `0.5684`) có **Test MRR thấp hơn** (0.5448) so với cấu hình của truy vấn "latency-constrained" (θ=5%, Test 0.5468). Trong không gian Test, cấu hình "accuracy-constrained" bị trội bởi cấu hình kia — trong khi đắt hơn 14 ms và tốn thêm 832 MB VRAM. Đây không phải lỗi của controller — controller trung thành với protocol **chọn-theo-Valid** và không nhìn vào Test khi quyết định. Sự nghịch lý là hệ quả trực tiếp của hiện tượng double-dipping vào tập valid 543 query (đã được dùng để early-stop MLP và chọn α*). **Mọi claim hiệu năng trong KLTN đều dựa trên cột Test MRR.**

---

## 5. Cross-check với kết quả các tuần trước

| Checkpoint | Metric | Sweep Tuần 9 (FP32) | §7 Grid T7-8 | Chênh lệch |
|:---|:---|:---:|:---:|:---:|
| seed42, θ=10% | VRAM | 2585 MB | 2547 MB (mean 3 seed) | ~38 MB ✅ |
| PPR θ=10% | Test MRR | ~0.535 | 0.5354 ± 0.0036 | ≤ 0.003 ✅ |
| Rerank θ=10% | Test MRR | ~0.545 | 0.5448 ± 0.0038 | ≤ 0.003 ✅ |

> **Ghi chú latency:** Latency/q trong `reports/artifacts/nell/budget_results/` đo tại bs=16; §7 đo tại bs=8 → §7 nhỏ hơn ~2×. Ngoài ra, quy ước chia latency cũng khác (Tuần 6: `eval valid+test ÷ n_valid+n_test`, §7: `test-only ÷ 7,984`). Số chuẩn hoá trong §7 là tham chiếu chính thức.

---

## 6. Nhận xét phân tích chuyên sâu

Phân tích Pareto trên NELL-995 tiếp tục khẳng định hiệu quả vượt trội của PIVOT-Rerank:

1. **Mức tăng Valid MRR rất lớn:** Tại θ=1%, PIVOT-Rerank cải thiện Valid MRR từ 0.4840 lên 0.5357 (+0.0517 = +10.7%). Đây là mức cải thiện lớn hơn nhiều so với WN18RR (+0.0028), cho thấy PPR-only đặc biệt kém hiệu quả trên NELL do đồ thị dày đặc nhiều hub-node.

2. **Tốc độ hội tụ theo budget:** Valid MRR của PIVOT-Rerank hội tụ nhanh — từ θ=5% (0.5655) đến θ=10% (0.5684) chỉ tăng 0.0029, trong khi latency tăng từ 27.77 ms lên 41.78 ms (+50%). Điểm θ=5% là tối ưu Pareto rõ ràng nhất.

3. **Valid/Test gap đáng kể:** Mức cải thiện Valid không phản ánh đầy đủ cải thiện Test (ví dụ: +0.0681 valid vs +0.0093 test tại θ=10%). Đây là limitation của tập validation NELL-995 nhỏ (543 query) đã được double-dip trong pipeline. **Số Test mới là số đáng tin cậy để báo cáo**.

4. **Hybrid NELL-995:** Chưa chạy (tùy chọn). Lệnh để bổ sung: `INCLUDE_HYBRID=1 bash run_grid_t78.sh nell` (24 lượt thêm). Kết quả WN18RR gợi ý Hybrid sẽ giảm VRAM ~27% tại θ=1% và tăng tốc frontier ở budget nhỏ.