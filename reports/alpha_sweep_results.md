# Kết Quả Sweep Tham Số α — Post-hoc Reranking (WN18RR, Seed 42)

**Mục đích:** Tìm giá trị α tối ưu cho công thức kết hợp điểm số cuối cùng:

  Final_Score(i) = (1 - α) × Score_GNN(i)  +  α × Score_MLP_norm(i)

**Checkpoint GNN:** `data/WN18RR/saveModel/topk_0.1_layer_8_ValMRR_0.564_seed42.pt`  
**Checkpoint MLP:** `data/WN18RR/budget_results/pruning_mlp_v2_best_seed_42.pt`  
**GPU:** RTX 5060 Ti | **Topk:** 10% | **Dataset:** WN18RR (test=3,034 queries)

---

## Bảng Kết Quả Sweep Tham Số α

| α | Valid MRR | **Test MRR** | Test H@1 | Test H@10 | Eval Time | Peak GPU | Trạng thái |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 0.5644 | 0.5644 | 51.18% | 66.34% | 164.7s | 1469.79 MB | DONE |
| 0.1 | 0.5671 | 0.5667 | 51.47% | 66.77% | 227.4s | 1469.79 MB | DONE |
| 0.2 | 0.5675 | 0.5676 | 51.66% | 66.75% | 236.7s | 1469.79 MB | DONE |
| 0.3 | 0.5677 | 0.5682 | 51.75% | 66.82% | 254.9s | 1469.79 MB | DONE |
| 0.4 | 0.5681 | 0.5685 | 51.80% | 66.91% | 242.7s | 1469.79 MB | DONE |
| 0.5 | 0.5679 | 0.5691 | 51.88% | 66.86% | 239.1s | 1469.79 MB | DONE |
| 0.6 | 0.5686 | 0.5689 | 51.83% | 66.98% | 237.5s | 1469.79 MB | DONE |
| 0.7 | 0.5687 | 0.5694 | 51.90% | 66.98% | 240.3s | 1469.79 MB | DONE |
| 0.8 ⭐ | 0.5688 | **0.5696** | 51.96% | 67.07% | 235.0s | 1469.79 MB | DONE |
| 0.9 | 0.5663 | 0.5682 | 51.80% | 66.78% | 226.8s | 1469.79 MB | DONE |

> **Mốc báo cáo gốc (PPR):** Test MRR = `0.567` (từ README.md của tác giả)
> **Tốt nhất hiện tại (α=0.8):** Test MRR = `0.5696` — **Vượt kết quả báo cáo gốc** ✨

---

## Log Chi Tiết Từng Lần Chạy

### α = 0.0 — DONE ✅
File log: `data/WN18RR/results/2026-07-03-09:05:00_baseline.txt`
```
[VALID] MRR:0.564374 H@1:0.511000 H@10:0.661500
[TEST]  MRR:0.564407 H@1:0.511800 H@10:0.663400
[LATENCY] eval_total_ms:164700.00
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.1 — DONE ✅
File log: `data/WN18RR/results/2026-07-03-09:00:01.txt`
```
[VALID] MRR:0.567135 H@1:0.514500 H@10:0.664100
[TEST]  MRR:0.566691 H@1:0.514700 H@10:0.667700
[LATENCY] eval_total_ms:227400.00
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.2 — DONE ✅
File log: `data/WN18RR/results/task log`
```
[VALID] MRR:0.567477 H@1:0.514800 H@10:0.664100
[TEST]  MRR:0.567561 H@1:0.516600 H@10:0.667500
[LATENCY] eval_total_ms:236700.00
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.3 — DONE ✅
File log: `data/WN18RR/results/2026-07-05-00:21:09.txt`
```
[VALID] MRR:0.567725 H@1:0.515300 H@10:0.664100
[TEST]  MRR:0.568161 H@1:0.517500 H@10:0.668200
[LATENCY] eval_total_ms:254900.00
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.4 — DONE ✅
File log: `data/WN18RR/results/2026-07-05-00:37:08.txt`
```
[VALID] MRR:0.568096 H@1:0.516600 H@10:0.664800
[TEST]  MRR:0.568471 H@1:0.518000 H@10:0.669100
[LATENCY] eval_total_ms:242700.00
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.5 — DONE ✅
File log: `data/WN18RR/results/2026-07-05-00:52:03.txt`
```
[VALID] MRR:0.567923 H@1:0.515700 H@10:0.664500
[TEST]  MRR:0.569056 H@1:0.518800 H@10:0.668600
[LATENCY] eval_total_ms:239100.00
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.6 — DONE ✅
File log: `data/WN18RR/results/2026-07-05-01:15:11.txt`
```
[VALID] MRR:0.568577 H@1:0.516600 H@10:0.665500
[TEST]  MRR:0.568948 H@1:0.518300 H@10:0.669800
[LATENCY] eval_total_ms:237500.00
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.7 — DONE ✅
File log: `data/WN18RR/results/2026-07-05-01:30:08.txt`
```
[VALID] MRR:0.568661 H@1:0.516800 H@10:0.666000
[TEST]  MRR:0.569404 H@1:0.519000 H@10:0.669800
[LATENCY] eval_total_ms:240300.00
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.8 — DONE ✅
File log: `data/WN18RR/results/2026-07-05-01:45:17.txt`
```
[VALID] MRR:0.568756 H@1:0.516800 H@10:0.664600
[TEST]  MRR:0.569603 H@1:0.519600 H@10:0.670700
[LATENCY] eval_total_ms:235000.00
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.9 — DONE ✅
File log: `data/WN18RR/results/2026-07-05-01:59:41.txt`
```
[VALID] MRR:0.566297 H@1:0.514300 H@10:0.664000
[TEST]  MRR:0.568156 H@1:0.518000 H@10:0.667800
[LATENCY] eval_total_ms:226800.00
[PEAK_GPU_MEM] 1469.79MB
```

---

## Phân Tích Xu Hướng

| Khoảng α | Thay đổi Test MRR | Nhận xét |
|:---------|:-----------------:|:---------|
| 0.0 → 0.1 | +0.0023 (+0.40%) | Tăng |
| 0.1 → 0.2 | +0.0009 (+0.15%) | Tăng |
| 0.2 → 0.3 | +0.0006 (+0.11%) | Tăng |
| 0.3 → 0.4 | +0.0003 (+0.05%) | Tăng |
| 0.4 → 0.5 | +0.0006 (+0.10%) | Tăng |
| 0.5 → 0.6 | -0.0001 (-0.02%) | Giảm |
| 0.6 → 0.7 | +0.0005 (+0.08%) | Tăng |
| 0.7 → 0.8 | +0.0002 (+0.03%) | **Tăng - đạt điểm cực trị!** |
| 0.8 → 0.9 | -0.0014 (-0.25%) | Giảm |

**Quan sát nổi bật:**
- Reranking giúp cải thiện độ chính xác liên tục từ α=0.0 đến α=0.8.
- Khi vượt quá điểm cực trị 0.8, chất lượng dự đoán bắt đầu đi xuống do trọng số của MLP lấn át và làm lu mờ cấu trúc liên kết đa bước của GNN.

---
Cập nhật lần cuối: 2026-07-05 (Hoàn tất toàn bộ sweep từ 0.0 đến 0.9)
