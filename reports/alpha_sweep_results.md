# Kết Quả Sweep Tham Số α — Post-hoc Reranking (WN18RR, Seed 42)

**Mục đích:** Tìm giá trị α tối ưu cho công thức kết hợp điểm số cuối cùng:

  Final_Score(i) = (1 - α) × Score_GNN(i)  +  α × Score_MLP_norm(i)

**Checkpoint GNN:** `data/WN18RR/saveModel/topk_0.1_layer_8_ValMRR_0.564_seed42.pt`  
**Checkpoint MLP:** `data/WN18RR/budget_results/pruning_mlp_v2_best_seed_42.pt`  
**GPU:** RTX 5060 Ti | **Topk:** 10% | **Dataset:** WN18RR (test=3,034 queries)

**Lệnh chạy:**
```bash
python3 train_auto.py --data_path ./data/WN18RR/ --only_eval \
    --weight ./data/WN18RR/saveModel/topk_0.1_layer_8_ValMRR_0.564_seed42.pt \
    --pruning_model_path ./data/WN18RR/budget_results/pruning_mlp_v2_best_seed_42.pt \
    --rerank_alpha <α> --gpu 0
```

---

## Bảng Kết Quả Tổng Hợp

| α | Valid MRR | **Test MRR** | Test H@1 | Test H@10 | Eval Time | Peak GPU | Trạng thái |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0.0 | 0.5644 | 0.5644 | 51.18% | 66.34% | 164.7s | 1469.79 MB | DONE |
| 0.1 | 0.5671 | 0.5667 | 51.47% | 66.77% | 227.4s | 1469.79 MB | DONE |
| 0.2 | 0.5675 | 0.5676 | 51.66% | 66.75% | 236.7s | 1469.79 MB | DONE |
| 0.3 ⭐ | 0.5677 | **0.5682** | 51.75% | 66.82% | 254.9s | 1469.79 MB | DONE |
| 0.4    |    —      |    —       |    —      |    —      |    —          |    —          | ⏳ RUNNING |
| 0.5    |    —      |    —       |    —      |    —      |    —          |    —          | ⏳ PENDING |

> **Mốc báo cáo gốc (PPR):** Test MRR = `0.567`
> **Tốt nhất hiện tại (α=0.3):** Test MRR = `0.5682` — **Vượt kết quả báo cáo gốc** ✨

---

## Log Chi Tiết Từng Lần Chạy

### α = 0.0 (Baseline — Không Reranking)
File log: `data/WN18RR/results/2026-07-03-09:05:00_baseline.txt`
```
[VALID] MRR:0.564374 H@1:0.511042 H@10:0.661503
[TEST]  MRR:0.564407 H@1:0.511806 H@10:0.663369
[LATENCY] eval_total_ms:164693.72 data_prep_ms:4968.22 forward_ms:103766.21 ranking_ms:50398.76
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.1
File log: `data/WN18RR/results/2026-07-03-09:00:01.txt`
```
[VALID] MRR:0.567135 H@1:0.514502 H@10:0.664140
[TEST]  MRR:0.566691 H@1:0.514678 H@10:0.667677
[LATENCY] eval_total_ms:227408.52 data_prep_ms:4981.74 forward_ms:167972.67 ranking_ms:48851.62
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.2
File log: task log (chạy trước khi có cơ chế ghi file --only_eval)
```
[VALID] MRR:0.567477 H@1:0.514832 H@10:0.664140
[TEST]  MRR:0.567561 H@1:0.516592 H@10:0.667518
[LATENCY] eval_total_ms:236712.51 data_prep_ms:5104.51 forward_ms:177307.07 ranking_ms:48488.19
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.3 — DONE ✅
File log: `data/WN18RR/results/2026-07-05-00:21:09.txt`
```
[VALID] MRR:0.567725 H@1:0.515326 H@10:0.664140
[TEST]  MRR:0.568161 H@1:0.517549 H@10:0.668156
[LATENCY] eval_total_ms:254858.71 data_prep_ms:5570.44 forward_ms:180816.00 ranking_ms:61858.11
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.4 — DONE ✅
File log: `data/WN18RR/results/2026-07-05-00:37:08.txt`
```
[VALID] MRR:0.568096 H@1:0.516645 H@10:0.664799
[TEST]  MRR:0.568471 H@1:0.518028 H@10:0.669113
[LATENCY] eval_total_ms:242744.15 data_prep_ms:5552.55 forward_ms:179347.15 ranking_ms:51292.28
[PEAK_GPU_MEM] 1469.79MB
```

### α = 0.5 — ⏳ Đang chạy...
*(Kết quả sẽ được cập nhật)*

---

## Phân Tích Xu Hướng

| Khoảng α | Thay đổi Test MRR | Nhận xét |
|:---------|:-----------------:|:---------|
| 0.0 → 0.1 | +0.0023 (+0.41%) | MLP bổ sung ngữ nghĩa quan hệ rõ ràng |
| 0.1 → 0.2 | +0.0009 (+0.16%) | Tiếp tục tăng — cực trị đang đến gần |
| 0.2 → 0.3 | +0.0006 (+0.11%) | Xu hướng tăng vẫn duy trì |
| 0.3 → 0.4 | +0.0003 (+0.05%) | **Tiếp tục tăng nhẹ — vẫn chưa đạt cực trị!** |
| 0.4 → 0.5 | (chờ kết quả) | Dự đoán đây sẽ là cực trị hoặc bắt đầu giảm |

**Quan sát nổi bật:** MRR vẫn **tăng đơn điệu** từ α=0.0→0.3 (chưa đạt cực trị).
Xu hướng tăng đang chậm dần → cực trị có thể nằm trong [0.3, 0.4].

**Dự đoán lý thuyết:** Tại α > 0.4-0.5, thành phần MLP chiếm ưu thế sẽ làm mất tính multi-hop reasoning của GNN.

---

Cập nhật lần cuối: 2026-07-05 (α=0.4 hoàn tất, α=0.5 đang chạy)
