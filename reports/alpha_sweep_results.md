# Kết Quả Sweep Tham Số α — Post-hoc Reranking (WN18RR, Seed 42)

**Mục đích:** Tìm giá trị α tối ưu cho công thức kết hợp điểm số cuối cùng:

  Final_Score_i = (1 - α) × Score_GNN_i  +  α × Score_MLP_i

**Checkpoint GNN:** data/WN18RR/saveModel/topk_0.1_layer_8_ValMRR_0.564_seed42.pt
**Checkpoint MLP:** data/WN18RR/budget_results/pruning_mlp_v2_best_seed_42.pt
**GPU:** RTX 5060 Ti | **Topk:** 10% | **Dataset:** WN18RR (40,943 entities)

---

## Bảng Kết Quả Tổng Hợp

| α      | Valid MRR | Test MRR   | Test H@1  | Test H@10 | Eval Time (s) | Peak GPU (MB) | Trạng thái |
|:------:|:---------:|:----------:|:---------:|:---------:|:-------------:|:-------------:|:----------:|
| 0.0 (Baseline PPR) | 0.5644 | **0.5644** | 51.18% | 66.34% | 164.7 | 1469.79 | DONE |
| 0.1    | 0.5671    | **0.5667** | 51.47%    | 66.77%    | 227.4         | 1469.79       | DONE       |
| 0.2    | 0.5675    | **0.5676** | 51.66%    | 66.75%    | 236.7         | 1469.79       | DONE       |
| 0.3    |    —      |    —       |    —      |    —      |    —          |    —          | RUNNING... |
| 0.4    |    —      |    —       |    —      |    —      |    —          |    —          | PENDING    |
| 0.5    |    —      |    —       |    —      |    —      |    —          |    —          | PENDING    |

> **Mốc báo cáo gốc (PPR):** Test MRR = 0.567
> **Tốt nhất hiện tại (α=0.2):** Test MRR = 0.5676 — Vượt kết quả báo cáo gốc!

---

## Log Chi Tiết Từng Lần Chạy

### α = 0.0 (Baseline — Không Reranking)
File log: data/WN18RR/results/2026-07-03-090500_baseline.txt

  [VALID] MRR:0.564374 H@1:0.511042 H@10:0.661503
  [TEST]  MRR:0.564407 H@1:0.511806 H@10:0.663369
  [LATENCY] eval_total_ms:164693.72 data_prep_ms:4968.22 forward_ms:103766.21 ranking_ms:50398.76
  [PEAK_GPU_MEM] 1469.79MB

### α = 0.1
File log: data/WN18RR/results/2026-07-03-090001.txt

  [VALID] MRR:0.567135 H@1:0.514502 H@10:0.664140
  [TEST]  MRR:0.566691 H@1:0.514678 H@10:0.667677
  [LATENCY] eval_total_ms:227408.52 data_prep_ms:4981.74 forward_ms:167972.67 ranking_ms:48851.62
  [PEAK_GPU_MEM] 1469.79MB

### α = 0.2
File log: task-3336.log (trước khi có cơ chế ghi file --only_eval)

  [VALID] MRR:0.567477 H@1:0.514832 H@10:0.664140
  [TEST]  MRR:0.567561 H@1:0.516592 H@10:0.667518
  [LATENCY] eval_total_ms:236712.51 data_prep_ms:5104.51 forward_ms:177307.07 ranking_ms:48488.19
  [PEAK_GPU_MEM] 1469.79MB

### α = 0.3 — Đang chạy...
(Kết quả sẽ được cập nhật sau khi tiến trình nền hoàn tất)

### α = 0.4 — Đang chờ...
(Kết quả sẽ được cập nhật sau khi tiến trình nền hoàn tất)

### α = 0.5 — Đang chờ...
(Kết quả sẽ được cập nhật sau khi tiến trình nền hoàn tất)

---

## Phân Tích Xu Hướng

| Khoảng α        | Thay đổi MRR | Nhận xét |
|:----------------|:------------:|:---------|
| 0.0 → 0.1       | +0.0023 (+0.41%) | MLP bổ sung ngữ nghĩa quan hệ rõ ràng |
| 0.1 → 0.2       | +0.0009 (+0.16%) | Điểm cực trị đang tiến gần |
| 0.2 → 0.3+      | (chờ kết quả)    | Dự đoán MRR bắt đầu giảm |

**Dự đoán:** Điểm cực trị tối ưu nằm trong α ∈ [0.15, 0.35].
Tại α > 0.5, MLP lấn át GNN làm mất tính multi-hop reasoning.

---
Cập nhật lần cuối: 2026-07-05
