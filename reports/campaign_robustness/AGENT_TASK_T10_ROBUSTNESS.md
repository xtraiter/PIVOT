# NHIỆM VỤ AGENT — TUẦN 10: ROBUSTNESS SUITE (WN18RR)

> **Câu lệnh khởi động:** *"Đọc `.agents/AGENTS.md`, walkthrough (§0, §1, §7, §8)
> và toàn bộ file này, rồi thực thi Phase 0 → 5. Tuân thủ QUY TẮC CỨNG và
> DỪNG đúng điều kiện STOP."*

---

## BỐI CẢNH

Câu hỏi khoa học: *khi KG quan sát bị khuyết cạnh, cơ chế nào chống chịu tốt
hơn — PPR thuần hay PIVOT-Rerank?* Giả thuyết: feature thống kê toàn cục của
MLP (`tail_freq_q`, `rel_match`) ít nhạy với mất cạnh cục bộ hơn PPR (vốn phụ
thuộc đường đi), nên degradation curve của Rerank thoải hơn; khác biệt kỳ vọng
rõ nhất ở **reldel** (đánh vào relation hiếm — vùng PPR mù).

**CẢ BA KỊCH BẢN ĐỀU LÀ KẾT QUẢ HỢP LỆ** — ghi nhận trung thực, không "cứu"
số liệu:

- **(a)** Rerank degrade chậm hơn → giả thuyết được ủng hộ
- **(b)** Hai đường song song → cải thiện bền vững dưới nhiễu
- **(c)** Rerank degrade nhanh hơn → finding về distribution shift của learned component

Không kịch bản nào là "thất bại".

**Thiết kế:** test-time perturbation, không retrain — xóa cạnh trong đồ thị quan
sát (facts∪train), giữ nguyên GNN/MLP checkpoint train trên đồ thị sạch; α*
giữ per-seed từ Tuần 9 (seed42→0.8, seed123→0.6, seed1234→0.7), KHÔNG tune lại
trên đồ thị nhiễu. Điểm clean tái dùng từ grid θ=10% — không chạy lại.

**Công cụ** (đặt ở gốc repo):
- `make_perturbed_datasets_v2.py` — tạo dataset nhiễu (bản v2, sửa lỗi thiết kế)
- `run_robustness_t10.sh` — điều phối chạy eval
- `build_robustness.py` — tổng hợp và vẽ biểu đồ
- File này (`AGENT_TASK_T10_ROBUSTNESS.md`) — briefing này

---

## QUY TẮC CỨNG

1. FP32 (`--no_amp`), test-only (`--eval_split test`) — script đã cài sẵn.
2. Rerank = CHỈ `--pruning_model_path` + `--rerank_alpha`; **tuyệt đối không** `--use_learned_pruning`.
3. **PPR cache phải MỚI trên từng đồ thị nhiễu.** Không copy/symlink `ppr_scores/` từ WN18RR sạch vào thư mục nhiễu dưới bất kỳ hình thức nào. Dấu hiệu nhiễm: MRR trên đồ thị nhiễu ≥ clean + 0.005.
4. Không bịa số; mọi số vào báo cáo phải nằm trong `robustness_agg.csv`.
5. Không sửa code ngoài **một patch được phê duyệt trước** ở Phase 0.3.
6. Script robustness bản CŨ (`make_perturbed_datasets.py`, `run_robustness_suite.sh`) — đánh dấu deprecated ngay Phase 0.3, KHÔNG dùng.

---

## ĐIỀU KIỆN STOP

- Đĩa trống < 50GB (Phase 0.1).
- Sanity FAIL (Phase 1).
- Cảnh báo "CANH BAO: nghi lan cache" từ `build_robustness.py`.
- Bất kỳ lượt nào crash quá 1 lần retry; hoặc OOM.

---

## PHASE 0 — Chuẩn bị (~20 phút)

### 0.1 Cổng dung lượng đĩa

Mỗi config nhiễu tốn ~19GB PPR cache:

```bash
df -h --output=avail /home | tail -1
```

| Dung lượng trống | Hành động |
|:---|:---|
| **≥ 100 GB** | Chạy đủ 4 config: `del05 del10 del20 reldel` |
| **50–100 GB** | Chạy 2 config: `del10 reldel` (ghi rõ lý do vào báo cáo; 2 điểm nhiễu vẫn đạt deliverable) |
| **< 50 GB** | **STOP** — báo người dùng dọn đĩa |

### 0.2 Tạo dataset nhiễu

Thay danh sách config theo kết quả 0.1:

```bash
python3 make_perturbed_datasets_v2.py --data_path ./data/WN18RR --seed 42 \
    --configs del05 del10 del20 reldel
```

Kiểm tra sau khi chạy:
1. File `reports/campaign_robustness/WN18RR_perturbation_summary.txt` có số `deleted` khớp tỷ lệ danh nghĩa (±1 triple)
2. `diff data/WN18RR/test.txt data/WN18RR_del10/test.txt` → **rỗng**
3. `ls data/WN18RR_del10/ppr_scores/ 2>/dev/null` → **không tồn tại**

### 0.3 Patch 1 dòng được phê duyệt trước

`train_auto.py` dispatch params theo tên dataset và sẽ `exit()` với tên thư mục `WN18RR_del10`. Sửa đúng **1 dòng**:

```diff
-    if dataset == 'WN18RR':
+    if dataset.startswith('WN18RR'):
```

Ghi diff này vào `changes_summary.md` (mục Tuần 10) **ngay sau khi sửa**.

Đồng thời thêm comment deprecated vào đầu các script cũ nếu còn tồn tại:

```python
# DEPRECATED (Tuan 10-07-2026): su dung make_perturbed_datasets_v2.py thay the.
# Loi thiet ke: ban nay chi xoa trong train.txt; do thi quan sat = facts UNION train.
```

---

## PHASE 1 — Sanity (~1-2h, lần đầu kích hoạt PPR precompute)

```bash
bash run_robustness_t10.sh sanity
```

> ⚠️ **LƯU Ý:** Lượt sanity kích hoạt PPR precompute cho `del10` (~40,943 file pkl, tốn hàng chục phút đến ~1-2h). **Đây là hành vi đúng — không kill process.**

**PASS** khi script in `SANITY PASS`:
- `ppr_scores/` trong thư mục nhiễu đủ **40,943 file pkl**
- Test MRR seed42 < **0.5647** (thấp hơn clean seed42=0.5643 tại θ=10%)
- `use_learned_pruning=False`

**FAIL → STOP ngay.**

---

## PHASE 2 — Chạy đủ (nền, có thể chạy qua đêm)

```bash
nohup bash run_robustness_t10.sh run > rob_master.log 2>&1 &
echo "PID: $!"
```

Khối lượng: `N_config × 2 phương pháp × 3 seed` lượt eval (~3 phút/lượt sau khi
PPR của config đó đã có; lượt đầu mỗi config gánh PPR precompute).

Resume-safe: gián đoạn thì chạy lại đúng lệnh, script skip log đã có `[TEST]`.

Đếm log khi xong: `ls campaign_robustness/robustness_t10/rob_*.log | wc -l` → cần = `N_config × 6`.

---

## PHASE 3 — Tổng hợp & Cross-check (~10 phút)

```bash
python3 build_robustness.py --dir robustness_t10 --clean_dir grid_t78_wn18rr
```

**Cross-check bắt buộc:**

1. Hàng `clean` trong `robustness_agg.csv`:
   - `ppr ≈ 0.5638 ± 0.0017` (đọc từ chính log grid — phải khớp tuyệt đối)
   - `rerank ≈ 0.5685 ± 0.0021` (idem)
2. Không có cảnh báo `"nghi lan cache"` từ script.
3. Tính đơn điệu mềm: `MRR(del05) ≥ MRR(del10) ≥ MRR(del20)` cho từng phương pháp (cho phép vi phạm ≤ 1 bậc trong phạm vi 1 std — ghi nhận nếu có).

**Vi phạm mục 1 hoặc 2 → STOP.**

---

## PHASE 4 — Cập nhật tài liệu

### 4.1 Walkthrough — mục mới "Tuần 10 — Robustness Suite ✅"

Đặt sau mục Tuần 9, trước "So Sánh Tổng Thể". Gồm:

- **Thiết kế** với **3 ghi chú protocol bắt buộc**:
  1. *Filtered-ranking dùng filter của dataset nhiễu* (triple bị xóa không còn bị filter) — cả hai phương pháp chịu cùng protocol nên so sánh công bằng, cùng cách làm với Table 16 paper gốc.
  2. *Feature MLP tính trên đồ thị nhiễu* (đúng thiết kế test-time robustness).
  3. *Nếu chỉ chạy 2 config:* ghi rõ lý do dung lượng đĩa.
- **Bảng degradation** + **bảng khoảng cách Rerank−PPR** copy từ `campaign_robustness/robustness_t10/robustness_report_wn18rr.md`.
- **Nhúng** `campaign_robustness/robustness_t10/figure_robustness_wn18rr.png`.
- **Đoạn nhận xét 5–8 câu** CHỈ dựa trên `robustness_agg.csv` — nêu rõ kịch bản (a)/(b)/(c) và số liệu ô `reldel`.

### 4.2 §0 và §3 walkthrough

- **§0 bảng trạng thái:** Tuần 10 ⬜ → ✅
- **§3:** thêm dòng `campaign_robustness/robustness_t10/` (N log + agg csv + figure) và `reports/campaign_robustness/WN18RR_perturbation_summary.txt`

### 4.3 changes_summary.md — mục Tuần 10

- 3 file công cụ mới
- Patch 1 dòng `train_auto.py` (diff đầy đủ)
- Danh sách artifact sinh ra
- Ghi chú: script robustness bản CŨ đánh dấu deprecated

---

## PHASE 5 — Báo cáo kết thúc (≤ 20 dòng)

Báo cáo phải trả lời đủ:

1. Số config đã chạy + tổng thời gian (tách riêng PPR precompute vs eval)
2. Bảng cross-check pass/fail
3. Kết quả kịch bản (a)/(b)/(c), số liệu: Δ(Rerank − PPR) tại clean, del20, reldel
4. Bất thường
5. File đã cập nhật
6. Dung lượng đĩa đã dùng thêm + câu hỏi: *"Có xóa `data/WN18RR_del*/ppr_scores` sau khi chốt số không? (cache ~19GB/config, log là đủ truy vết)"*
