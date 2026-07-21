# FIXES_APPLIED — Nhật ký sửa lỗi tài liệu & vệ sinh repo (2026-07-21)

## reports/PIVOT_reproduction_walkthrough.md (812 → 698 dòng)
1. **Xóa mục "Tuần 10 — Robustness Suite" trùng lặp thứ hai** (chứa bộ số Hybrid không khớp CSV: 0.5152/0.4766/0.4199/0.5663). Bản giữ lại khớp chính xác `reports/csv_deliverables/TaskA_Robustness_Summary.csv` và `reports/robustness_t10/robustness_agg.csv`.
2. **Xóa khối NELL Tuần 7–8 cũ trùng lặp** (mảnh bảng vỡ + đoạn "Hybrid NELL-995: [CHƯA THỰC HIỆN — tùy chọn]" + demo controller phiên bản cũ) — mâu thuẫn với kết quả Hybrid NELL đã chạy đầy đủ ở khối mới phía trên.
3. **Sửa bảng 7 đặc trưng (Tuần 9)**: gỡ `in_degree_q` (không tồn tại trong mã nguồn — đã audit theo góp ý GVHD) và các tên cũ; thay bằng bảng khớp code: `ppr_log, ppr_rank_pct, deg_log, hop_dist, is_direct, tail_freq_q, rel_match`; kèm ghi chú đính chính.
4. **Viết lại mục Sweep α theo đúng protocol chọn-trên-Valid**: gỡ toàn bộ nội dung chọn "đỉnh theo Test" (α=0.82 ⭐, bảng "So Sánh Đỉnh Cực Trị", claim "+0.0029/✨"); thay bằng bảng α* per-seed (0.8/0.6/0.7) với kết quả chính thức 0.5685 ± 0.0021; bảng sweep giữ lại và dán nhãn "phân tích hậu nghiệm".
5. **Viết lại mục "So Sánh Với Checkpoint Tác Giả"**: gỡ mọi chữ "Vượt trội"; số PIVOT dùng α* theo Valid (0.5685 ± 0.0021 → "tương đương trong sai số, +0.0008"); claim tốc độ giữ nguyên nhưng ghi rõ là tối ưu kỹ thuật I/O.
6. **Đổi tiêu đề** "(Tuần 1–9)" → "(Tuần 1–10 + Feature Ablation)".
7. **Chuyển toàn bộ liên kết/ảnh từ đường dẫn tuyệt đối `file:///home/vanba/...` sang đường dẫn tương đối** (23 liên kết + 4 ảnh) — trước đây vỡ hết trên GitHub; nay 4/4 ảnh hiển thị: `grid_t78_wn18rr/figure1_frontier_wn18rr.png`, `grid_t78_nell/figure1_frontier_nell.png`, `figures/figure_feature_ablation.png`, `figures/figure_robustness_wn18rr.png`. Lệnh train mẫu đổi `/home/vanba/miniconda3/.../python3` → `python3`.

## reports/csv_deliverables/TaskB_NELL_Tier1_Tier2_Summary.csv
- Sửa nhãn cột sai "Tier-2 **Valid** MRR" → "Tier-2 **Test** MRR" (giá trị vốn là Test).
- Hàng PPR: thay 0.5361 ± 0.0 (số PPR test @θ=1%, std=0 vô lý) bằng **0.5354 ± 0.0036** (PPR test @θ=10%, khớp bảng grid chính thức).
- Chuẩn lại std các biến thể theo bộ số nghiệm thu (V2 0.5452±0.0041, V4 0.5453±0.0035, L3 0.5461±0.0041 — mean±std ddof=1 trên 3 seed tại α*).

## README.md
- Thay README nguyên bản của one-shot-subgraph bằng README PIVOT: đóng góp, bảng kết quả chính, sơ đồ repo, môi trường, lệnh tái lập 6 nhóm thí nghiệm, ghi công + BibTeX paper gốc.

## .gitignore
- Bỏ dòng `*.npz` lặp; gỡ ghi chú lộ hạ tầng (Vast.ai); ghi chú rõ checkpoint/log được commit có chủ đích làm minh chứng.

## .agents/AGENTS.md & reports/changes_summary.md
- Chuyển mọi liên kết `file:///home/vanba/...` sang đường dẫn tương đối (nội dung diff lịch sử giữ nguyên).

## run_robustness_t10.sh
- `PY=/home/vanba/miniconda3/...` → `PY=${PY:-python3}` để script chạy được trên máy khác (override qua biến môi trường).

## Khuyến nghị chưa thực hiện (tùy chọn)
- `git mv grid_wn18rr_master.log reports/artifacts/` (log 10MB đang nằm ở gốc repo).

## Cách áp dụng
Giải nén `PIVOT_fixed_files.zip` đè lên thư mục repo local, rồi:
```bash
git add -A && git commit -m "docs: fix duplicated sections, protocol-compliant alpha, correct feature table, relative paths; new README; fix NELL CSV; portable robustness script" && git push
```
