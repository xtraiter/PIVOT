# CODE_AUDIT — Kiểm toán & dọn dẹp mã nguồn (2026-07-21)

Nguyên tắc: **giữ nguyên layout phẳng của 8 file lõi kế thừa repo gốc** (giúp diff-vs-upstream trong `reports/changes_summary.md` còn giá trị đối chiếu), tách phần PIVOT bổ sung vào `experiments/` (script chạy thí nghiệm) và `analysis/` (tổng hợp/truy vấn kết quả). Mọi lệnh gọi từ **repo root**.

## Bảng phán quyết từng file

| File | Trước → Sau | Phán quyết |
|:--|:--|:--|
| `learned_pruning.py` | **304 → 40 dòng** | **Cắt 5/7 hàm chết** (`build_candidate_features` bản 4-feature legacy, `listwise_bce_loss`, `pairwise_hinge_loss`, `train_pruning_model`, `recall_at_k_after_pruning` — không nơi nào gọi). Giữ đúng 2 thứ được dùng: `PruningMLP`, `prune_candidates`. Docstring 60 dòng "kể chuyện" → 8 dòng mô tả đúng vai trò. |
| `train_auto.py` | 231 → 195 | **Xóa hàm chết `git_push_update`** (30 dòng auto-commit đã bị vô hiệu bằng `return`) + 2 call-site. |
| `PPR_sampler.py` | 426 → 424 | Bỏ `import copy` trùng, `import logging` và `coo_matrix` không dùng. Logic Hybrid 50/50 + trích 7 đặc trưng **giữ nguyên** (đã kiểm khớp tài liệu). |
| `base_model.py` | 392 → 390 | Dọn mẫu `if X: pass / else:` → `if not X:`. AMP/profiling/`_post_hoc_rerank` giữ nguyên. |
| `run_learned_pruning_wn18rr.py` / `_nell.py` | −15 dòng/file | **Xóa bản sao chép `get_hop_distances`** (định nghĩa trùng với `PPR_sampler`) → import dùng chung; thêm 3 dòng bootstrap `sys.path` để chạy từ `experiments/`. Phần trích đặc trưng train-time cố ý giữ riêng (khác ngữ cảnh đồ thị với inference-time). |
| `sweep_alpha_wn18rr.py` + `_fact95.py` | **2 file (384) → 1 file (197)** | Hai bản chỉ khác 5 dòng (fact_ratio + tên output) → **gộp**, thêm `--fact_ratio {0.75,0.95}`, tên output suy ra tự động (tái tạo được cả hai artifact lịch sử). |
| `budgeted_protocol.py` | ~0 | `--train_script` default resolve theo repo-root (chạy được từ mọi cwd). |
| `showResults.py` | −2 | Bỏ `exit()` trần cuối file. |
| `model.py`, `load_data.py`, `utils.py`, `search_auto.py`, `base_HPO.py` | 0 | **Giữ nguyên** — file lõi/HPO kế thừa repo gốc, không thuộc phạm vi "code PIVOT thêm vào". |
| `build_pareto.py`, `build_robustness.py`, `pareto_optimizer.py` | ~0 | Chỉ đổi default `--dir/--clean_dir` sang `reports/grid_t78_*`, `reports/robustness_t10` (khớp vị trí minh chứng trong repo). |
| `run_grid_t78.sh`, `run_robustness_t10.sh` | ~0 | Output/tham chiếu trỏ về `reports/…`; hint cuối trỏ `analysis/build_robustness.py`; `PY=${PY:-python3}`. **Resume-safe giữ nguyên** — log minh chứng sẵn có trong `reports/` được nhận diện skip đúng. |

**Tổng code PIVOT giảm ~520 dòng (−10%)**, không đổi bất kỳ hành vi tính toán nào (mọi sửa là xóa code chết / gộp trùng lặp / tham số hóa đường dẫn). Toàn bộ `.py` qua `py_compile`, cả hai `.sh` qua `bash -n`.

## Những thứ CỐ Ý không đụng
1. **8 file lõi upstream giữ layout phẳng ở root** — đây là fork nghiên cứu; giữ layout gốc làm câu chuyện "kế thừa + diff tường minh" mạnh hơn một cấu trúc src/ tự chế.
2. **Trích đặc trưng lặp giữa `PPR_sampler` (inference) và `run_learned_pruning_*` (train)** — trùng ~40 dòng nhưng chạy trên hai đồ thị khác nhau (observed-graph vs fact-graph fact_ratio=0.75); hợp nhất sẽ tăng rủi ro sai protocol để tiết kiệm ít dòng.
3. **`reports/changes_summary.md` giữ nguyên nội dung lịch sử** (chỉ sửa link tương đối) — đây là hồ sơ diff, không phải code sống.
