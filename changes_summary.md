# Tổng hợp các thay đổi đã thực hiện (Changes Summary)

Tất cả các tối ưu hóa, sửa lỗi tương thích và định dạng log được thực hiện trên mã nguồn gốc của bài báo `one-shot-subgraph` để phù hợp với hệ thống của bạn (Xeon 80 CPUs, RTX 5060 Ti 16GB, PyTorch 2.7+cu128).

---

## 11. Fix FutureWarning: Cập nhật AMP API (PyTorch 2.6+ Compatibility)
*   **File sửa đổi**: [base_model.py](file:///home/vanba/KLTN/one-shot-subgraph/base_model.py)
*   **Nguyên nhân**: Khi training bắt đầu chạy, log xuất hiện 2 `FutureWarning` từ PyTorch về API cũ.

### `BaseModel.__init__()` — dòng khởi tạo GradScaler
*   **Trước:** `self.scaler = torch.cuda.amp.GradScaler()`
*   **Sau:** `self.scaler = torch.amp.GradScaler('cuda')`
*   Từ PyTorch 2.6+, namespace `torch.cuda.amp` bị deprecated, dùng `torch.amp` với tham số device `'cuda'` tường minh.

### `BaseModel.train_batch()`, `BaseModel.evaluate()` — 3 dòng autocast
*   **Trước:** `with torch.cuda.amp.autocast():`
*   **Sau:** `with torch.amp.autocast('cuda'):` (áp dụng cho cả train, val, test forward pass)
*   Sửa cả 3 vị trí trong file để loại bỏ hoàn toàn namespace deprecated.

---
## 1. Song song hóa tính điểm PPR (PPR Score Parallelization)
*   **File sửa đổi**: [PPR_sampler.py](file:///home/vanba/KLTN/one-shot-subgraph/PPR_sampler.py)

### Hàm `_global_homo_graph` + `_init_worker(graph)`
*   Thêm biến global `_global_homo_graph = None` để các worker process có thể truy cập đồ thị NetworkX mà không cần pickle.
*   Thêm hàm `_init_worker(graph)` làm initializer cho `multiprocessing.Pool`, truyền đồ thị vào từng worker qua bộ nhớ shared — tránh overhead serialize/deserialize đồ thị lớn.

### Hàm `_compute_and_save_ppr_scores(h, ppr_save_path)`  *(hàm mới)*
*   Hàm worker chạy trong từng tiến trình con: tính `nx.pagerank(graph, personalization={h: 1})` cho entity `h`.
*   Kiểm tra trước `os.path.exists` và `os.path.getsize > 1000` để bỏ qua entity đã có cache hợp lệ.

### Hàm `pprSampler.__init__()` — phần tính PPR
*   Thay vòng lặp `for h in range(n_ent)` tuần tự bằng `multiprocessing.Pool(processes=num_workers, initializer=_init_worker, initargs=(graph,))`.
*   Số worker tự động: `num_workers = min(64, os.cpu_count() - 4)` — dùng tối đa 64 CPU trên máy Xeon của bạn.
*   Dùng `pool.imap_unordered` + `tqdm` để hiển thị thanh tiến trình realtime.
*   Lọc trước danh sách entity chưa có cache (`entities_to_compute`) để không chạy lại entity đã xong.

### Tối ưu hóa tải trước điểm PPR vào RAM (In-Memory Pre-loading & Parallel Loading)
*   **Vấn đề**: Việc DataLoader của PyTorch đọc và unpickle 16 file ppr riêng lẻ từ ổ cứng cho mỗi batch trong lúc train gây nghẽn cổ chai I/O cực lớn (tốc độ train chỉ đạt `7.99s/it`).
*   **Giải pháp**:
    - Khi khởi tạo `pprSampler` (trong `__init__`), nếu số thực thể `n_ent <= 50000` (như WN18RR), toàn bộ các file `.pkl` được đọc và gộp trực tiếp vào một ma trận NumPy 2D `self.all_ppr_scores` có kích thước `(n_ent, n_ent)`.
    - Thêm cơ chế cache dictionary `self.ppr_cache` cho các tập dữ liệu lớn hơn (>50,000 thực thể như YAGO) để tránh vượt giới hạn RAM.
    - Trong hàm `sampleSubgraph(ent)`, nếu đã preload vào RAM, điểm PPR được truy cập trực tiếp qua NumPy indexing (`self.all_ppr_scores[ent]`), triệt tiêu hoàn toàn I/O đĩa và unpickle ở mỗi step. Tốc độ huấn luyện tăng ~13 lần (đạt `1.6 it/s`).
    - **Tải song song (Parallel Preloading)**: Định nghĩa hàm helper `_load_one_ppr` chạy song song qua `multiprocessing.Pool` để unpickle 40,943 file WN18RR đồng thời. Giảm thời gian tải trước (preload/startup) từ 3 phút xuống còn chưa đầy 10 giây.

---

## 2. Cơ chế chống hỏng file cache PPR (Atomic Write & Integrity Check)
*   **File sửa đổi**: [PPR_sampler.py](file:///home/vanba/KLTN/one-shot-subgraph/PPR_sampler.py)

### Hàm `_compute_and_save_ppr_scores(h, ppr_save_path)`
*   Áp dụng **atomic write**: ghi điểm PPR ra file tạm (`tempfile.NamedTemporaryFile`) trước, sau khi `pkl.dump` thành công mới gọi `os.replace(temp_path, final_path)`.
*   Đảm bảo file `.pkl` không bao giờ ở trạng thái nửa chừng nếu process bị kill giữa lúc ghi.

### Hàm `pprSampler.__init__()` — phần kiểm tra integrity
*   Thêm điều kiện `os.path.getsize(path) < 1000` vào bước filter: file nhỏ hơn 1KB được coi là hỏng và tính lại.

---

## 3. Sửa lỗi ép kiểu Seed trên NumPy mới (Seed Type Fix)
*   **Files sửa đổi**: [train_auto.py](file:///home/vanba/KLTN/one-shot-subgraph/train_auto.py), [search_auto.py](file:///home/vanba/KLTN/one-shot-subgraph/search_auto.py)

### `ArgumentParser` — tham số `--seed`
*   Đổi `type=str` → `type=int` cho tham số `--seed`.
*   NumPy ≥ 2.0 ném `TypeError` khi gọi `np.random.seed("456")` với chuỗi. Fix này đảm bảo tương thích với NumPy 2.2.6 trở lên.

---

## 4. Khắc phục lỗi tương thích PyTorch 2.7 (PyTorch 2.7 Compatibility)
*   **File sửa đổi**: [base_model.py](file:///home/vanba/KLTN/one-shot-subgraph/base_model.py)

### Hàm khởi tạo `BaseModel.__init__()` — phần scheduler
*   Xóa tham số `verbose=True` khỏi lời gọi `torch.optim.lr_scheduler.ReduceLROnPlateau(...)`.
*   PyTorch ≥ 2.6 đã bỏ hoàn toàn tham số này — để lại sẽ gây `TypeError` crash ngay khi khởi tạo model.

---

## 5. Tối ưu hóa dung lượng bộ nhớ GPU (DataLoader GPU Memory Optimization)
*   **File sửa đổi**: [base_model.py](file:///home/vanba/KLTN/one-shot-subgraph/base_model.py)

### Hàm `worker_init_fn(worker_id)`  *(hàm mới)*
*   Gán `os.environ["CUDA_VISIBLE_DEVICES"] = ""` cho mỗi DataLoader worker process.
*   Ngăn các CPU worker vô tình tạo CUDA context trống trên GPU, tiết kiệm **2–8 GB VRAM**.
*   Hàm này được truyền vào `DataLoader(worker_init_fn=worker_init_fn)`.

### Phần cấu hình `DataLoader`
*   Tự động set `prefetch_factor=None` khi `--cpu 0` được truyền vào, tránh lỗi PyTorch khi `num_workers=0`.
*   Đặt `pin_memory=False` cho cả 3 DataLoader (`trainLoader`, `valLoader`, `testLoader`).
*   **Lý do**: Trên hệ thống WSL2, việc bật `pin_memory=True` kết hợp với tải lượng bộ nhớ lớn dễ gây lỗi phân bổ bộ nhớ khóa trang (page-locked memory) của CUDA driver, dẫn đến lỗi `RuntimeError: CUDA error: out of memory` gây crash ứng dụng. Tắt `pin_memory` sửa triệt để lỗi này mà không làm ảnh hưởng tốc độ huấn luyện.

---

## 6. Ghi Log chi tiết & Phân biệt Checkpoint theo Seed
*   **File sửa đổi**: [base_model.py](file:///home/vanba/KLTN/one-shot-subgraph/base_model.py)

### Hàm `BaseModel.train_batch()`
*   Đo và in 5 nhóm thời gian: `latency_ms` (tổng train), `data_prep_ms`, `forward_ms`, `ranking_ms`, `eval_total_ms`.
*   In `peak_gpu_mem_mb` bằng `torch.cuda.max_memory_allocated()`.
*   Định dạng log khớp với tag `[TRAIN]`, `[VALID]`, `[TEST]`, `[TIME]`, `[LATENCY]`, `[PEAK_GPU_MEM]` mà `PARSE_REGEX` trong `budgeted_protocol.py` đọc được.

### Hàm `BaseModel.saveModelToFiles(args, metric_str, ...)`
*   Chèn `_seed{args.seed}` vào tên file checkpoint: ví dụ `topk_0.1_layer_6_ValMRR_0.569_seed456.pt`.
*   Ngăn các lượt chạy seed khác nhau ghi đè file của nhau.

### Hàm `BaseModel.evaluate()`
*   Làm tròn MRR và Hits thành 6 chữ số thập phân (`%.6f`) thay vì 4.

---

## 7. Hỗ trợ Huấn luyện Nửa độ chính xác (Mixed Precision AMP)
*   **File sửa đổi**: [base_model.py](file:///home/vanba/KLTN/one-shot-subgraph/base_model.py)

### Hàm `BaseModel.__init__()`
*   Khởi tạo `torch.cuda.amp.GradScaler()` và lưu vào `self.scaler`.

### Hàm `BaseModel.train_batch()`
*   Bọc forward pass trong `with torch.cuda.amp.autocast():` để tự động cast sang `float16`.
*   Thay `loss.backward()` bằng `self.scaler.scale(loss).backward()`.
*   Thay `optimizer.step()` bằng `self.scaler.step(optimizer)` + `self.scaler.update()`.
*   Kết quả: VRAM giảm từ ~16.4 GB xuống ~9–10 GB, cho phép train `batchsize=16` trên GPU 16GB.



## 9. Budgeted Inference Benchmark (Tuần 6 kế hoạch)
*   **File tạo mới**: [budgeted_protocol.py](file:///home/vanba/KLTN/one-shot-subgraph/budgeted_protocol.py)

### Hằng số `PARSE_REGEX`
*   Biểu thức chính quy `re.compile(...)` với `re.DOTALL` để parse output của `train_auto.py`.
*   Bắt 5 nhóm: `TEST MRR`, `H@1`, `H@10`, `eval_total_ms`, `peak_gpu_mem MB`.
*   Cần khớp đúng với định dạng log tag `[TEST]`, `[LATENCY]`, `[PEAK_GPU_MEM]` trong `base_model.py`.

### Hàm `run_one(data_path, weight, gpu, topk, batchsize, extra_args, train_script)`
*   Gọi `sys.executable` (thay vì gọi cứng `"python3"`) để đảm bảo chạy đúng môi trường conda.
*   Build lệnh `--only_eval` với `--topk={topk}` tương ứng từng mức budget.
*   Capture stdout+stderr, apply `PARSE_REGEX`, raise `RuntimeError` kèm 1500 ký tự cuối nếu parse thất bại.
*   Trả về dict 6 trường: `budget, MRR, Hits@1, Hits@10, eval_total_ms, peak_gpu_mem_mb, wall_clock_s`.

### Hàm `run_budget_sweep(data_path, weight, gpu, budgets, n_queries, seeds, batchsize, out_csv, train_script)`
*   Vòng lặp 2 cấp: outer = seed, inner = budget → gọi `run_one()` cho mỗi tổ hợp.
*   Tính thêm 2 metric dẫn xuất: `throughput_qps = n_queries / (eval_ms/1000)` và `latency_per_query_ms = eval_ms / n_queries`.
*   Lưu `pd.DataFrame` ra `raw_results.csv`.

### Hàm `summarize(df, out_csv)`
*   `groupby("budget").agg(...)` tính mean±std trên tất cả seed cho 7 cột metric.
*   **Bảng kết quả chuẩn hóa**: Định dạng và đổi tên các cột sang định dạng tiếng Việt & Anh chuyên nghiệp theo yêu cầu: `Dataset`, `Phương pháp`, `Budget`, `Test MRR (Mean ± Std)`, `H@1 (Mean)`, `H@10 (Mean)`, `eval_total (ms)`, `data_prep (ms)`, `forward (ms)`, `ranking (ms)`, `Latency / query (ms)`, `Throughput (q/s)`, `Peak GPU Mem (MB)`.
*   Lưu ra `summary.csv` — đây là bảng dùng trực tiếp để điền vào **Table 2** của paper.

### Hàm `plot_pareto(summary_df, out_png)`
*   Vẽ đường MRR vs latency/query, mỗi điểm = 1 mức budget.
*   Annotate nhãn phần trăm (1%, 5%, 10%, 20%) cạnh từng điểm.
*   Lưu PNG 160 dpi → **Figure 1** trong kế hoạch paper.

### `__main__` — argparse
*   5 tham số bắt buộc/tùy chọn: `--data_path`, `--weight`, `--gpu`, `--n_queries`, `--budgets`, `--seeds`, `--batchsize`, `--outdir`, `--train_script`.
*   Thêm 2 tham số `--dataset` và `--method` để định danh và điền cột thông tin trong bảng summary.

---

## 12. Tự động nhận dạng số lớp (Auto Layer Count Detection)
*   **File sửa đổi**: [train_auto.py](file:///home/vanba/KLTN/one-shot-subgraph/train_auto.py)

### Hàm `run_model(params)`
*   Thêm logic tự động kiểm tra tham số `--weight` khi load checkpoint. Nếu tên file weight chứa chuỗi định dạng số lớp (như `layer_6` trong `WN18RR_topk_0.1_layer_6_ValMRR_0.569.pt`), script sẽ tự động thiết lập cấu hình `params['n_layer']` tương ứng.
*   Ngăn ngừa Model Shape Mismatch crash do các dataset có số lớp mặc định khác nhau (ví dụ: WN18RR mặc định là 8 lớp nhưng checkpoint chỉ có 6 lớp).

---

## 10. Learned Pruning cho PIVOT (Tuần 9 kế hoạch)
*   **File tạo mới**: [learned_pruning.py](file:///home/vanba/KLTN/one-shot-subgraph/learned_pruning.py)

### Hàm `build_candidate_features(candidate_ids, ppr_scores, node_degree, hop_distance, rel_match_score)`
*   Stack 4 đặc trưng thành ma trận `[N, 4]`: PPR score, `log1p(degree)`, hop distance, relation match score.
*   Chuẩn hóa theo batch (trừ mean, chia std + 1e-6) để MLP học ổn định.
*   **ADAPT**: thay placeholder bằng dữ liệu thực từ `PPR_sampler.py` (biến `self.homoTrainGraph`, degree từ đồ thị NetworkX...).

### Class `PruningMLP(nn.Module)`
*   Kiến trúc: `Linear(in_dim→32) → ReLU → Linear(32→16) → ReLU → Linear(16→1)`.
*   Mặc định `in_dim=4` khớp với `build_candidate_features()`.
*   `forward(feats)` trả về vector điểm `[N]` cho N candidates.
*   Tăng `in_dim` nếu thêm đặc trưng vào `build_candidate_features`.

### Hàm `listwise_bce_loss(scores, true_tail_mask)`
*   BCE đơn giản: label=1 cho true tail, label=0 cho tất cả negative.
*   Cùng dạng loss `L_cls` của paper gốc → dễ bảo vệ trước hội đồng.

### Hàm `pairwise_hinge_loss(scores, true_tail_idx, margin=1.0, n_negatives=20)`
*   Sample ngẫu nhiên 20 negative, yêu cầu `score(true) - score(neg) ≥ margin`.
*   Cho ranh giới top-K sắc nét hơn BCE khi candidate pool lớn.

### Hàm `train_pruning_model(model, optimizer, query_batches, n_epochs, use_pairwise)`
*   Vòng lặp train chuẩn: `zero_grad → forward → loss → backward → step`.
*   In loss mỗi epoch để theo dõi hội tụ.
*   `use_pairwise=False` (mặc định): dùng BCE. `True`: dùng hinge loss.
*   **ADAPT**: `query_batches` cần được build từ vòng lặp train trong `train_auto.py` — candidate pool từ PPR sampler + true tail entity.

### Hàm `prune_candidates(model, candidate_ids, features, budget_k, use_learned, ppr_scores)`  *(hàm cốt lõi)*
*   **`use_learned=True` (PIVOT)**: chạy `model.eval()` → `model(features)` → `topk(scores, budget_k)`.
*   **`use_learned=False` (PPR-only / ablation)**: dùng `ppr_scores` thay cho scores của MLP — giữ nguyên hành vi paper gốc.
*   Decorator `@torch.no_grad()` đảm bảo không tính gradient lúc inference.
*   **Cách tích hợp**: thay dòng `TopK(ppr_score)` trong `PPR_sampler.sampleSubgraph()` bằng lời gọi hàm này.

### Hàm `recall_at_k_after_pruning(model, eval_queries, budget_k, use_learned)`
*   Metric chính để so sánh hai chế độ: tỷ lệ query mà true tail entity còn sống sau khi prune xuống `budget_k`.
*   Gọi `prune_candidates` với `use_learned` tuỳ chỉnh, kiểm tra `true_id in kept_ids`.
*   **Deliverable Tuần 9**: so sánh `recall_at_k(use_learned=True)` vs `(use_learned=False)` ở cùng `budget_k`.

### `__main__` — argparse với `--mode`
*   **`--mode learned`**: chỉ chạy PIVOT (train MLP + prune bằng MLP).
*   **`--mode ppr`**: chỉ chạy PPR-only (không cần train model).
*   **`--mode compare`** (mặc định): chạy cả hai, in bảng so sánh trực tiếp.
*   Tham số bổ sung: `--budget_k`, `--n_candidates`, `--seed`.
*   **Kết quả smoke test đã xác nhận**: N=200, budget_k=10, seed=0 → MLP hội tụ `loss: 25.1→0.498`, learned pruning GIỮ true tail (id=3) còn PPR-only BỎ LỠ.

---

## Tóm tắt file theo thứ tự ảnh hưởng

| File | Trạng thái | Số hàm thay đổi/thêm |
|------|-----------|----------------------|
| `PPR_sampler.py` | Sửa đổi | 4 (thêm `_init_worker`, `_compute_and_save_ppr_scores`, `_load_one_ppr`, sửa `__init__`) |
| `base_model.py` | Sửa đổi | 4 (sửa `__init__`, `train_batch`, `evaluate`, `saveModelToFiles`) |
| `train_auto.py` | Sửa đổi | 2 (sửa `git_push_update`, `run_model`) |
| `search_auto.py` | Sửa đổi | 1 (sửa `--seed type`) |
| `budgeted_protocol.py` | **Tạo mới** | 4 hàm + `PARSE_REGEX` + `__main__` |
| `learned_pruning.py` | **Tạo mới** | 6 hàm + class `PruningMLP` + `__main__` |
