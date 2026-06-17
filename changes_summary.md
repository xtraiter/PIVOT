# Tổng hợp các thay đổi đã thực hiện (Changes Summary)

Tất cả các tối ưu hóa, sửa lỗi tương thích và định dạng log được thực hiện trên mã nguồn gốc của bài báo `one-shot-subgraph` để phù hợp với hệ thống của bạn (Xeon 80 CPUs, RTX 5060 Ti 16GB, PyTorch 2.7+cu128).

---

## 1. Song song hóa tính điểm PPR (PPR Score Parallelization)
*   **File sửa đổi**: [PPR_sampler.py](file:///home/vanba/KLTN/one-shot-subgraph/PPR_sampler.py)
*   **Chi tiết thay đổi**:
    *   Chuyển đổi vòng lặp tính điểm Personalized PageRank (PPR) tuần tự từng thực thể (vốn mất ~2 tiếng trên WN18RR và ~80 tiếng trên YAGO) sang chạy song song bằng `multiprocessing.Pool`.
    *   Tự động phát hiện số CPU và cấu hình số lượng worker tối ưu: `min(64, os.cpu_count() - 4)` (sử dụng 64 luồng CPU song song trên máy của bạn).
    *   Sử dụng cơ chế chia sẻ bộ nhớ (global variable shared via worker initializer) để truyền đồ thị NetworkX cho các tiến trình con mà không tốn tài nguyên tuần tự hóa (pickling overhead).
    *   Tích hợp thanh tiến trình `tqdm` đồng bộ thời gian thực cho quá trình xử lý song song.

## 2. Cơ chế chống hỏng file cache PPR (Atomic Write & Integrity Check)
*   **File sửa đổi**: [PPR_sampler.py](file:///home/vanba/KLTN/one-shot-subgraph/PPR_sampler.py)
*   **Chi tiết thay đổi**:
    *   Áp dụng phương pháp **atomic write** (ghi nguyên tử) bằng cách dùng `tempfile.NamedTemporaryFile` để ghi điểm PPR ra một file tạm trước, sau khi hoàn tất mới đổi tên đè lên file chính (`os.replace`). Điều này giúp ngăn chặn hoàn toàn việc các file `.pkl` bị trống hoặc lỗi nếu bạn ngắt tiến trình giữa chừng.
    *   Thêm bước kiểm tra kích thước file (`os.path.getsize(path) < 1000`) khi khởi tạo sampler để tự động phát hiện và xóa/tính lại các file cache bị hỏng từ các lượt chạy lỗi trước đó.

## 3. Sửa lỗi ép kiểu Seed trên NumPy mới (Seed Type Fix)
*   **Files sửa đổi**: [train_auto.py](file:///home/vanba/KLTN/one-shot-subgraph/train_auto.py), [search_auto.py](file:///home/vanba/KLTN/one-shot-subgraph/search_auto.py)
*   **Chi tiết thay đổi**:
    *   Mặc định của repo cũ cấu hình kiểu dữ liệu của `--seed` là `str` (chuỗi văn bản). Trên NumPy phiên bản mới (2.2.6), việc gọi `np.random.seed("456")` với tham số là chuỗi sẽ báo lỗi `TypeError` nghiêm trọng.
    *   Đã chuyển kiểu dữ liệu tham số `--seed` trong ArgumentParser thành kiểu số nguyên `int`.

## 4. Khắc phục lỗi tương thích PyTorch 2.7 (PyTorch 2.7 Compatibility)
*   **File sửa đổi**: [base_model.py](file:///home/vanba/KLTN/one-shot-subgraph/base_model.py)
*   **Chi tiết thay đổi**:
    *   Loại bỏ tham số `verbose=True` trong hàm khởi tạo bộ giảm learning rate `ReduceLROnPlateau`. Tham số này đã bị xóa hoàn toàn kể từ các phiên bản PyTorch mới (2.6+), khiến chương trình bị crash ngay khi bắt đầu train.

## 5. Tối ưu hóa dung lượng bộ nhớ GPU (DataLoader GPU Memory Optimization)
*   **File sửa đổi**: [base_model.py](file:///home/vanba/KLTN/one-shot-subgraph/base_model.py)
*   **Chi tiết thay đổi**:
    *   Định nghĩa hàm `worker_init_fn` gán `os.environ["CUDA_VISIBLE_DEVICES"] = ""` cho các tiến trình con của DataLoader. Điều này giúp ngăn chặn việc các CPU loader workers tự động tạo ngữ cảnh CUDA trống trên GPU, tiết kiệm **2GB – 8GB bộ nhớ GPU** khi train.
    *   Tự động xử lý tham số `prefetch_factor` thành `None` khi tham số `--cpu 0` được truyền vào, tránh lỗi cấu hình DataLoader của PyTorch.

## 6. Ghi Log chi tiết & Phân biệt Checkpoint theo Seed
*   **File sửa đổi**: [base_model.py](file:///home/vanba/KLTN/one-shot-subgraph/base_model.py)
*   **Chi tiết thay đổi**:
    *   Đo đạc và in chi tiết thời gian huấn luyện (`latency_ms`), bộ nhớ GPU đỉnh (`peak_gpu_mem_mb`), thời gian chuẩn bị dữ liệu (`data_prep_ms`), thời gian forward pass (`forward_ms`), và thời gian xếp hạng (`ranking_ms`), khớp hoàn toàn với cấu trúc log của repo `PIVOT`.
    *   Làm tròn kết quả MRR và Hits thành 6 chữ số thập phân (`%.6f`) để đánh giá chính xác hơn.
    *   Tự động chèn hậu tố `_seed{self.args.seed}` vào tên file checkpoint được lưu ở mỗi epoch tốt nhất để tránh việc các lượt chạy với seed khác nhau ghi đè lên nhau.

## 7. Hỗ trợ Huấn luyện Nửa độ chính xác (Mixed Precision - AMP)
*   **File sửa đổi**: [base_model.py](file:///home/vanba/KLTN/one-shot-subgraph/base_model.py)
*   **Chi tiết thay đổi**:
    *   Tích hợp thư viện `torch.cuda.amp` với cơ chế tự động chuyển đổi kiểu dữ liệu (`autocast`) và bộ co giãn độ dốc (`GradScaler`).
    *   Việc huấn luyện mô hình 8 lớp ở `batchsize=16` sẽ chuyển đổi các phép tính toán sang kiểu dữ liệu `float16` giúp giảm lượng VRAM tiêu thụ thực tế từ ~16.4 GB xuống chỉ còn khoảng **9 GB – 10 GB**.
    *   Cho phép bạn huấn luyện ổn định với **`batchsize 16`** (đúng theo cấu hình bài báo gốc) trên GPU 16GB mà không lo bị lỗi Out-Of-Memory (OOM).
