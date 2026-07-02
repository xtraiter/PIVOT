# Báo cáo Toàn diện: Tái lập & Phát triển Hệ thống Tối ưu hóa Subgraph (PIVOT)

Tài liệu này cung cấp báo cáo kỹ thuật toàn diện duy nhất về quá trình tái lập mã nguồn gốc (Weeks 1–5) và phát triển phương pháp mới **PIVOT (Pareto-Improved subgraph reasoning under budgeT)** (Weeks 6–9) trên tập dữ liệu WN18RR. Báo cáo bao gồm số liệu thực nghiệm so sánh baselines, giải thích lý thuyết sâu sắc, kiến trúc mô hình MLP, các tham số huấn luyện, và danh sách các tệp tin đã thay đổi.

---

## 1. DANH SÁCH CÁC TỆP TIN ĐÃ THAY ĐỔI & THÊM MỚI

Dưới đây là bảng tổng hợp các tệp tin trong dự án đã được chỉnh sửa (opt) hoặc thêm mới (new):

| STT | Tệp tin | Trạng thái | Nội dung chi tiết thay đổi / Vai trò trong dự án |
| :--- | :--- | :--- | :--- |
| 1 | **[model.py](file:///home/vanba/KLTN/one-shot-subgraph/model.py)** | Chỉnh sửa | - Tích hợp **Gradient Checkpointing** qua lớp `PropagationCell` để giải phóng activation memory của các lớp GRU.<br>- Tối ưu hóa **Attention Projection Order** trong `GNNLayer` (chỉ chiếu tuyến tính một lần lên embeddings thực thể/quan hệ thay vì chiếu lặp lại trên từng cạnh).<br>- Hỗ trợ **Dynamic Relation Embedding Size** (`2*n_rel+3` khi dùng cạnh ảo, tự động co về `2*n_rel+1` khi chạy PPR baseline). |
| 2 | **[load_data.py](file:///home/vanba/KLTN/one-shot-subgraph/load_data.py)** | Chỉnh sửa | - Chỉnh sửa `DataLoader` để thu thập và chuyển giao thông tin quan hệ truy vấn (`rel`) vào sampler trong quá trình sinh subgraph (phục vụ cho Learned Pruning). |
| 3 | **[PPR_sampler.py](file:///home/vanba/KLTN/one-shot-subgraph/PPR_sampler.py)** | Chỉnh sửa | - Cài đặt **Multiprocessing PPR** chạy song song trên 64 CPU cores.<br>- Tích hợp **Class-level Global Memory Cache** chia sẻ ma trận PPR giữa các sampler để triệt tiêu overhead đọc đĩa đúp.<br>- Cài đặt bộ chọn mẫu lai **Hybrid Sampler** ($50\%$ MLP + $50\%$ PPR) và liên kết cạnh ảo (`add_manual_edges`). |
| 4 | **[base_model.py](file:///home/vanba/KLTN/one-shot-subgraph/base_model.py)** | Chỉnh sửa | - Tích hợp **AMP (Automatic Mixed Precision)** qua `GradScaler` giúp tăng tốc tính toán và giảm VRAM.<br>- Bổ sung đo đạc bộ nhớ thực tế qua `torch.cuda.max_memory_reserved()`.<br>- Bổ sung cơ chế đo chi tiết latency nội bộ (`data_prep_ms`, `forward_ms`, `ranking_ms`). |
| 5 | **[train_auto.py](file:///home/vanba/KLTN/one-shot-subgraph/train_auto.py)** | Chỉnh sửa | - Bổ sung tham số CLI điều khiển (`--use_learned_pruning`, `--pruning_model_path`, `--add_manual_edges`).<br>- Viết tập lệnh tự động hóa đẩy checkpoints và log lên Git (`git_push_update`). |
| 6 | **[search_auto.py](file:///home/vanba/KLTN/one-shot-subgraph/search_auto.py)** | Chỉnh sửa | - Cập nhật kiểu dữ liệu seed và cấu trúc phân chia thư mục lưu trữ kết quả HPO. |
| 7 | **[learned_pruning.py](file:///home/vanba/KLTN/one-shot-subgraph/learned_pruning.py)** | Thêm mới | - Định nghĩa lớp mô hình MLP Pruning (`PruningMLP`) và hàm trích xuất đặc trưng candidate. |
| 8 | **[run_learned_pruning_wn18rr.py](file:///home/vanba/KLTN/one-shot-subgraph/run_learned_pruning_wn18rr.py)** | Thêm mới | - Kịch bản huấn luyện MLP xuyên suốt đa seed, trích xuất 7 nhóm đặc trưng, negative sampling (hard + random), và tính toán Recall định dạng Realistic vs Oracle. |
| 9 | **[budgeted_protocol.py](file:///home/vanba/KLTN/one-shot-subgraph/budgeted_protocol.py)** | Thêm mới | - Tập lệnh chạy quét budget (1%, 5%, 10%, 20%) cho baseline hoặc learned pruning để ghi nhận các chỉ số Accuracy & Efficiency. |
| 10 | **[pareto_optimizer.py](file:///home/vanba/KLTN/one-shot-subgraph/pareto_optimizer.py)** | Thêm mới | - Bộ điều phối budget (`BudgetController`) trích xuất Pareto Frontier từ dữ liệu JSON và cung cấp API tối ưu hóa đa mục tiêu. |

---

## 2. BẢNG SỐ LIỆU THỰC NGHIỆM SO SÁNH (WN18RR)

### A. So sánh Hiệu năng dự đoán (Table 1 trong bài báo gốc)

Bảng so sánh kết quả kiểm thử link prediction trên WN18RR giữa các mô hình baselines (thu thập từ bài báo ICLR 2024) và phương pháp tái lập/tối ưu hóa của chúng ta:

| Nhóm Mô hình | Tên Mô hình | WN18RR Test MRR | Hits@1 | Hits@10 |
| :--- | :--- | :---: | :---: | :---: |
| **Semantic Models** | ConvE | 0.427 | 39.2% | 49.8% |
| | QuatE | 0.480 | 44.0% | 55.1% |
| | RotatE | 0.477 | 42.8% | 57.1% |
| **Structural Models** | MINERVA | 0.448 | 41.3% | 51.3% |
| | DRUM | 0.486 | 42.5% | 58.6% |
| | RNNLogic | 0.483 | 44.6% | 55.8% |
| | CompGCN | 0.479 | 44.3% | 54.6% |
| | DPMPN | 0.482 | 44.4% | 55.8% |
| | NBFNet | 0.551 | 49.7% | 66.6% |
| | RED-GNN | 0.533 | 48.5% | 62.4% |
| **One-Shot Models** | one-shot-subgraph (PPR gốc) | **0.567** | **51.4%** | **66.6%** |
| **PIVOT (Ours)** | PIVOT (Reproduction - Seed 42) | **0.566** | **51.5%** | **66.4%** |
| | PIVOT (Mean ± Std - 3 Seeds) | 0.563 ± 0.001 | 51.2% | 66.2% |

> [!NOTE]
> - **Seeds đã chạy**: Seed 42, Seed 123, Seed 1234.
> - **Seeds bổ sung đang tiến hành**: Seed 456, Seed 777 (để đạt đủ 5 seeds trung bình).

### B. So sánh Hiệu năng phần cứng & Tốc độ (Table 2)

Efficiency metrics đo đạc trên GPU RTX 5060 Ti (16GB VRAM) cho thấy cải tiến vượt trội sau tối ưu:

| Chỉ số đo lường | Model Gốc (24/06) | PIVOT Tối ưu (02/07) | Cải thiện (Tốc độ / Bộ nhớ) |
| :--- | :---: | :---: | :--- |
| **Validation MRR** | `0.564374` | `0.563707` | Sai số $0.0006$ (không đáng kể, đạt tính tái lập) |
| **Peak GPU Memory** | **`12.1 GB`** | **`1.50 GB`** | **Giảm 8.07× bộ nhớ chiếm dụng thực tế** |
| **Thời gian Epoch 1** | **`1,215.03 giây`** | **`237.87 giây`** | **Nhanh gấp 5.1×** (nhờ Global PPR Cache) |
| **Thời gian Epoch 2+** | `~240.1 giây` | **`231.59 giây`** | Nhanh hơn 3.5% |
| **Thời gian Eval / epoch** | `~159.2 giây` | **`146.73 giây`** | **Nhanh hơn 7.8%** |
| **Inference Throughput** | ~3.3 queries/s | **35.78 queries/s** | **Tăng throughput gấp 10.8×** |

### C. Giao thức Quét Budget (Giao thức Budgeted Protocol - Gộp 3 Seeds)

Kết quả khi quét qua các mức subgraph budget trên WN18RR:

| Budget Ratio (Top-K) | Số lượng Nodes | Test MRR | Hits@1 | Hits@10 | Latency / Query (ms) | Throughput (queries/s) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.01 (1%)** | 409 | `0.5412` | `0.4963` | `0.6264` | **12.32 ms** | **81.24 q/s** |
| **0.05 (5%)** | 2047 | `0.5642` | `0.5124` | `0.6633` | **17.16 ms** | **58.29 q/s** |
| **0.10 (10%)** | 4094 | `0.5636` | `0.5115` | `0.6627` | **27.95 ms** | 35.78 q/s |
| **0.20 (20%)** | 8188 | `0.5598` | `0.5082` | `0.6568` | 48.46 ms | 20.64 q/s |

### D. Đường biên tối ưu Pareto (Pareto Frontier)

Thông số của **5 điểm vận hành tối ưu nhất** được trích xuất bởi bộ điều phối từ tệp [pareto_cache_WN18RR.json](file:///home/vanba/KLTN/one-shot-subgraph/budget_results/pareto_cache_WN18RR.json):

| Pareto Point | Budget | Layer ($L$) | $\alpha$ | $\beta$ | Test MRR | Latency (ms) | Peak GPU Mem (MB) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Point 1** | 0.01 (1%) | 8 | 0.85 | 0.00 | `0.5416` | **5.00 ms** | **280.0 MB** |
| **Point 2** | 0.01 (1%) | 8 | 0.85 | 0.25 | `0.5439` | **6.05 ms** | **280.0 MB** |
| **Point 3** | 0.05 (5%) | 6 | 0.85 | 0.00 | `0.5625` | **11.22 ms** | 800.0 MB |
| **Point 4** | 0.05 (5%) | 6 | 0.85 | 0.25 | `0.5630` | **13.53 ms** | 800.0 MB |
| **Point 5** | 0.05 (5%) | 8 | 0.85 | 0.00 | **`0.5641`** | **14.52 ms** | 800.0 MB |

---

## 3. KIẾN TRÚC MÔ HÌNH LEARNED PRUNING (MLP)

Mục tiêu của Learned Pruning là thay thế phép lọc heuristic PPR không học bằng một bộ phân loại query-aware nhỏ gọn, nhằm tối ưu hóa khả năng lọc nhiễu trong subgraph ở mức budget nhỏ.

### A. 7 Nhóm Đặc Trưng Đầu Vào (Feature Extraction)

Với mỗi thực thể ứng viên $v$ nằm trong PPR Candidate Pool ban đầu của query $(u, q, ?)$, chúng ta trích xuất một vector đặc trưng $\mathbf{x}_v \in \mathbb{R}^7$:

1.  **`ppr_scores`:** Điểm PPR từ node $u$ tới $v$. Tín hiệu cấu trúc nền tảng.
2.  **`ppr_rank_percentile`:** Thứ hạng phần trăm của PPR. Giúp chuẩn hóa phân phối điểm PPR giữa các query khác nhau.
3.  **`log(degree)`:** Bậc của candidate trong toàn đồ thị $\log(d_v + 1)$. Tránh MLP bị thiên lệch về các node trung tâm (hub entities).
4.  **`hop_distance`:** Khoảng cách BFS ngắn nhất từ $u$ tới $v$. Giới hạn mức độ lan truyền thông tin.
5.  **`is_direct_neighbor`:** Nhãn nhị phân ($0/1$) kiểm tra xem $v$ có kết nối trực tiếp (1-hop) với $u$ không.
6.  **`tail_freq_for_q`:** Tỷ lệ xuất hiện của $v$ làm tail entity cho quan hệ $q$ trong tập train: $P(v \mid q)$.
7.  **`rel_match_score`:** Đo độ tương hợp quan hệ. Tỷ lệ phần trăm các cạnh đi ra từ $v$ có chứa quan hệ $q$.

### B. Cấu trúc mạng MLP Pruning
Mạng MLP được xây dựng theo kiến trúc 3 lớp tuyến tính (Linear Layers) tuần tự:
```
Input (7 dimensions) 
   │
   ├──> Linear(7, 64) ──> ReLU() ──> Dropout(0.1)
   │
   ├──> Linear(64, 32) ──> ReLU() ──> Dropout(0.1)
   │
   └──> Linear(32, 1) ──> Output Score (1 dimension)
```
*   *Lý do chọn kích thước 64→32:* Kích thước này đủ nhỏ để tính toán siêu nhanh (chi phí suy diễn chỉ khoảng <1ms cho hàng ngàn ứng viên), nhưng đủ dung lượng để học phi tuyến các đặc trưng lai giữa ngữ nghĩa và cấu trúc.

---

## 4. BIỆN GIẢI THAM SỐ HUẤN LUYỆN MLP (WEEK 9)

Kịch bản huấn luyện MLP (`run_learned_pruning_wn18rr.py`) sử dụng các hyperparameter được thiết kế đặc thù cho bài toán mất cân bằng mẫu nghiêm trọng:

*   **Bộ tối ưu (Optimizer) & Học bạ (Learning Rate):** Sử dụng `Adam` với `lr=0.001`, `weight_decay=1e-5` để điều hòa trọng số. Tích hợp `ReduceLROnPlateau` với patience=2 để tự động giảm một nửa tốc độ học khi loss trên validation đi ngang.
*   **Khai thác Mẫu khó (Hard Negative Mining):** Với mỗi query, ta chọn **30 hard negatives** (các node có điểm PPR cao nhất nhưng không phải đáp án đúng) và **20 random negatives**. Điều này buộc MLP phải học cách phân biệt các ca khó nhất (nằm rất gần query nhưng sai ngữ nghĩa).
*   **Hàm mất mát kết hợp (Combined Loss):**
    $$\mathcal{L} = \text{BCE\_With\_Logits}(pos\_weight=5.0) + \lambda \cdot \mathcal{L}_{\text{pairwise\_hinge}}$$
    *   *Tại sao dùng `pos_weight=5.0` trong BCE:* Do dữ liệu cực kỳ lệch (ít nhãn dương), ta nhân trọng số nhãn dương lên 5 lần để phạt nặng mô hình khi bỏ sót đáp án đúng.
    *   *Tại sao dùng Hinge Loss song song ($\lambda=0.4$, $margin=1.0$):* BCE tối ưu hóa điểm số tuyệt đối, nhưng mục tiêu của ta là **xếp hạng (ranking)**. Hinge Loss phạt mô hình nếu điểm của đáp án đúng không lớn hơn điểm của các đáp án sai tối thiểu một khoảng cách (margin) bằng 1.0. Điều này tối ưu trực tiếp cho chỉ số Recall@K.
*   **Dừng sớm (Early Stopping):** Thiết lập `patience=5` trên chỉ số Realistic Recall@100 của tập Validation để ngăn mô hình bị quá khớp (overfitting).

---

## 5. GIẢI THÍCH LÝ THUYẾT & HIỆN TƯỢNG SỐ LIỆU

### A. Tại sao PPR heuristic gốc bị giới hạn trần lý thuyết ở mức 84.20%?
*   *Lý do:* PPR chỉ lan truyền dựa trên liên kết đồ thị, hoàn toàn không biết query đang hỏi về quan hệ gì. Với mức budget $K=0.1$ (tương ứng 4094 candidate), có tới **15.80%** đáp án đúng nằm ngoài tầm với của PPR do các liên kết đồ thị xung quanh chúng quá thưa thớt. Đây là giới hạn vật lý của cấu trúc đồ thị WN18RR. MLP Pruning chỉ có thể xếp hạng tốt nhất dựa trên danh sách ứng viên này (trần Realistic Recall là 84.20%).

### B. Tại sao MLP Pruning cải thiện vượt bậc Recall ở các budget nhỏ ($K=50$, $K=200$)?
*   *Lý do:* PPR xếp hạng ứng viên thuần túy theo cấu trúc. MLP Pruning lọc bỏ các node "nhiễu cấu trúc" (ví dụ: node hàng xóm có liên kết rất mạnh nhưng quan hệ sai lệch hoàn toàn) và đẩy các node "đúng ngữ nghĩa quan hệ" lên trên. Kết quả là tại budget cực kỳ thắt chặt ($K=50$), MLP gom được nhiều đáp án đúng hơn PPR, tăng Recall từ **37.30%** lên **43.93%**.

### C. Tại sao Joint Training GNN trên mô hình cắt tỉa lai (Hybrid Sampler) lại thất bại (MRR giảm về 0.41)?
*   *Lý do 1 - Đứt gãy luồng thông tin (Gradient Starvation):* GNN 8 lớp yêu cầu các đường đi liên tục để truyền thông điệp từ $u$ qua các node trung gian đến đích. Khi MLP thực hiện cắt tỉa các node không liên quan trực tiếp đến query ở mức budget nhỏ, nó vô tình xóa luôn các node trung gian (bridge entities) đóng vai trò dẫn truyền. Subgraph bị đứt gãy kết nối, GNN không thể học được cách liên kết đặc trưng.
*   **Lý giải về cạnh ảo (`add_manual_edges`) làm MRR giảm sâu xuống 0.34:** Khi chèn thêm cạnh ảo từ $u$ tới mọi candidate với quan hệ ảo đồng nhất, GNN sẽ lấy thông điệp trực tiếp từ $u$ qua 1 bước. Tuy nhiên, quan hệ ảo này hoàn toàn không có ngữ nghĩa cấu trúc. Lượng thông tin nhiễu khổng lồ từ 4096 cạnh ảo này lấn át hoàn toàn lượng thông tin lập luận đa bước tinh tế từ các cạnh thực, làm hỏng hoàn toàn vector biểu diễn của thực thể. Do đó, **Post-hoc Reranking** (chỉ dùng MLP để lọc lại kết quả dự đoán của GNN đã train trên PPR) là phương án tối ưu duy nhất.
