# PIVOT Customization Rules

Sử dụng quy tắc này để tự động đồng bộ hóa tài liệu thay đổi mã nguồn qua các phiên làm việc (chat sessions) khác nhau:

- Mọi tác nhân AI khi bắt đầu phiên làm việc hoặc thực hiện thay đổi mã nguồn (sửa file cũ hoặc thêm file mới) **BẮT BUỘC** phải đọc và cập nhật tệp tin [changes_summary.md](file:///home/vanba/KLTN/one-shot-subgraph/reports/changes_summary.md).
- Nếu sửa đổi tệp tin cũ: cập nhật phần mô tả thay đổi kèm theo khối mã `git diff` trực tiếp dưới tệp đó.
- Nếu thêm tệp tin mới: cập nhật bảng tóm tắt và ghi lại toàn bộ nội dung mã nguồn của tệp tin mới vào mục lục **"Nội dung đầy đủ các tệp tin mới thêm"** ở cuối tệp [changes_summary.md](file:///home/vanba/KLTN/one-shot-subgraph/reports/changes_summary.md).

---

## Quy tắc thiết kế mô hình và Đánh giá (PIVOT Core Guidelines)

1. **Chính sách Post-hoc Reranking**:
   - Tránh huấn luyện Joint GNN trực tiếp trên subgraphs bị lọc bởi Hybrid Sampler để ngăn chặn hiện tượng **đứt gãy liên kết (Connectivity Starvation)**.
   - GNN phải được huấn luyện trên subgraphs sinh bởi PPR baseline chuẩn. Việc kết hợp tri thức ngữ nghĩa của MLP Pruning được thực hiện ở bước xếp hạng cuối cùng (Post-hoc Reranking) thông qua tổ hợp tuyến tính điểm số: `final_score = (1 - alpha) * GNN_score + alpha * MLP_score` (điều khiển bởi tham số `--rerank_alpha`).

2. **Chính sách Bộ điều phối Pareto (Pareto Controller)**:
   - Lựa chọn cấu hình phần cứng/mô hình `(alpha, beta, layer, budget)` dựa trên 5 điểm tối ưu không bị trị phối (Pareto Frontier) của từng dataset.
   - Luôn sử dụng bộ điều phối `BudgetController` trong [pareto_optimizer.py](file:///home/vanba/KLTN/one-shot-subgraph/pareto_optimizer.py) để trả lời các truy vấn tối ưu hóa ràng buộc:
     - Tối đa hóa MRR dưới giới hạn Latency ≤ T.
     - Tối thiểu hóa Latency dưới yêu cầu MRR ≥ X.

3. **Giao thức Đánh giá Budgeted Protocol**:
   - Luôn chuẩn hóa việc quét budget theo tỷ lệ thực thể giữ lại (1%, 5%, 10%, 20%).
   - Báo cáo đầy đủ và đồng thời cả hai khía cạnh: Độ chính xác (MRR, Hits) và Hiệu năng thực tế (Latency/query, Peak GPU Memory reserved, Throughput).
