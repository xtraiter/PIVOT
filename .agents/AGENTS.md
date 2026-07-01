# PIVOT Customization Rules

Sử dụng quy tắc này để tự động đồng bộ hóa tài liệu thay đổi mã nguồn qua các phiên làm việc (chat sessions) khác nhau:

- Mọi tác nhân AI khi bắt đầu phiên làm việc hoặc thực hiện thay đổi mã nguồn (sửa file cũ hoặc thêm file mới) **BẮT BUỘC** phải đọc và cập nhật tệp tin [changes_summary.md](file:///home/vanba/KLTN/one-shot-subgraph/changes_summary.md).
- Nếu sửa đổi tệp tin cũ: cập nhật phần mô tả thay đổi kèm theo khối mã `git diff` trực tiếp dưới tệp đó.
- Nếu thêm tệp tin mới: cập nhật bảng tóm tắt và ghi lại toàn bộ nội dung mã nguồn của tệp tin mới vào mục lục **"Nội dung đầy đủ các tệp tin mới thêm"** ở cuối tệp [changes_summary.md](file:///home/vanba/KLTN/one-shot-subgraph/changes_summary.md).
