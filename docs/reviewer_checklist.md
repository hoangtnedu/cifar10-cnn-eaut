# Checklist chỉnh sửa theo phản biện

## Bắt buộc nên hoàn thành

- [ ] Chạy đồng đều 5 seed cho toàn bộ mô hình.
- [ ] Cập nhật bảng Accuracy và F1-macro dạng `mean ± std`.
- [ ] Lưu kết quả từng seed để có căn cứ thống kê.
- [ ] Bổ sung cấu hình môi trường thực nghiệm: Colab, Python, PyTorch, GPU, CPU, RAM.
- [ ] Ghi rõ cách đo latency: CPU, batch size = 1, warm-up, số lần lặp đo.
- [ ] Giải thích vì sao FLOPs thấp chưa chắc latency thấp.
- [ ] Hạ giọng kết luận về SimpleCNN: chỉ kết luận trong phạm vi CIFAR-10 và thiết lập huấn luyện hiện tại.
- [ ] Bổ sung hạn chế nghiên cứu.
- [ ] Bổ sung hướng nghiên cứu tiếp theo: tối ưu siêu tham số riêng, kiểm định thống kê, nhiều bộ dữ liệu hơn.

## Nên làm thêm

- [ ] Bổ sung khoảng tin cậy 95% hoặc kiểm định thống kê.
- [ ] Vẽ lại biểu đồ với font lớn hơn.
- [ ] Giải thích ký hiệu `~` và `±`.
- [ ] Rà soát thuật ngữ Anh - Việt.
- [ ] Đưa bảng xuất hiện trước đoạn phân tích liên quan.

## Câu diễn giải an toàn về kết quả

> Trong phạm vi thiết lập thực nghiệm của nghiên cứu này trên CIFAR-10, SimpleCNN đạt điểm cân bằng tốt nhất giữa độ chính xác và chi phí tính toán. Kết quả này không nhằm khẳng định SimpleCNN vượt trội tuyệt đối so với các kiến trúc hiện đại trong mọi bối cảnh, mà cho thấy mô hình nông, gọn nhẹ vẫn có thể là lựa chọn phù hợp khi dữ liệu ảnh có kích thước nhỏ và tài nguyên triển khai bị giới hạn.
