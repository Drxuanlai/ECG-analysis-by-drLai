# ECG Core v1.4.1 – Vertical Stroke / QRS Spike Preservation (QA)

- Chỉ bổ sung bằng chứng nét dốc đứng ứng viên trên ROI đã xác nhận. **Không phải QRS/R detector lâm sàng.**
- So sánh đường dựng v1.4 với đường đề xuất v1.4.1. Dấu đỏ đánh dấu nét dốc đứng được tìm thấy; không tự vẽ liên tục qua vùng thiếu bằng chứng.
- Tại khu v1.4, chọn lead rồi bật **So sánh trước/sau bảo toàn nét dốc đứng**; xuất `ecg_core_v1_4_1_spike_validation.zip`.
- Nếu ứng viên dốc đứng bị thiếu ở các đỉnh R, chưa được coi là cải thiện thành công. Nếu phát hiện nhầm đường lưới/xung chuẩn, giữ REVIEW.
- File này dùng nghiên cứu kiểm chứng, không dùng đo PR/QRS/QT, ms/mV hoặc hỗ trợ quyết định điều trị.
