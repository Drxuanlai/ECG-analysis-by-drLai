# ECG Core v1.4.2 — QRS Segment Reconstruction QA

## Cập nhật GitHub

Đây là PATCH ONLY dành cho branch đã cài v1.4.1. Giữ đúng các thư mục, chỉ thêm hoặc cập nhật:

- `core/qrs_segment_reconstruction.py` — thêm mới
- `ui/multi_layout_panel.py` — thay thế file cũ
- `tests/test_qrs_segment_reconstruction.py` — thêm mới
- `QRS_SEGMENT_V1_4_2_GUIDE_VI.md` — thêm mới

Không cần upload `app.py`, `legacy`, benchmark, ảnh mẫu hoặc các module không thay đổi.

## Kiểm thử

Trong giao diện v1.4, chọn bố cục và ROI đã kiểm tra, bật dựng tín hiệu 1-D, sau đó bật **So sánh đường gốc và QRS segment candidate** tại khu vực v1.4.2.

Đọc bốn ô: original ROI, mask ứng viên, waveform nền v1.4, đề xuất segment v1.4.2. Dấu xanh biểu thị đoạn ứng viên đã lần được; dấu đỏ là ứng viên bị từ chối, không có nghĩa đã chẩn đoán QRS.

Export `ecg_core_v1_4_2_qrs_segments_validation.zip` chứa ảnh và tọa độ/nhận xét của từng đoạn.

## Giới hạn cần lưu ý

- Một số đỉnh R vẫn có thể bị bỏ sót; nhiễu lưới xám có thể tạo ứng viên giả.
- Độ phủ có mực không phải độ chính xác; nhãn `SEGMENT_CANDIDATE_REVIEW` không được hiểu thành QRS đã xác minh.
- Đường cũ ngoài vùng đề xuất có thể vẫn sai.
- Không đo PR/QRS/QT, biên độ, không chẩn đoán từ kết quả này.
