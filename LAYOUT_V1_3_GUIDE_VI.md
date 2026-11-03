# ECG Core v1.3 – Adaptive Layout & Lead ROI QA

## Cách chạy

- Upload nội dung ZIP lên branch `refactor/ecg-core-v1` (không thay `main`).
- Deploy Streamlit với entrypoint `app.py`.
- Mở mục **ECG Core v1.3 | Adaptive Layout & Lead ROI QA**.
- Upload ảnh PNG/JPG/HEIC; bật OCR khi server có `tesseract`.
- Kiểm tra overlay 12 khung ROI. Nếu lỗi ranh giới, chỉnh ba thanh **Đầu cột** hoặc bốn thanh **Y0–Y3**; chỉ chỉnh ROI riêng khi thật cần thiết.
- Tải ZIP validation gồm `layout_overlay.png` và `layout_v13.json`. Có thể tùy chọn xuất 12 ảnh segmentation để so sánh.

## Phạm vi và hạn chế

- OCR chỉ là bằng chứng; OCR kém sẽ fallback theo hình học và báo REVIEW.
- Ứng dụng không nhận diện phối cảnh phi tuyến/giấy gấp thành hình phẳng trong bản này, chưa có công cụ kéo thả trực tiếp trên ảnh.
- Crop theo ô có thể chứa xung hiệu chuẩn, nhãn và nét của lead bên cạnh. Không dùng để đo PR/QRS/QT/ST hoặc xác nhận bệnh lý.
- Mỗi lần đổi ảnh, kiểm tra vị trí thanh điều chỉnh trước khi xuất (Streamlit giữ state theo key).
- Các module legacy Gemini/failover không thay đổi. Giao diện QA độc lập; chưa gắn ROI mới vào pipeline AI để tránh làm ảnh hưởng bản chạy ổn.
