# ECG Core v1.5.7 — QRS Component Localization (r / S / QS) — PATCH ONLY

## Cài đặt

Đưa các file trong ZIP vào **đúng thư mục gốc** của branch `refactor/ecg-core-v1`, trên nền v1.5.6:

- `ui/beat_panel.py`: **thay** file hiện tại.
- `core/qrs_components.py`: **thêm** module mới.
- `tests/test_qrs_components.py`: **thêm** bài kiểm thử.
- `QRS_COMPONENTS_V1_5_7_GUIDE_VI.md`: **thêm** tài liệu này.

Không thay `app.py`, `legacy/`, `requirements.txt` hoặc Gemini/failover. Module không đòi hỏi dependency mới (dùng numpy/Pillow đã có).

## Kiểm thử

Trong Streamlit, vào `Beat-by-Beat Candidate Review`, chọn V1, kiểm tra ROI rồi chạy các bước QA như v1.5.6. Một khối mới **v1.5.7 · Thành phần QRS: r nhỏ / S sâu** sẽ hiển thị:

- Chấm tròn xanh lam: đáy S ứng viên theo ảnh.
- Ô vuông tím: đỉnh r nhỏ ứng viên (nếu tìm thấy).
- Dấu cộng xanh lá: các điểm bác sĩ đã đánh dấu.
- Đường ngang cam: baseline ước lượng.

Xuất `ecg_core_v1_5_7_qrs_components_validation.zip` để kiểm tra thêm ba file:
`qrs_components_v157.json`, `qrs_components_vs_reference_v157.json`, `qrs_components_overlay_v157.png`.

## Nguyên tắc an toàn

- Phát hiện thành phần từ **ảnh ROI và mực**, không dùng các điểm bác sĩ đánh dấu để tạo dự đoán.
- Các điểm bác sĩ chỉ dùng để **đối chiếu sau dự đoán**.
- Nếu chưa tìm được r nhỏ, không kết luận QS; ghi `r_NOT_RESOLVED_QS_NOT_PROVEN`.
- Với QRS dương chủ yếu, QS, rSR′, QRS rộng hoặc nhiễu, thuật toán có thể không đưa ra đề xuất hữu ích. Không dùng báo cáo này làm đo lường lâm sàng.
- Mọi thành phần đề xuất đều `REVIEW`. Không tính RR, QRS duration, QT, ST hoặc chẩn đoán từ các mốc này.
- Nếu ROI/crop bị đổi, nhãn cũ không được sử dụng trên ROI khác.

## Phạm vi nghiệm thu

Đối chiếu riêng V1 trên ECG hiện tại (3 đáy S tham chiếu), sau đó thử DII/V4, và ca có QRS phức tạp; **không dùng chính ca tối ưu để tuyên bố accuracy chung**. Bảng so khớp 2D đánh giá vị trí, không chứng minh nhận diện đúng thành phần sóng.
