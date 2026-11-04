# ECG Core v1.5.1 — Beat Candidate Validation & Artifact Triage

**Patch only.** Áp dụng lên repository đã triển khai ECG Core v1.5.

Tệp thay đổi:
- `core/beat_validation.py`: module mới để gom ứng viên gần nhau, đánh dấu vùng nhiễu, giữ các trường hợp không giải quyết được (UNRESOLVED).
- `ui/beat_panel.py`: cập nhật giao diện QA, nhãn bác sĩ và ZIP validation.
- `tests/test_beat_validation.py`: unit tests mới.

Không thay đổi `app.py`, Gemini, failover, layout hay `legacy/`.

## Thao tác

1. Giải nén patch rồi tải các tệp vào đúng vị trí trên branch `refactor/ecg-core-v1`.
2. Mở Streamlit > ECG Core v1.5 | Beat-by-Beat Candidate Review.
3. Tải ảnh, chọn bố cục, kiểm tra ROI, chạy engine.
4. Xem các sự kiện E001...; `CANDIDATE_REVIEW` không đồng nghĩa QRS đã xác minh; `UNRESOLVED` không đồng nghĩa nhiễu.
5. Bác sĩ có thể mở phần gán nhãn để chọn QRS / ARTIFACT / UNCERTAIN. Nhập x-pixel các QRS bị bỏ sót sau khi đối chiếu ảnh gốc.
6. Tải `ecg_core_v1_5_1_beat_validation.zip`: có kết quả v1.5 cũ, sự kiện v1.5.1 và ảnh overlay.

## Giới hạn

- Gom ứng viên dùng khoảng cách hình học tương đối trong ROI, **không ép RR đều**; trường hợp nhịp nhanh thực có thể bị gom chung. Tất cả đều cần kiểm tra.
- Không gọi kết quả là `VERIFIED_BEAT` khi chưa gán nhãn bác sĩ.
- Không được tính nhịp tim, PR, QRS duration, QTc hoặc chẩn đoán từ các ứng viên này.
- Bộ lọc vùng dày không xóa nhịp, chỉ chuyển trạng thái `UNRESOLVED`.
- Không có nhãn tham chiếu đầy đủ thì không công bố sensitivity/precision.
- Để tính các chỉ số đó về sau cần ghi cả vị trí QRS bị bỏ sót và kiểm tra các nhãn bác sĩ.
