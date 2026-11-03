# ECG Core v1.3.3 – Multi-Layout + Signal/Artifact QA

- Giữ nguyên `legacy/app_v17_1_original.py`, `ecg_digitizer.py` và API/failover.
- Mở expander **ECG Core v1.3.3 | Multi-Layout + Signal/Artifact QA**.
- Chọn ảnh gốc; engine đề xuất bố cục bằng OCR nếu đủ bằng chứng. Nếu `UNKNOWN`, bác sĩ **chọn một lần** 3×4 hoặc 6×2. Không dùng màn QA v1.3.2 cho ảnh 6×2.
- Với 3×4, xác nhận **có/không** dải Rhythm II. Với 6×2, mặc định **không có** rhythm riêng.
- Chỉnh ranh giới cột/hàng hay lead sai trên ảnh overlay; xuất ZIP validation.
- Panel mask: original | raw dark | lưới màu tự tin | chữ/xung chuẩn nghi ngờ | pixel giao cắt chưa chắc | candidate ink overlay.
- Không được diễn giải các mask là waveform đã số hóa, không đo PR/QRS/QT hoặc ST từ dữ liệu này.
- Tất cả đều REVIEW cho đến khi kiểm tra trên ảnh gốc và có ground truth.
- Các detector lưới xám, chữ và xung chuẩn vẫn là ước lượng và có thể chưa nhận hết.

**Kiểm thử:** `python -m pytest -q` ở thư mục project, sau đó chạy Streamlit `streamlit run app.py` và xem ảnh overlay/validation.
