# ECG Core v1.1 — Adaptive Segmentation QA

## Cài đặt

1. Upload nội dung ZIP vào branch `refactor/ecg-core-v1`, không thay `main`.
2. Đặt `app.py` ở thư mục gốc; giữ `legacy/app_v17_1_original.py` và `ecg_digitizer.py` tại gốc.
3. Deploy Streamlit bằng `app.py` và giữ Secrets ở Cloud.
4. Mở expander **ECG Core v1.1 | Adaptive Segmentation QA** ở cuối giao diện.
5. Tải ảnh ECG hoặc crop chuyển đạo (cho phép tối đa 12 ảnh) rồi tải ZIP validation.

**Lưu ý:** Engine hiện đánh giá hình học mask, không nhận QRS, chưa có số đo hoặc chẩn đoán. `CANDIDATE` chỉ nói mask không rỗng và độ phủ ban đầu hợp lý, KHÔNG phải PASS lâm sàng. Bước tiếp theo cần gắn ROI theo lead và ground truth để định lượng recall nét ECG.

Bốn panel: ảnh gốc, ảnh cân bằng ánh sáng, mask, overlay. Các mask tương ứng 4 phương pháp để xem trực tiếp nơi thuật toán làm mất nét. Không gửi ảnh này ra API.

Nếu ảnh mới tối, `relative`, `adaptive`, `blackhat`, `otsu` cho các lựa chọn khác nhau. Việc tự chọn là ưu tiên sơ bộ để tránh mask bằng 0, không được xem là xác minh ECG.

Chạy test: `python -m pytest tests/test_adaptive_segmentation.py -q` (nếu có pytest). Phải đối chiếu bằng mắt trên cả ba ca.
