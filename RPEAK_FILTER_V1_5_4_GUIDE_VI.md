# ECG Core v1.5.4 — R-Peak Candidate Filtering & Morphology Validation (QA)

Đây là **bản vá chỉ gồm 4 file**, áp dụng **sau v1.5.3**. Không ghi đè `app.py`, `legacy/`, `requirements.txt`, thư mục benchmark hoặc dữ liệu tham chiếu.

## Cập nhật GitHub

Trên branch `refactor/ecg-core-v1`, giữ nguyên cấu trúc khi upload:

- **Thay:** `ui/beat_panel.py`
- **Thêm:** `core/rpeak_filter.py`
- **Thêm:** `tests/test_rpeak_filter.py`
- **Thêm:** `RPEAK_FILTER_V1_5_4_GUIDE_VI.md`

## Cách dùng

1. Mở QA v1.5 trong Streamlit, chọn ảnh/lead và ROI đã kiểm tra, bật chạy phát hiện ứng viên.
2. Tại khu **v1.5.3**, kéo thanh **Tỷ lệ gợi ý đưa vào nhóm xem trước** (mặc định 35%).
3. Mở bảng **v1.5.4 – So sánh gợi ý tím gốc và gợi ý ưu tiên**. Dấu **xanh lam** chỉ là nhóm xem trước; dấu **tím** vẫn là **UNRESOLVED**. Dấu xanh lá là tham chiếu của bác sĩ.
4. Tiếp tục đánh dấu R bằng các cửa sổ như v1.5.3; xác nhận đã rà soát từng cửa sổ.
5. Xuất `ecg_core_v1_5_4_rpeak_filter_validation.zip`.

## Ý nghĩa thống kê

- Thống kê `spatially_matched_count` chỉ so khớp tọa độ x trong dung sai đã chọn. Không kiểm chứng hình thái QRS.
- Khi chưa rà soát hoàn toàn tất cả cửa sổ và xác nhận bộ tham chiếu, `metrics = null`.
- Số lượng 14 gợi ý ưu tiên trên 40 gợi ý gốc từ ca mẫu không phản ánh số nhịp đã phát hiện và **không phải độ nhạy/precision trên tập độc lập**.
- Thuật toán **không sử dụng RR đều**, **không sử dụng điểm bác sĩ** để xếp ưu tiên. Không xóa hint gốc, kể cả hint bị hạ ưu tiên.
- Gợi ý nằm ngoài nhóm ưu tiên có thể chứa ngoại tâm thu, QRS âm hoặc QRS rộng. Đừng chỉ đánh dấu R theo nhóm ưu tiên.

## Kiểm thử

`python -m unittest discover -s tests -p 'test_rpeak_filter.py'`

Bản này **chỉ phục vụ nghiên cứu, không tính nhịp tim, khoảng thời gian hoặc đưa ra chẩn đoán lâm sàng**.
