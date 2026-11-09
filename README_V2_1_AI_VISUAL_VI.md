# ECG Core v2.1 — AI Visual Reader + Adaptive Cropping

## Mục tiêu
Đọc **một ảnh ECG gốc** bằng Gemini theo checklist 12 chuyển đạo; tự đề xuất một số ROI để đọc lại, có xác nhận bố cục của bác sĩ. Hai lượt AI không được coi là bằng chứng độc lập về độ chính xác lâm sàng. Chưa tự khẳng định phép đo RR, QRS, QTc, ST.

## Upload vào GitHub (PATCH ONLY)
Năm file mới, **không ghi đè file cũ**:
- `visual_reader_v21_app.py` (entrypoint Streamlit độc lập)
- `core/visual_reader_v21.py`
- `core/gemini_visual_v21.py`
- `tests/test_visual_reader_v21.py`
- `README_V2_1_AI_VISUAL_VI.md`

Không cần thêm dependency nếu repo có `streamlit`, `Pillow`, `google-genai`. Tùy chọn `pillow-heif` để đọc HEIC.

**Streamlit Cloud:** tạo app mới từ cùng repo branch `refactor/ecg-core-v1`, main file path = `visual_reader_v21_app.py`. Không thay `app.py` cũ hoặc `benchmark_v2_app.py`.

**Secrets**: `GEMINI_API_KEY = "..."` (hoặc `GOOGLE_API_KEY`) và tùy chọn `GEMINI_MODEL = "gemini-2.5-flash"`. Không commit khóa bí mật.

## Luồng sử dụng
1. Tải một ảnh ECG đã khử định danh (JPG/PNG/HEIC nếu có `pillow-heif`).
2. Nhấn **AI đọc cấu trúc 12 chuyển đạo**. Xem checklist từng lead, các giả thuyết có bằng chứng và giới hạn đọc ảnh.
3. Xác định layout **3×4 hoặc 6×2**, dải Rhythm II (nếu có). Xem **overlay ROI và từng ảnh crop**. Nếu crop không chuẩn, dùng mục **Sửa ROI riêng** ngay trên app để bảo toàn QRS, rồi mới xác nhận.
4. Ứng dụng chọn tối đa 6 vùng trong lượt 2: checklist II/V1/V2 và các lead nghi bất thường. Có thể đổi lựa chọn.
5. Bác sĩ xác nhận nhãn/ROI, nhấn lượt AI thứ hai và xem so sánh. Lượt hai nhận **chỉ ảnh crop, không nhận chẩn đoán lượt một**, nhằm hạn chế dẫn dắt.
6. Export ZIP JSON không kèm ảnh bệnh nhân mặc định.

## Giới hạn quan trọng
- ROI chỉ là **hình học xấp xỉ**; không tự phân lập QRS cao/sâu khi các lead chồng nhau. Sai crop có thể khiến AI đọc sai.
- Ảnh toàn cảnh trong lượt 1 có thể bị thu nhỏ và mất P/ST nhỏ. Crop lượt 2 không thay thế rà soát đủ 12 lead.
- Hai lần đọc bằng **cùng một AI** không phải kiểm chứng độc lập; kết quả lâm sàng luôn cần bác sĩ xác minh.
- Không có bộ đo định lượng đã được nghiệm thu trong v2.1; không tạo con số ms/mV không có căn cứ.
- Không chia sẻ ảnh chứa thông tin nhận dạng lên GitHub public. Ảnh gửi Gemini theo chính sách dịch vụ/API mà bạn dùng.
- Demo mới chỉ là nền tảng để thử A/B/C: 1 gốc, gốc + crop cố định, gốc + crop thích ứng. Chưa có kết quả chứng minh cấu hình C tốt hơn trên benchmark.

## Kiểm thử
`python -m pytest tests/test_visual_reader_v21.py -q`
`python -m py_compile visual_reader_v21_app.py core/visual_reader_v21.py core/gemini_visual_v21.py`

## Chương trình nghiệm thu
Trên ECG benchmark đã có ground truth: đánh giá AI đọc đúng lead, nhịp, ST/T, QRS; false-negative phát hiện nguy hiểm, phát biểu không có bằng chứng, kết quả ROI, thời gian/chi phí và tỷ lệ bất đồng. Lượt 2 phải chặn khi ROI bị cắt/không đúng nhãn.
