# ECG Core v2.4.2 – AI QRS Inventory & Reconciliation

**PATCH ONLY trên nền v2.4.1. Nghiên cứu, không sử dụng đơn độc cho quyết định lâm sàng.**

## Những file cần upload GitHub

- `ui/grid_measurement_panel_v23.py` — **thay thế** file của v2.4.1.
- `core/qrs_inventory_v242.py` — thêm mới.
- `tests/test_qrs_inventory_v242.py` — thêm mới.
- `README_V2_4_2_AI_QRS_INVENTORY_VI.md` — thêm mới.

Không thay `app.py`, `visual_reader_v21_app.py`, `requirements.txt`, Secrets, 5 nhóm crop, v2.3.2 click/grid, module v2.4 và v2.4.1.

**Phải có sẵn dependencies của các bản trước**, nhất là `google-genai`, `scipy`, `streamlit-image-coordinates`.

## Cách sử dụng

1. Vào Measurement QA > lead `II-rhythm` > mục **E. v2.4.2**.
2. Bấm **AI KIỂM KÊ QRS (toàn bộ dải)**. Hệ thống chia ảnh Rhythm thành các đoạn 950 px, chồng lề 140 px, không suy QRS từ nhịp đều.
3. Xem `AI đề xuất N QRS`, bảng tọa độ, các trường hợp bị cắt ở mép ảnh, không rõ QRS và cảnh báo thiếu đoạn. Đây là **AI_UNVERIFIED**, không phải QRS được xác nhận.
4. Bấm **Tìm ứng viên QRS** của bộ dò Python để đối chiếu. Xem các vị trí đồng thuận, chỉ AI thấy và chỉ Python thấy. Trùng tọa độ **không chứng minh** ứng viên đó là QRS; bất đồng **không chứng minh** là giả.
5. Rà soát toàn ảnh, chỉ chọn ứng viên Python trùng QRS thực. Nếu QRS thật thiếu, thêm ứng viên `GUIDED` bằng tọa độ X, nhưng phải ghi nhận đây là dữ liệu bác sĩ hướng dẫn (không chấm detector độc lập).
6. Đánh dấu **Tôi đã rà soát HẾT dải II Rhythm** rồi nhấn **Áp dụng QRS đã duyệt**. Chỉ bấy giờ mới cấp B1…Bn và tạo cửa sổ P–QRS–ST–T.
7. Đọc AI trong từng cửa sổ và xuất ZIP Measurement QA v2.4.2. `qrs_inventory_v242`, `qrs_reconciliation_v242`, `beat_v241` đều có trong JSON.

## An toàn dữ liệu và kiểm chứng

- Mỗi ảnh con được gửi đến Gemini cùng prompt độc lập; không gửi mốc bác sĩ làm đầu vào để AI tự tạo dự đoán.
- Dự đoán của Gemini là tọa độ **ước lượng**; hình ảnh không được tự động coi là có QRS chỉ vì AI cung cấp số tọa độ.
- Có nhiều đoạn chồng lề nên một QRS ở vùng tiếp giáp có thể được đề xuất hai lần. Module gom các vị trí cách nhau ≤15 px và lưu cờ `overlap_merge_review`; trường hợp mép cần xem lại.
- `AI_ONLY` và `PYTHON_ONLY` là **bất đồng hai bộ dò**, không được gọi TP/FP/FN nếu không có ground truth.
- Không ép nhịp đều hoặc số QRS giống nhau. Không tự bỏ ngoại tâm thu sớm/QRS rộng/QS.
- Các phép đo ECG vẫn chỉ dùng các landmark bác sĩ duyệt và hiệu chuẩn đã xác nhận.

## Giới hạn

- Gemini có thể đếm thiếu, đếm thừa hoặc ước lượng tọa độ sai. Lượt kiểm kê không phải bằng chứng định lượng.
- Một QRS thật có thể nằm ngay biên giữa hai đoạn; AI phải gắn `partial_events`; cơ chế ghép vẫn cần duyệt.
- Bộ dò Python có thể nhầm sóng T, lưới hoặc xung chuẩn; mặc định không tự chọn QRS theo điểm số.
- Chưa chạy tích hợp Streamlit/Gemini thật từ môi trường phát triển; cần nghiệm thu Cloud.
