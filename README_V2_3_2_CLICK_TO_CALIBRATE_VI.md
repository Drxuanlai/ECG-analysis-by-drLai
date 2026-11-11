# ECG Core v2.3.2 — Click-to-Calibrate & Zoomed Landmark QA

## Mục đích
Sửa giao diện v2.3.1: không buộc người dùng đoán X/Y của đường lưới, không thu nhỏ cả Rhythm II thành ảnh ~63 px cao, không lưu mốc (0,0) khi chưa nhận được sự kiện click.

## Cập nhật trên GitHub
Upload các file sau vào đúng đường dẫn của branch đang chạy **ECG v2.3.1**:
1. `ui/grid_measurement_panel_v23.py` — thay file cũ.
2. `core/click_coordinates_v232.py` — thêm mới.
3. `tests/test_click_coordinates_v232.py` — thêm mới.
4. `README_V2_3_2_CLICK_TO_CALIBRATE_VI.md` — thêm mới.

Không thay app.py/visual_reader_v21_app.py, Gemini routes, direct-group crop, core/grid_evidence_v23.py hoặc benchmark. **Bắt buộc** kiểm tra `requirements.txt` có `streamlit-image-coordinates>=0.1.9,<1` để bấm ảnh hoạt động.

## A. Hiệu chuẩn lưới
1. Chọn lead và ấn đúng tốc độ giấy/gain theo chữ in hoặc xung chuẩn.
2. Chọn `X1`, bấm vào đường lưới đậm **đứng** thứ nhất trên ảnh phóng to.
3. Chọn `X2`, bấm đường lưới đậm **đứng** ngay kế bên (khoảng cách 5 mm).
4. Chọn `Y1`, bấm đường lưới đậm **ngang** thứ nhất; chọn `Y2`, bấm đường đậm ngang liền kế (5 mm).
5. So sánh đường đánh dấu với ảnh, rồi chọn `Xác nhận hiệu chuẩn lưới đã chọn`. Nếu không nhìn rõ lưới thì KHÔNG xác nhận và KHÔNG đo.

## C. Đánh dấu PQRST
1. Chọn cửa sổ phóng to (đoạn 1, 2, 3...) của Rhythm II, khoảng 520 px/đoạn.
2. Bấm lên chính nét ECG. Kiểm tra dòng `Đã nhận click: X=..., Y=...`.
3. Chọn beat ID và loại mốc, rồi `Lưu mốc bác sĩ`.
4. Nếu chưa có click hợp lệ, nút lưu bị khóa; không cho tự lưu `(0,0)`.
5. Nếu dùng `Nhập X/Y` thủ công, phải xác nhận đã kiểm tra tọa độ.

## Giới hạn kỹ thuật
- Bản này chưa có tự động tìm giao điểm lưới; chỉ số hóa thao tác bác sĩ bấm.
- Ảnh gốc có biến dạng, nếp gấp, lưới mờ cần kiểm tra cục bộ. Không có kết quả định lượng khi chưa hiệu chuẩn.
- Đã kiểm tra 19 unit tests logic; chưa chạy mô phỏng tương tác trên Streamlit Cloud / trình duyệt thật.
- Chỉ dùng QA nghiên cứu, không dùng độc lập cho chẩn đoán hoặc quyết định điều trị.
