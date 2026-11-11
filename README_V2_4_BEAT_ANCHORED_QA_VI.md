# ECG Core v2.4 — Beat-Anchored Landmark QA (PATCH ONLY)

## Mục tiêu

Giữ nguyên ECG Core v2.3.1 (5 ảnh nhóm trực tiếp, AI Visual 2 lượt, API routes), giữ v2.3.2 (bấm lưới và mốc), thêm mục **E. v2.4 — Beat-Anchored Landmark QA** trong khu Measurement QA.

### Các bước trong giao diện
1. Upload 1 ECG, chọn đúng layout và năm nhóm như trước; chọn **II-rhythm** ở Measurement QA.
2. Hiệu chuẩn 4 đường lưới bằng click (X1/X2/Y1/Y2), xác nhận sau khi đối chiếu với đường lưới đậm.
3. Nhấn **Tìm ứng viên QRS từ ảnh**. Đây chỉ là tín hiệu đề xuất từ các nét đứng, có thể có **false positive**, không phải QRS đã xác nhận; không sử dụng để chẩn đoán hoặc tính RR tự động.
4. Chọn từng **cửa sổ nhịp** theo vị trí X. Xem ảnh crop chỉ chứa vùng gần ứng viên QRS. Nếu detector không tìm thấy, có thể bật **GUIDED** và nhập tọa độ X để kiểm thử giao diện; kết quả GUIDED không được coi là kiểm chứng độc lập của detector.
5. Nhấn **AI định vị PQRST chỉ trong cửa sổ này**. Gemini chỉ nhận ảnh cửa sổ QRS **không chứa mốc bác sĩ**; tọa độ AI được kiểm tra giới hạn và ánh xạ về ROI. Kết quả gắn `AI_BEAT_WINDOW_UNVERIFIED`.
6. Bảng QA báo các mốc ngoài cửa sổ, gần biên và vị trí không gần nét tối. Sau khi suy luận mới xem ảnh overlay **màu tím = AI**, **xanh lá = bác sĩ**.
7. Mở **So khớp 2D** để xem chênh lệch X/Y, nếu đã có tham chiếu bác sĩ. Sai lệch không làm thay đổi mốc AI; đây là phép QA, không phải bằng chứng độ chính xác lâm sàng.
8. Xuất ZIP Measurement QA v2.4; nội dung mới trong trường `beat_v24`.

### Thay đổi files (không thay app entrypoint)

- `ui/grid_measurement_panel_v23.py` — **thay** file cũ (dựa trên v2.3.2).
- `core/beat_anchor_v24.py` — **thêm** mới.
- `tests/test_beat_anchor_v24.py` — **thêm** mới.
- `README_V2_4_BEAT_ANCHORED_QA_VI.md` — **thêm** mới.

Thư viện: `numpy` và `scipy` phải có trong `requirements.txt` (hai thư viện đã được dùng trong lịch sử dự án). Nếu chưa có, thêm `numpy>=1.24` và `scipy>=1.11,<2`. Cần giữ `streamlit-image-coordinates>=0.1.9,<1` từ v2.3.2.

### Giới hạn / nghiệm thu

- Nhánh tìm nét đứng có thể sinh nhiều ứng viên giả trên lưới giấy, T cao, chữ, xung chuẩn hoặc QRS bệnh lý; **không** được xem danh sách là các beat đã xác minh.
- AI vẫn có thể định vị sai R/P/T, không chính xác đến từng pixel. Mốc AI **không tham gia tính ms/mV** ở mục D.
- Ở chế độ GUIDED, bác sĩ cung cấp vị trí cửa sổ; vì vậy đánh giá chỉ là kiểm thử **landmark localization có gợi ý vùng**, không phải hiệu năng detector end-to-end.
- Các cửa sổ và tọa độ dự đoán tự xóa khi thay đổi anchor hoặc chiều rộng cửa sổ. Khuyến nghị xuất ZIP trước khi thay đổi cấu hình.
- Chưa kiểm tra trực tiếp trình duyệt Streamlit Cloud; cần nghiệm thu thực tế trước khi dùng cho nghiên cứu định lượng.
- Bài test trên ảnh dùng để phát triển **không phải tập TEST độc lập**. Không tính sensitivity/precision lâm sàng từ chúng.
