# ECG Core v2.3.1 — Direct Five-Group Cropping

## Mục tiêu
Thay **12 ROI trên ảnh toàn cảnh và phép ghép 12 crop** bằng **4/5 ROI nhóm hình chữ nhật trực tiếp từ ảnh ECG gốc**, đúng yêu cầu trải nghiệm nhanh của bác sĩ. Không động đến Gemini routes, prompt lượt 1/2, hay Grid Measurement v2.3.

## PATCH ONLY
- `visual_reader_v21_app.py`: **thay** file cũ tại root repo (giữ nguyên tên để không cần deploy app khác).
- `core/group_crops_v231.py`: **thêm mới**.
- `tests/test_direct_groups_v231.py`: **thêm mới**.
- `README_V2_3_1_DIRECT_GROUP_CROP_VI.md`: **thêm mới**.

Giữ nguyên `core/group_crops_v22.py`, `core/visual_reader_v21.py`, `ui/grid_measurement_panel_v23.py`, `core/grid_evidence_v23.py`, `core/gemini_visual_v21.py`, `requirements.txt`, `app.py`, và Secrets. V2.3.1 nhập một số utility và báo cáo từ v2.2; KHÔNG xóa v2.2.

## Quy tắc crop
- **3×4 + II rhythm:** 4 cột ECG (`I–III`, `aVR–aVF`, `V1–V3`, `V4–V6`) và một hình chữ nhật dài `II-rhythm` ở dưới. Có 5 ảnh JPG trực tiếp từ ảnh gốc.
- **6×2, không có rhythm:** hai cột, mỗi cột tách 2 nửa (3 hàng), tổng 4 ảnh. Không tạo II-rhythm giả.
- **6×2 có rhythm thật:** thêm ảnh thứ 5 nếu bác sĩ xác nhận trên ảnh có rhythm strip.
- App mặc định ước lượng geometry; luôn xem và có thể chỉnh 4/5 nhóm trước khi gửi AI. Crop *trực tiếp* từ ảnh gốc, giữ giấy/lưới/nhãn/xung chuẩn trong vùng. Không ghép từng lead.
- **Chú ý:** chỉnh các ROI lead ở mục đo chỉ đổi ROI dành cho Measurement QA. Không làm thay đổi ảnh nhóm. Chỉnh nhóm ở mục riêng để đổi ảnh gửi Gemini.
- ZIP xuất thêm `direct_group_boxes` và `group_crop_method= DIRECT_SINGLE_RECTANGLE_FROM_ORIGINAL` trong `evidence_report_v231.json`.

## Cách triển khai
1. Giải nén PATCH; upload 4 file đúng đường dẫn lên branch `refactor/ecg-core-v1`.
2. Đợi Streamlit build / Reboot app nếu cần. Main file path vẫn là `visual_reader_v21_app.py`.
3. Upload 1 ECG 3×4, chạy lượt 1, xác nhận bố cục, nhìn overlay: **chỉ 5 viền màu**; mở 'Xem trước 4/5 ảnh nhóm' để xác nhận chúng là các hình chữ nhật ảnh gốc, không có nhãn vẽ thêm.
4. Test tương tự với 6×2; nếu không có rhythm strip, bỏ chọn checkbox.
5. Nếu có cắt S sâu hoặc R cao, mở 'Chỉnh trực tiếp 4/5 KHUNG NHÓM', thay tọa độ rồi bấm lưu.
6. Xác nhận crop và chạy AI lượt 2. Export `ecg_core_v2_3_1_direct_group_visual_validation.zip`.

## QA
`python -m unittest discover -s tests -p 'test_direct_groups_v231.py' -v`

Đây là QA hình học, không phải chứng minh chính xác đọc ECG hay chẩn đoán. Mọi số đo ECG chỉ dùng khi calibration và landmark được bác sĩ duyệt.
