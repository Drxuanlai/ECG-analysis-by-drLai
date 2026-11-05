# ECG Core v1.5.6 — QRS Fiducial & 2D Validation

## Cài đặt (PATCH ONLY trên v1.5.5)

Upload đúng đường dẫn trong nhánh `refactor/ecg-core-v1`:

- `ui/beat_panel.py` — thay thế file phiên bản v1.5.5.
- `core/fiducial_2d.py` — thêm mới.
- `tests/test_fiducial_2d.py` — thêm mới.
- `FIDUCIAL_2D_V1_5_6_GUIDE_VI.md` — hướng dẫn.

Không cần thay `app.py`, AI/failover, `requirements.txt` nếu môi trường v1.5.5 chạy được.

## Thao tác

1. Trong Beat Candidate Review, chọn ảnh/ROI đúng rồi đánh dấu trên các đoạn như v1.5.3.
2. Trước khi lưu mỗi dấu, chọn **loại mốc**: R_PEAK (đỉnh R dương), S_NADIR (đáy S), QS_NADIR (đáy phức bộ QS không có R dương), hoặc QRS_FIDUCIAL (mốc phức bộ chưa phân loại).
3. Nếu đã đánh dấu từ phiên bản cũ: phần **Sửa loại mốc đã đánh dấu** cho phép đổi loại nhưng giữ nguyên X/Y. Điểm từ v1.5.5 trở về trước được đọc dưới loại R_PEAK để bảo toàn tương thích; vui lòng kiểm tra lại đặc biệt ở V1.
4. Rà soát đầy đủ các đoạn. Mở mục **v1.5.6 · QRS fiducial + kiểm tra 2D**. Xem số điểm khớp XY, những điểm X đúng Y sai, và bảng có sai số X/Y riêng.
5. Xuất `ecg_core_v1_5_6_fiducial_2d_validation.zip`.

## Ý nghĩa kết quả

- `XY_WITHIN_TOLERANCE`: máy và bác sĩ đặt gần nhau về tọa độ **hai chiều**; CHƯA đủ chứng minh đó là cùng một loại sóng.
- `X_ONLY_Y_MISMATCH`: gần đúng thời điểm nhưng sai cao độ; **không tính như đỉnh R được định vị đúng**.
- `morphology_qc = UNVERIFIED`: chưa phân loại hình thái bằng thuật toán đã xác minh.
- `R_PEAK`, `S_NADIR`, `QS_NADIR`, `QRS_FIDUCIAL`: do người đọc chọn, không phải nhãn máy tự suy luận.
- So khớp một-một ưu tiên ghép XY hợp lệ; ứng viên không ưu tiên không bị xóa.
- Chỉ có thống kê hình học khi người dùng hoàn tất rà soát tất cả cửa sổ, KHÔNG gọi là sensitivity/precision chẩn đoán QRS.
- Không dùng quy luật RR đều, không tự đổi nhãn ARTIFACT, không đổi px sang ms khi chưa kiểm tra hiệu chuẩn.

## Kiểm tra trên dữ liệu đã gửi

Tại dung sai X=5px, Y=8px, dùng chính ba ZIP v1.5.5 để chạy đánh giá hình học:

| Lead | Tham chiếu | Khớp XY | Khớp ưu tiên XY | X khớp nhưng Y sai |
| --- | ---: | ---: | ---: | ---: |
| II | 3 | 1 | 1 | 2 |
| V4 | 3 | 2 | 2 | 1 |
| V1 | 3 | 0 | 0 | 3 |

Các con số này chỉ mô tả sai lệch trên ảnh kiểm thử cũ, không phải tỷ lệ phát hiện QRS lâm sàng. Đặc biệt V1 cần rà soát lại loại mốc thật (R/S/QS) trước khi coi là nhãn chuẩn.
