# ECG Core v1.5.2 – PATCH ONLY

## Cập nhật lên branch `refactor/ecg-core-v1`

**Nền tảng yêu cầu:** v1.5.1 đã hoạt động. Chỉ tải các file trong bản vá này lên đúng đường dẫn, không xóa các module cũ.

- **Thay** `ui/beat_panel.py`.
- **Thêm** `core/rpeak_review.py`.
- **Thêm** `tests/test_rpeak_review.py`.
- **Thêm 1 dòng** `streamlit-image-coordinates>=0.1.9,<1` vào **cuối** `requirements.txt` hiện có; KHÔNG thay thế các dependencies của project bằng một requirements mới.

Sau deploy, mở `ECG Core v1.5 | Beat-by-Beat Candidate Review`, chọn ECG và ROI chính xác, bấm chạy, kéo xuống mục `v1.5.2 · Đánh dấu đỉnh R tham chiếu độc lập`.

## Điểm mới

1. Khôi phục các **gợi ý hình ảnh** ở vị trí khác với danh sách sự kiện v1.5.1. **Không** coi chúng là QRS hoặc R peak thật.
2. Cho phép **bấm trên ảnh** rồi nhấn `Thêm điểm R tham chiếu`; hoặc nhập tọa độ pixel x/y. Có hoàn tác/xóa.
3. Bảng `event_rpeak_localization` ghi riêng `x_event`, `x_rpeak_hint`, `delta_x_px` và cờ chất lượng. Thuật toán không sửa nhãn người đọc.
4. Không tính sensitivity/precision khi bác sĩ chưa hoàn tất đánh dấu tất cả R. Kể cả khi hoàn tất, các chỉ số chỉ là **mức khớp tọa độ trong dung sai pixel**, không phải đánh giá chẩn đoán lâm sàng.
5. Nhãn `ARTIFACT` đã gán **không được máy tự đổi**. Nếu nằm sát R tham chiếu sẽ hiện cờ `artifact_label_conflicts`.
6. Sửa metadata `lead`, `image_name`, `roi_box` bị `null` trong `beat_events_v151.json`.

## Xuất file

`ecg_core_v1_5_2_rpeak_validation.zip` gồm dữ liệu v1.5/v1.5.1 như cũ và:
- `rpeak_image_hints_v152.json`
- `event_rpeak_localization_v152.json`
- `rpeak_reference_v152.json` (nhãn người đọc, tọa độ, cờ hoàn tất, so khớp)
- `rpeak_reference_overlay_v152.png`

**Giới hạn:** Không có đo HR, RR định lượng, PR/QRS/QT, không khẳng định phát hiện QRS đúng. Chưa có nhãn chuẩn đầy đủ trên benchmark; các đường gợi ý có thể dính nhiễu lưới, chữ, sóng T hoặc QRS cạnh nhau.
