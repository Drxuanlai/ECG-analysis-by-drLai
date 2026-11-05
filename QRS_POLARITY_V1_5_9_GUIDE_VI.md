# ECG Core v1.5.9 — Polarity-Aware QRS (PATCH ONLY)

## Mục tiêu

Khắc phục nhánh v1.5.8 chỉ ưu tiên S âm sâu và bỏ sót R dương của DII. Bổ sung nguồn ứng viên R dương độc lập, tiếp tục giữ nguồn S âm của v1.5.8. Không giả định RR đều, không coi ứng viên là QRS đã xác minh.

## Cập nhật GitHub (branch refactor/ecg-core-v1)

- `core/qrs_polarity.py`: **THÊM MỚI**.
- `ui/beat_panel.py`: **THAY** file cũ từ v1.5.8.
- `tests/test_qrs_polarity_v159.py`: **THÊM MỚI**.
- `QRS_POLARITY_V1_5_9_GUIDE_VI.md`: **THÊM MỚI**.

Giữ nguyên `app.py`, `core/qrs_components.py`, `legacy/`, `requirements.txt`, Secrets và toàn bộ các module khác. Phải có sẵn v1.5.8.

## Giao diện và kiểm thử

Vào `Beat-by-Beat Candidate Review` chọn ROI rồi mở `v1.5.9 · QRS hai cực tính`. Gợi ý màu tím: R dương; vòng xanh: S âm; viền cam: dấu dương chưa phân định tại biên/xung chuẩn. X đỏ là dấu đã được v1.5.8 nghi xung chuẩn, nhưng v1.5.9 **không coi là bằng chứng để tự xóa QRS thật**.

Nghiệm thu DII → V1 → V4. Xuất `ecg_core_v1_5_9_polarity_qrs_validation.zip`, gồm `qrs_polarity_v159.json`, `qrs_polarity_vs_reference_v159.json`, `qrs_polarity_overlay_v159.png` cùng các phần QA cũ để đối chiếu.

## Những gì đã được kiểm tra

- DII của ZIP v1.5.8: ba ứng viên R dương gần x=186, 363, 537. Ứng viên gần x=624 (vùng xung chuẩn cuối ROI) giữ trạng thái chưa phân định.
- V1: giữ ba ứng viên S âm, và vùng cuối ảnh chứa xung chuẩn không trở thành S được xác minh.
- V4: ba ứng viên R dương; ứng viên cận xung chuẩn được đánh dấu chưa phân định; S âm chạm đáy ROI không được xác minh.
- `PYTHONPATH=. pytest -q tests/test_qrs_components_v158.py tests/test_qrs_polarity_v159.py`: 9 PASS.

**Các tọa độ trên là ứng viên kỹ thuật, chưa được đối chiếu đầy đủ mốc tham chiếu độc lập.** Không kết luận accuracy, sensitivity/precision, khoảng QRS, nhịp tim, hay bệnh lý. Không áp dụng cho người bệnh thật nếu chưa đánh giá lâm sàng độc lập.

## Giới hạn còn tồn tại

Dò nét đứng và mask mực nhạy với chữ in, đường lưới, mép ROI và xung chuẩn. Phân loại chính xác rS/RS/qR/rSR' và QRS rộng cần mô hình hình thái được kiểm chứng độc lập; `polarity_proposal` hiện chỉ là nhãn loại bằng chứng. Các ứng viên vùng cuối ROI vẫn có thể là QRS thật và phải được bác sĩ rà soát.
