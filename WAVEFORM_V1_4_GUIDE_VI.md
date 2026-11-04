# ECG Core v1.4.0 — Waveform Reconstruction Foundation

## Mục tiêu
- Sử dụng đúng ROI sau khi bác sĩ chỉnh ở panel v1.3.4.
- Dựng **ứng viên đường sóng** từ bằng chứng nét tối, không giả định nhịp đều, QRS hẹp hay có sóng P.
- Giữ cột không có bằng chứng ở dạng `null` trong JSON/CSV (không nội suy, không đo đạc).
- Xuất 4 ô đối chiếu, overlay và bảng từng cột; mọi kết quả là `REVIEW`.

## Cách chạy
1. Mở panel `ECG Core v1.3.4 | Row Boundary + Mask QC`.
2. Chọn bố cục 3×4 hoặc 6×2; chỉnh ROI (đặc biệt DII ở 6×2) và kiểm tra overlay.
3. Mở `ECG Core v1.4 | Waveform Reconstruction Foundation` bên trong panel.
4. Chọn `II`, bật `Dựng tín hiệu 1-D từ ROI này (QA)`.
5. Xem điểm xanh/cam và đoạn đứt, tải `ecg_core_v1_4_waveform_validation.zip`.

## An toàn và giới hạn
- Không dựa vào v1.4 để phát hiện hoặc chẩn đoán rối loạn nhịp.
- Không chuyển pixel thành ms/mV; không đo PQRST, QT/QTc, ST.
- Không cho phép phần mềm sử dụng kết quả này làm clinical ground truth.
- Trace có thể bám lưới, chữ hoặc xung chuẩn, hoặc bỏ sót QRS thẳng đứng. Phải kiểm tra overlay.
- Ảnh ECG không được đăng public khi còn định danh người bệnh.

## Kiểm thử
Chạy `python -m pytest -q` ở thư mục gốc (cần pytest). Bài test synthetic là kiểm tra tính nhất quán phần mềm, không phải accuracy test.
