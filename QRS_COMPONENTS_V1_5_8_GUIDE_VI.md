# ECG Core v1.5.8 — Calibration Pulse Rejection & Small-r Recovery

## Cập nhật PATCH ONLY (phụ thuộc v1.5.7)

- Thay `core/qrs_components.py`.
- Thay `ui/beat_panel.py`.
- Thêm `tests/test_qrs_components_v158.py`.
- Thêm tài liệu này.

Không sửa `app.py`, `requirements.txt`, Streamlit Secrets, Gemini API/failover, layout hay bộ nhãn bác sĩ.

## Thay đổi

1. **Đường nền cục bộ trước từng ứng viên S:** xử lý r nhỏ có thể nằm dưới đường nền trung vị của toàn lead do baseline drift. Kiểm tra nét mực ở khoảng 11–39 px trước S, sau đó dò r trong 3–20 px trước S. Không dùng điểm bác sĩ trong dự đoán.
2. **Xung chuẩn:** nhận dạng ứng viên dạng cạnh dốc đứng kết hợp đoạn thẳng nằm ngang (shelf) hoặc không trở về đường nền sau S. Chỉ khi có nhiều bằng chứng mới không đưa vào danh sách QRS. Ứng viên bị tách vẫn xuất trong `excluded_candidates` với lý do; không xóa khỏi lịch sử.
3. **Overlay:** dấu xanh lam = S ứng viên, ô tím = r ứng viên; dấu X đỏ = ứng viên nghi xung chuẩn; đường cam = đường nền toàn ROI (r dùng đường nền cục bộ, ghi trong JSON).
4. **Xuất:** `ecg_core_v1_5_8_qrs_components_validation.zip` chứa `qrs_components_v158.json` và `qrs_components_overlay_v158.png`.

## Kết quả trên V1 đã dùng để phát triển

- 3 ứng viên S: x = 151, 335, 510 px.
- Khôi phục được r ứng viên trước S đầu tiên: x ≈ 147, y ≈ 86 px (đường nền cục bộ ≈ 94 px).
- Ứng viên xung chuẩn: x ≈ 568 px, đưa vào `excluded_candidates`.

**Không xem các vị trí này là kết luận lâm sàng.** Kết quả chỉ trên một ROI đã dùng để phát triển. Vẫn phải đánh giá hồi quy đa máy, đa bố cục và các ECG bệnh lý. Không được suy ra QS nếu không tìm được r. Không được dùng các điểm ứng viên để định lượng PR/QRS/QT.

## Nghiệm thu

Chạy mục v1.5.8 riêng trên V1, sau đó DII và V4. Kiểm tra các dấu X đỏ không đè lên QRS thật; kiểm tra ba dấu S và r nhỏ đầu tiên. Xuất ZIP gửi đối chiếu.
