# ECG Core v1.3.1 — Layout box / Waveform box

- **Xanh:** `layout_box` dùng gắn danh tính lead, không đổi so với v1.3.
- **Cam:** `waveform_box` cho kiểm tra tín hiệu; mở rộng theo chiều dọc nhưng không qua cột thời gian.
- **Tím:** `II-rhythm`.
- Vùng waveform overlap giữa hai lead cùng cột **phải REVIEW**; không đủ căn cứ tách hai đường sóng.
- Nếu nhãn V1–V3 bị sai hàng, chỉnh **layout** trước; waveform padding không sửa được gắn nhãn sai.
- Nếu sóng sâu chạm đáy, chỉnh riêng waveform ROI của lead mà không dịch hàng/cột chung.
- File `layout_waveform_v131.json` lưu cả hai box, trạng thái, lý do và vùng overlap.
- Mục tiêu là QA bố cục và tránh cắt biên. Không dùng phiên bản này để đo điện thế/độ rộng QRS hay chẩn đoán.
- Update branch `refactor/ecg-core-v1`; giữ Streamlit Secrets và `app.py`.
