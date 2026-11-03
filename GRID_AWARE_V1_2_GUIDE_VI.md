# ECG Core v1.2 – Grid-Aware QA

- Đây là một công cụ QA độc lập, không sửa các thuật toán Gemini/failover hoặc PQRST hiện hành.
- Mở app.py, dùng expander **ECG Core v1.2 | Grid-Aware Segmentation QA**.
- Upload 3 ảnh ECG đã khử định danh. HEIC cần pillow-heif; nếu hệ thống không có thư viện, xuất PNG bằng Preview.
- Xuất ZIP `ecg_core_v1_2_grid_aware_validation.zip` gồm hình đối chiếu 6 ô và JSON.
- Các ô theo thứ tự: ECG gốc | tăng tương phản | bản đồ màu lưới | mask v1.2 | mask v1.1 | overlay.
- **Không đánh giá PASS lâm sàng** từ density/IoU/coverage. Đường đẳng điện cũng có thể là đường ngang dài.
- Geometric 3×4 chỉ để QA: xem vùng crop, đặc biệt ảnh lệch, header, hoặc bố cục khác.
- Chạy `python -m pytest -q` hoặc `python scripts/verify.py`.
- Upload các file vào nhánh `refactor/ecg-core-v1` (không thay main); entrypoint vẫn `app.py`.
