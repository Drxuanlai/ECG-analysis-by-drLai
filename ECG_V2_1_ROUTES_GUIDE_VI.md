# ECG Core v2.1 — ECG_API_ROUTES compatibility patch

## Cập nhật

- Thay `visual_reader_v21_app.py`.
- Thay `core/gemini_visual_v21.py`.
- Thêm `tests/test_gemini_routes_v21.py` (không bắt buộc để chạy app).
- Các file khác giữ nguyên.

## Streamlit Secrets

Giữ nguyên TOML array-of-tables đang sử dụng ở app ECG legacy:

```toml
[[ECG_API_ROUTES]]
name = "Gemini chính"
provider = "Google Gemini"
model = "gemini-2.5-flash"
quota_project = "..."
api_key = "..."
```

Có thể cấu hình nhiều route. Không đưa key vào GitHub hay ZIP validation.
Nếu app v2.1 là deployment riêng, phải cấu hình Secrets trong chính deployment đó.

## Cách hoạt động

App dùng route theo thứ tự trong Secrets. Khi một lượt gọi lỗi, thử route kế tiếp.
Lượt 1 và lượt 2 có thể sử dụng route khác nhau. Giao diện hiển thị số route và nhãn route đã dùng, không hiển thị khóa.
Nếu tất cả route thất bại, xem logs, kiểm tra model có khả dụng và quota.

**Lưu ý:** patch này làm failover theo thứ tự cấu hình (priority fallback), chưa phải cân bằng tải round-robin. Không kiểm chứng model có tồn tại hay tài khoản có quyền truy cập. Không làm thay đổi app legacy hoặc các mô-đun phân tích ECG khác.
