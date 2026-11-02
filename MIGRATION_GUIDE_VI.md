# Hướng dẫn cập nhật GitHub an toàn

1. Trên GitHub tạo branch mới `refactor/ecg-core-v1` từ `main`. Không sửa `main` lúc đầu.
2. Giải nén ZIP. Upload **nội dung bên trong thư mục** `ECG_Core_v1_0`, không upload chính ZIP để chạy.
3. Trong Streamlit Cloud tạo deployment thử nghiệm dùng branch mới và file chính `app.py`. Secrets giữ nguyên ở dashboard; không chuyển chúng vào GitHub.
4. Nếu app báo thiếu `ecg_digitizer.py`, lấy file gốc đúng phiên bản từ repo cũ và đưa vào root branch mới; nó không nằm trong các file được cung cấp lần này.
5. Test ảnh ECG đang dùng: upload, ROI, Rhythm II, phân tích AI, download JSON/PNG, reset case. So sánh cùng dữ liệu với `main`.
6. Nếu khác kết quả hoặc lỗi, không merge. Quay về deployment ở `main`. Chỉ merge sau khi kiểm chứng đầy đủ.

Lưu ý: Đây là bước dựng khung + bảo toàn source, **chưa tách toàn bộ hàm ra module**; làm vậy một lần với app 6459 dòng dễ gây regression.
