# ECG Core v1.5 — UI Consolidation + Beat Candidate QA

## Những file thay đổi so với v1.4.2

- `app.py` — chạy app legacy không đổi, chỉ mount một khu ECG Core QA.
- `ui/multi_layout_panel.py` — giữ nguyên nội dung QA cũ, chuyển thành tùy chọn ẩn.
- `ui/beat_panel.py` — giao diện phát hiện ứng viên theo từng ROI.
- `core/beat_candidates.py` — ứng viên hình thái mực độc lập, không ép nhịp đều.
- `tests/test_beat_candidates.py` — unit tests.

Không xóa file từ v1.4.2. Thay `app.py`, `ui/multi_layout_panel.py`; thêm ba file còn lại.

## Giao diện

1. Phần AI đọc ECG cũ vẫn giữ nguyên phía trên.
2. Mở `ECG Core v1.5 · Nghiên cứu / QA` → `Beat-by-Beat Candidate Review`.
3. Chọn bố cục theo ảnh gốc (3×4 hoặc 6×2), chọn lead và xác nhận ROI. Có tùy chọn nhập tọa độ đã được điều chỉnh ở QA cũ.
4. Bật chạy và xem ảnh ứng viên trên ROI. Xuất ZIP.
5. Khi cần, bật `Mở công cụ QA chuyên sâu ... (legacy)` để truy cập v1.3.4–v1.4.2.

## Quan trọng

- **Không phải QRS detector đã được kiểm chứng.** Dấu cam chỉ là vị trí mực có hình thái dốc đứng. Ứng viên không đồng nghĩa một nhịp tim.
- Không tính HR, RR (ms), QRS duration, QT, PQRST, chẩn đoán hoặc hiệu chuẩn mV từ engine này.
- Không giả định RR đều; không loại bỏ các ứng viên chỉ vì RR không đều.
- Trên ảnh gốc, lưới/chữ/xung chuẩn và QRS rộng thấp dốc có thể tạo sai dương/âm. Tất cả là `CANDIDATE_REVIEW`.
- `legacy/app_v17_1_original.py`, AI providers và API failover không chỉnh sửa.

## Kiểm thử nhanh

```bash
python -m compileall -q app.py core ui tests
python -m pytest tests/test_beat_candidates.py -q
```

Để nghiệm thu lâm sàng, cần nhãn QRS do bác sĩ xác nhận; chưa tính độ nhạy/PPV nếu không có ground truth.
