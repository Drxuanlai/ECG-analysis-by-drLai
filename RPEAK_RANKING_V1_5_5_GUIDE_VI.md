# ECG Core v1.5.5 — Morphology-aware R-Peak Ranking (PATCH ONLY)

Cài trên branch `refactor/ecg-core-v1` đã có v1.5.4.

## File cập nhật
- **Thay** `ui/beat_panel.py`
- **Thêm** `core/rpeak_filter_v155.py`
- **Thêm** `tests/test_rpeak_filter_v155.py`
- **Thêm** `RPEAK_RANKING_V1_5_5_GUIDE_VI.md`

Không thay đổi `app.py`, `requirements.txt`, Gemini, API failover, các nhãn ARTIFACT hoặc bộ đánh dấu R theo cửa sổ.

## Thay đổi
Bộ xếp hạng v1.5.4 thưởng quá mạnh các nét đứng kéo dài; có thể ưu tiên nhiễu hơn đỉnh R cao. v1.5.5 chấm điểm lại các gợi ý hiện có dựa trên:
1. Vị trí cực trị tương đối với đường nền *ước lượng từ các gợi ý yếu*, có kiểm tra tính tin cậy.
2. Độ đậm/độ tương phản ngay sát vị trí chấm.
3. Độ nổi bật của nét so với hai bên.
4. Độ mạnh gợi ý từ ảnh hiện tại.

Không dùng khoảng RR đều, không dùng nhãn R của bác sĩ để xếp hạng; giữ tất cả gợi ý bị hạ ưu tiên dưới `UNRESOLVED`. Điểm số chỉ ưu tiên kiểm tra, **không xác minh QRS**. Nếu ước lượng đường nền không đáng tin, không áp đặt giá trị đo.

## Kiểm thử V4 từ dữ liệu cung cấp
- 22 gợi ý đầu vào, tỷ lệ ưu tiên 35% → 8 gợi ý.
- v1.5.4: 1 trong 3 điểm R tham chiếu khớp ±5 px.
- v1.5.5: 3 trong 3 điểm khớp ±5 px.

**Lưu ý:** Đây là ca dùng trong quá trình thiết kế. Không sử dụng các tỷ lệ này làm ước lượng sensitivity trên tập độc lập; còn gợi ý giả chưa được loại hết.

## Chạy và xuất dữ liệu
Vào `Beat-by-Beat Candidate Review` trên Streamlit, chọn ROI V4, DII, V1 lần lượt, kiểm tra ảnh và đặt điểm R trên tất cả các cửa sổ. Tải `ecg_core_v1_5_5_rpeak_ranking_validation.zip`.

ZIP validation chứa `rpeak_filtered_v154.json` và `rpeak_ranked_v155.json` là hai bộ kết quả độc lập trên cùng ROI để đối chiếu, cùng overlay và nhãn người dùng.

**Không đo PR/QRS/QT, không suy ra nhịp tim hoặc chẩn đoán từ các gợi ý chưa xác minh.**
