# ECG Core v1.3.4 — Row Boundary Refinement + Mask QC

## Thay đổi
- Giữ nguyên chế độ bố cục 3×4 và 6×2, mapping lead, Rhythm II do người dùng xác nhận.
- Đề xuất ranh giới hàng dựa trên vùng hoạt động nét tối theo hàng, với giới hạn dịch chuyển và trạng thái REVIEW.
- Mặc định KHÔNG tự áp ranh giới gợi ý; bác sĩ phải bật thử và đối chiếu overlay.
- QC mask mới tách chỉ số: ứng viên mực, straight-line, uncertain, nguy cơ nét gần biên, hỗ trợ theo cột.
- Xuất mask_qc_v134.json và từng ảnh qc_edge_overlay.png; không xóa uncertainty hoặc nội suy waveform.

## Cách thử
1. Upload ECG trong panel v1.3.4; chọn bố cục đúng.
2. Xem đề xuất ranh giới, tùy chọn áp đề xuất, điều chỉnh thanh Y nếu cần.
3. Kiểm tra đủ 12 ROI, nhất là aVR ở 6×2 và V1–V3 ở 3×4.
4. Xuất ZIP validation. So sánh lần trước và ảnh gốc.

## Giới hạn
Mọi ROI và mask đều REVIEW. Lọc lưới xám và phân biệt QRS khỏi chữ/xung chuẩn chưa được xác nhận chính xác. Không dùng định lượng y khoa.
