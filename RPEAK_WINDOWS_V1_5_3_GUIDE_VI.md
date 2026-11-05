# ECG Core v1.5.3 - Segmented R-Peak Annotation

## Muc tieu
Chia ROI dai thanh cac cua so khong chong lan; moi diem bac si bam duoc doi sang toa do pixel cua **toan bo ROI goc**. Khong su dung cac diem nay de tu ket luan nhip hoac QRS.

## PATCH ONLY tren v1.5.2
Chi can them/thay 4 file:
- `ui/beat_panel.py` (thay)
- `core/rpeak_windows.py` (moi)
- `tests/test_rpeak_windows.py` (moi)
- `RPEAK_WINDOWS_V1_5_3_GUIDE_VI.md` (moi)

Khong upload lai `app.py`, benchmark, legacy. Giu `streamlit-image-coordinates>=0.1.9,<1` trong `requirements.txt` (da cai o v1.5.2). Dung branch `refactor/ecg-core-v1`.

## Cach su dung
1. Mo Beat-by-Beat Candidate Review, upload anh, chon dung 3x4/6x2 va ROI.
2. Trong v1.5.3, xem tung **Doan 1/N** -> **Doan N/N**, bam truc tiep tung dinh R va chon `Them diem R tham chieu`.
3. Moi khi da xem het mot doan (ke ca doan khong co QRS), bam `Xac nhan da ra soat HET doan nay`.
4. Thanh tien do cho biet bao nhieu doan duoc ra soat. Co the chon lai doan, hoan tac, hoac xoa diem trong doan.
5. Chi sau khi moi doan da ra soat moi mo khoa checkbox `Toi xac nhan da danh dau DAY DU...`.
6. Tai `ecg_core_v1_5_3_windowed_rpeak_validation.zip`; file moi `rpeak_reference_v153.json` chua trang thai tung doan va toa do ROI goc. File `rpeak_reference_v152.json` van duoc xuat de tuong thich.

## Khoi phuc v1.5.2
Trong muc `Khoi phuc diem R`, tai ZIP v1.5.2 truoc do. He thong kiem tra hash anh, ten lead va hop ROI. Neu khop, 8 diem cu duoc nhap, nhung tat ca doan phai duoc **ra soat lai** (khong ke thua checkbox `reference_complete` cu). Khong tu ghi de neu phien hien tai da co nhan.

## Quy tac an toan
- ROI dai 2556 px duoc chia 4 doan moi doan 639 px. Moi click tu doan i duoc chuyen ve x toan ROI = start_i + x_local * (width_i / display_width).
- Kich thuoc anh hien thi va toa do ROI la hai he rieng; chi ROI goc duoc xuat.
- Cac cau hinh voi anh/ROI khac se bi chan khi khoi phuc.
- Neu sua/xoa mot diem, trang thai da ra soat cua doan lien quan duoc dat lai de kiem tra.
- Tat ca QRS/R-peak cua may van la ung vien; khong tinh nhip tim tu ung vien.

## Kiem thu
- `python -m compileall -q ui/beat_panel.py core/rpeak_windows.py`
- `python -m pytest -q tests/test_rpeak_windows.py`
- Da chay regression 76/76 tests tren source ket hop v1.4 + cac patch v1.4.1..v1.5.2.

**Chua smoke-test tren Streamlit Cloud**; kiem tra click, next/prev, undo, restore, complete trong app sau khi deploy.
