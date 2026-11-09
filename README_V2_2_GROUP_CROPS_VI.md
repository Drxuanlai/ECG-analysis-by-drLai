# ECG Core v2.2 - 5 nhom anh ECG theo bo cuc

## Giao dien

App giu ten entrypoint `visual_reader_v21_app.py` de deployment Streamlit cu tiep tuc hoat dong. Tieu de giao dien da doi thanh ECG Core v2.2.

1. Upload **mot anh goc**.
2. AI Visual luot 1 nhan xet 12 lead.
3. Xac nhan bo cuc `3x4` hoac `6x2`, va **co/khong** co Rhythm II that.
4. Xem truoc cac ROI tung lead, sua lech neu can.
5. Xem truoc anh nhom. Nhieu anh nhom duoc tao bang cach ghep CAC ROI DA GAN NHAN thanh tung hang, khong tu goi la tin hieu dong thoi.
6. Xac nhan anh nhom -> Gemini luot 2 mo ta doc lap moi lead.
7. Xem bang doi chieu, canh bao logic, va xuat ZIP.

## 5 nhom

- `LIMB_I_III`: I, II, III
- `AUGMENTED`: aVR, aVL, aVF
- `PRECORDIAL_V1_V3`: V1, V2, V3
- `PRECORDIAL_V4_V6`: V4, V5, V6
- `RHYTHM_II`: II-rhythm - **chi co neu anh goc co dai II rieng**

Bieu do 3x4 va 6x2 duoc xu ly bang `layout_boxes` hien tai; khong tai su dung toa do cua bo cuc 3x4 sang 6x2. Neu khong co rhythm strip, bao cao ghi `rhythm_strip_available=false`.

## File thay doi

- `visual_reader_v21_app.py` - THAY FILE CU (giu ten entrypoint)
- `core/group_crops_v22.py` - THEM MOI
- `tests/test_group_crops_v22.py` - THEM MOI
- `README_V2_2_GROUP_CROPS_VI.md` - THEM MOI

Khong thay `core/gemini_visual_v21.py`: van doc `ECG_API_ROUTES` (4 Gemini key) nhu v2.1.1. Khong them dependencies.

## Gioi han nghien cuu

- ROI la goi y hinh hoc, co the cat R cao, S sau hoac chong lead; bac si can kiem tra.
- Anh ghep nhom khong dam bao cac nhip cung cot thoi gian; khong dung de suy cung mot nhip.
- AI luot 2 KHONG thay phan ket luan tu AI luot 1; hai luot AI khong phai mot phep do doc lap.
- Kiem tra mau thuan chi la heuristic van ban, khong duoc su dung de xac nhan benh ly.
- Khong duoc tu dong do ST, QTc, QRS hoac chan doan ECG voi pipeline digitizer chua nghiem thu.
- ZIP export mac dinh KHONG kem anh. Chi kem anh crop khi tick xac nhan va da khu dinh danh benh nhan.
- Nhom thu 5 khong duoc tu dong sinh tu doan DII ngan.

## Deploy

Branch `refactor/ecg-core-v1`, main file `visual_reader_v21_app.py`. Upload file thay doi vao dung duong dan; app tu cap nhat. Kiem tra Secrets cua app co `[[ECG_API_ROUTES]]`.

## Test

`python -m unittest tests.test_group_crops_v22 -v`
