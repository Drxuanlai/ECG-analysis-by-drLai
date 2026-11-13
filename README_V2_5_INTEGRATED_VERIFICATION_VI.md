# ECG Core v2.5 - Integrated Verification Pipeline (research QA)

## Uu tien sua tu ca IMG_1394.jpg

- AI Visual pass 2 da thieu `II-rhythm` du da gui anh. v2.5 yeu cau co quan sat rieng ve II-rhythm va hien thi coverage gate.
- AI QRS Inventory chi co 9 goi y, mot so tile loi/thieu; KHONG duoc xem day la tong QRS. v2.5 hien bang trang thai tung tile, giu tile thanh cong va co nut `Chay lai CHI doan thieu/loi`.
- Python tao 24 ung vien, AI tao 9 de xuat; v2.5 gom 2 nguon thanh mot bang de bac si chon mot lan. Giao nhau KHONG xac nhan dung; AI-only/Python-only KHONG tu dong bi loai.
- QRS duoc bac si chon moi duoc tao beat window. Khong dung so luong trung binh RR de ep so beat; giu ngoai tam thu som.

## FILE CAN UPDATE

1. `visual_reader_v21_app.py` - THAY file cu (kiem tra coverage luot 2)
2. `ui/grid_measurement_panel_v23.py` - THAY file cu (inventory retry, bang review hop nhat)
3. `core/integrated_verification_v25.py` - THEM file moi
4. `tests/test_integrated_verification_v25.py` - THEM file moi

Giu nguyen `core/qrs_inventory_v242.py`, `core/full_beat_v241.py`, Gemini routes, 5 group crop va `requirements.txt`.

## QUY TRINH

1. Upload ECG da khu dinh danh; AI luot 1; xac nhan 5 khung; AI luot 2.
2. Neu luot 2 thieu bat ky lead nao, nhat la II Rhythm, chay lai luot 2. Khong coi coverage la clinical correctness.
3. Trong Measurement QA chon `II-rhythm`, chay AI Inventory. Neu tile thieu, bam `Chay lai CHI doan thieu/loi`. Chi khi n/n tile complete moi duoc duyet.
4. Bam `Tim ung vien QRS`. Xem bang AI+Python. Chon QRS thuc su tren net ECG, KHONG chon song T; neu bo sot, them GUIDED.
5. Xac nhan da xem TOAN BO dai va ap dung. Chi cac vi tri da duyet moi tao B1...Bn va cua so PQRST.
6. Calibrate grid va danh dau manual neu can do RR/QRS/ST. Du doan AI khong tu dong thanh so do.
7. Xuat ZIP QA. V2.5 van la nghien cuu, CHUA duoc dung doc lap cho quyet dinh dieu tri.

## KY VONG TEST

- `python -m compileall -q core ui visual_reader_v21_app.py`
- `PYTHONPATH=. pytest -q tests`
- Da kiem tra tai moi truong xay dung: 71 test pass. Chua kiem chung API Gemini va giao dien tren Streamlit Cloud.

## GIOI HAN

- Hai luot doc AI khong phai tham chieu lam sang doc lap.
- Danh sach hop nhat khong chung minh so QRS; bac si can xac nhan.
- So QRS chua duoc nghiem thu khi AI tile coverage khong day du hoac chua co nhan day du.
- Can dataset doc lap de tinh sensitivity/precision va do sai so toa do PQRST.
- V2.5 chi sua luong nghiem thu, khong tuyet doi hoa detector PQRST.
