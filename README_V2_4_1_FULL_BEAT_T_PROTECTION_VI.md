# ECG Core v2.4.1 - Full-Beat Window & T-Wave Protection

## Muc tieu

Sua loi v2.4: phan doan P-QRS ngan lam mat song T, con ung vien song T bi dem thanh nhip B2/B4.

## Co che an toan

1. Bo do net anh tra ve **ung vien**, KHONG phai nhung QRS da duoc xac minh.
2. Moi ung vien co ID C001... Chi nhung ung vien duoc nguoi dung **chon va bam Ap dung** moi tro thanh cac cua so B1, B2... de AI kiem tra. Ung vien khong chon luu `UNRESOLVED_NOT_COUNTED_AS_BEAT`, khong bi tu dong gan ARTIFACT/T.
3. Cua so co the chong lap, mo rong de thay P-QRS-ST-T theo khoang cach giua QRS da chon, khong ap dung RR deu hay nua do rong co dinh 95 px.
4. AI chi xem cua so nhac la nhin vao QRS duoc chon va song T sau QRS do; khong duoc tao them QRS tu T. Khi khong thay du T thi bo moc T_END, khong suy dien.
5. Co canh bao neu R/S do AI dua ra qua xa QRS anchor; AI khong duoc tu dong su dung de tinh ms/mV.
6. Cac moc bac si khong duoc dua vao prompt; khi dung guided mode thi danh gia khong doc lap.

## Update GitHub - PATCH ONLY

Branch: `refactor/ecg-core-v1`

- `ui/grid_measurement_panel_v23.py` - **thay file cu** dang chay v2.4.
- `core/full_beat_v241.py` - **them moi**.
- `tests/test_full_beat_v241.py` - **them moi**.
- `README_V2_4_1_FULL_BEAT_T_PROTECTION_VI.md` - **them moi**.

Khong thay `visual_reader_v21_app.py`, `app.py`, Secrets, bo 5 nhom crop, modul click, grid hay module `core/beat_anchor_v24.py` (van can de tao ung vien net anh va map toa do).

## Nghiem thu

1. Reboot Streamlit. Chon II-rhythm trong Measurement QA.
2. Keo xuong muc **E. v2.4.1** -> **Tim ung vien QRS**.
3. Xem tung vach, bo chon diem nam tai T hoac khong chac. Neu thieu QRS that, them `GUIDED` de QA (khong de cham detector).
4. **Ap dung cac QRS da chon**. Kiểm tra B1 gom du P-QRS-ST-T va T cua B1 khong thanh B2. Moi QRS that duoc chon phai van co cua so.
5. Neu can, chinh ty le truoc/sau QRS. AI dinh vi tren cua so day du, xem flags neu moc cach xa anchor.
6. Xuat ZIP Measurement QA va gui ket qua.

## Ket qua thu nghiem

- 27/27 bai test: 7 test moi v2.4.1; 7 test v2.4; 13 test click/grid v2.3.
- Smoke-test bo xep hang tren anh ECG Rhythm II: 16 ung vien, 14 de xuat uu tien, can nguoi dung chon lai. **Khong duoc coi la 14 QRS that.**
- Chua thu nghiem giao dien tren Streamlit Cloud hoac AI Gemini API thuc te.

## Gioi han

Day la QA nghien cuu. Viec chon phuc bo dung cho muc dich hien thi; KHONG phai ground truth chinh thuc. Bo tim ung vien v2.4 van co the bo sot QRS hoac nham song T. Khong duoc tu dong dung cac moc AI chua xac minh de do ECG cho cham soc benh nhan.
