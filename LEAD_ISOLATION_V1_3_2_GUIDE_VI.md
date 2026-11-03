# ECG Core v1.3.2 - Lead Isolation & Boundary-Aware QA

## Khac biet so voi v1.3.1
- Giu nguyen `layout_box`, `waveform_box`, UI ROI, AI/Gemini/failover legacy.
- Tao mask ink ung vien co **quyen so huu duy nhat** theo lead, trong cung cot thoi gian.
- Vet ink noi 2 vung core, hoac nam trong vung canh tranh gan ranh gioi, duoc danh dau AMBIGUOUS va khong tinh cho lead nao.
- Xuat anh overlay cyan/orange, PNG mask 12 lead, `ambiguous_ink_mask_v132.png` va JSON.

## Gioi han can ghi nho
- Day la **pixel attribution hypothesis**, khong phai ECG waveform da so hoa.
- Net co quyen so huu duy nhat van co the la chu in, luoi giay, calibration hoac nhiễu.
- Net QRS co the bi gan AMBIGUOUS neu vuot qua ranh gioi hay dinh vao luoi.
- Tat ca lead de REVIEW, khong co do PR/QRS/QT/ST va khong chuan doan.

## Huong dan
1. Upload source len branch `refactor/ecg-core-v1`; giu `.streamlit/secrets.toml` tren Cloud, khong commit secrets.
2. App Streamlit giu entrypoint `app.py`.
3. Trong panel `ECG Core v1.3.2 | Layout + Lead Isolation Review`, upload 3 ECG benchmark.
4. Chinh boundary roi neu can, xem hai overlay va tai ZIP validation cho tung case.
5. Uu tien review V1/V2/V3 va chuoi dao dong bien do lon.
