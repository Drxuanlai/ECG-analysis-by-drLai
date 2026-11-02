"""Minimal Streamlit harness for ECG Digitizer Engine 1.0."""
import io, json
import streamlit as st
from PIL import Image, ImageDraw, ImageOps
from ecg_digitizer import digitize_ecg, result_summary

st.set_page_config(page_title="ECG Digitizer Engine 1.0",layout="wide")
st.title("ECG Digitizer — Phase 2 final test")
st.caption("Engine tách riêng khỏi UI. Auto layout → centerline → grid calibration → QC.")
up=st.file_uploader("Tải ECG",type=["jpg","jpeg","png","webp","heic","heif"])
if up:
    image=ImageOps.exif_transpose(Image.open(io.BytesIO(up.getvalue()))).convert("RGB")
    c1,c2=st.columns(2)
    with c1:
        speed=st.selectbox("Tốc độ giấy (chỉ chọn khi đã xác nhận)",["Chưa xác nhận",25,50])
    with c2:
        gain=st.selectbox("Gain (chỉ chọn khi đã xác nhận)",["Chưa xác nhận",2.5,5,10,20])
    result=digitize_ecg(image,speed_mm_s=None if isinstance(speed,str) else speed,
                        gain_mm_mV=None if isinstance(gain,str) else gain)
    st.metric("QC toàn ECG",result["overall_qc"]["status"])
    st.write("Calibration:",result["calibration"])
    # Preview boxes on oriented-ish input is approximate here; engine boxes refer to oriented image.
    st.dataframe([{"Lead":k,"QC":v["qc"]["status"],"Median conf":round(v["qc"].get("median_confidence_supported",0),3),
                   "Support":round(v["qc"].get("supported_fraction",0)*100,1),
                   "Edge":v["qc"].get("touches_edge_columns",0),"Lý do":", ".join(v["qc"].get("reasons",[]))}
                  for k,v in result["leads"].items()],hide_index=True)
    st.download_button("Tải JSON summary",json.dumps(result_summary(result),ensure_ascii=False,indent=2),
                       "digitizer_engine_summary.json","application/json")
    st.info("OCR/vision nhãn lead sẽ truyền vào engine qua label_anchors; bản engine không giả OCR khi chưa có OCR dependency/model.")
