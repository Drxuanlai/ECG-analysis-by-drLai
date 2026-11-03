"""Research-only QA interface; isolated from Gemini/failover and original analysis."""
from __future__ import annotations
import io,json,zipfile
import streamlit as st
from PIL import Image,ImageOps
from core.grid_aware_segmentation import segment_grid_aware,diagnostic_sheet,VERSION

LEADS=[['I','aVR','V1','V4'],['II','aVL','V2','V5'],['III','aVF','V3','V6']]

def _open_image(raw):
    # Pillow HEIF registration optional; absence yields a precise error.
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:pass
    return ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')

def render():
    with st.expander('ECG Core v1.2 | Grid-Aware Segmentation QA',expanded=False):
        st.caption('Chỉ QA, không đo PQRST, không gọi Gemini và không xác nhận bệnh lý.')
        files=st.file_uploader('Ảnh ECG (có hỗ trợ HEIC nếu đã cài pillow-heif)',
            type=['png','jpg','jpeg','webp','heic','heif'],accept_multiple_files=True,key='grid_v12_files')
        if not files:return
        split=st.checkbox('Chia hình học 3×4 (QA בד; cần kiểm tra nhãn)',value=True,key='grid_v12_split')
        rec=[];buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            for fi,item in enumerate(files[:12]):
                try:
                    im=_open_image(item.getvalue()); im.thumbnail((2200,2200),Image.Resampling.LANCZOS)
                    crops=[]
                    if split:
                        w,h=im.size
                        for row in range(3):
                            center=(.205+.205*row)*h;half=.095*h
                            for col in range(4):
                                box=(int(col*w/4),max(0,int(center-half)),int((col+1)*w/4),min(h,int(center+half)))
                                crops.append((LEADS[row][col],im.crop(box),box))
                    else:crops=[('full',im,(0,0,*im.size))]
                    for lead,roi,box in crops:
                        ev=segment_grid_aware(roi)
                        info={'file':item.name,'lead':lead,'crop_box':box,'version':VERSION,
                            'status':ev['status'],'review_reasons':ev['review_reasons'],
                            'legacy_method':ev['legacy_method'],'metrics':ev['metrics']}
                        rec.append(info)
                        prefix=f'{fi+1:02}_{lead}'
                        with st.expander(f'{item.name} · {lead} · {ev["status"]}',expanded=False):
                            st.caption('6 ô theo thứ tự: original | enhanced | colored grid | grid-aware mask | v1.1 mask | overlay')
                            st.image(diagnostic_sheet(ev),width='stretch')
                            st.json(info,expanded=False)
                        for name,imobj in [('comparison',diagnostic_sheet(ev)),('grid',ev['grid_map']),('mask',ev['ink_mask']),('legacy_mask',ev['legacy_mask'])]:
                            b=io.BytesIO();imobj.save(b,format='PNG');z.writestr(f'{prefix}_{name}.png',b.getvalue())
                except Exception as exc:
                    st.warning(f'{item.name}: {type(exc).__name__}: {exc}')
                    rec.append({'file':item.name,'status':'FAIL','reason':str(exc)})
            z.writestr('grid_aware_v12_validation.json',json.dumps(rec,ensure_ascii=False,indent=2))
        st.write(f'QA: {len(rec)} vùng. Tất cả kết quả là REVIEW cho đến khi kiểm tra bằng ECG gốc.')
        st.download_button('Tải ZIP validation Grid-Aware v1.2',buf.getvalue(),
            'ecg_core_v1_2_grid_aware_validation.zip','application/zip',key='grid_v12_download')
