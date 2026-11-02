"""Standalone safe QA panel: separate upload, no API call, no clinical data sharing."""
import io
import json
import zipfile
import streamlit as st
from PIL import Image, ImageOps
from core.adaptive_segmentation import extract_ink, four_panel, VERSION


def render():
    with st.expander('ECG Core v1.1 | Adaptive Segmentation QA (thử nghiệm)',expanded=False):
        st.caption('Không sửa kết quả AI hiện có. Chỉ kiểm thử mask mực theo crop, KHÔNG xác nhận QRS/PQRST.')
        uploads=st.file_uploader('Chọn một hoặc nhiều ảnh ECG / crop chuyển đạo để kiểm thử',
            type=['png','jpg','jpeg','webp'],accept_multiple_files=True,key='adaptive_v1_upload')
        if not uploads:return
        auto=st.checkbox('Tự chia 12 chuyển đạo từ ảnh ECG bố cục 3×4 (chỉ để QA)',value=True,key='adaptive_auto_split')
        st.caption('Chia ROI theo hình học 3×4; không bảo đảm đúng với mọi định dạng. Kiểm tra từng crop trong ảnh validation.')
        zipbuf=io.BytesIO(); records=[]
        with zipfile.ZipFile(zipbuf,'w',compression=zipfile.ZIP_DEFLATED) as z:
            leadnames=[['I','aVR','V1','V4'],['II','aVL','V2','V5'],['III','aVF','V3','V6']]
            count=0
            for file_i,item in enumerate(uploads[:12]):
                try:
                    im=ImageOps.exif_transpose(Image.open(io.BytesIO(item.getvalue()))).convert('RGB')
                    if max(im.size)>2400:im.thumbnail((2400,2400),Image.Resampling.LANCZOS)
                    if auto:
                        # For printed 3x4+rhythm photos: exclude title and rhythm row.
                        # Deliberately generous vertical margins retain QRS excursions.
                        w,h=im.size
                        crops=[]
                        for row in range(3):
                            yc=(.205+row*.205)*h; half=.095*h
                            for col in range(4):
                                x0=int(w*(col/4));x1=int(w*((col+1)/4))
                                y0=max(0,int(yc-half));y1=min(h,int(yc+half))
                                crops.append((leadnames[row][col],im.crop((x0,y0,x1,y1))))
                    else:crops=[('image',im)]
                    for lead,roi in crops:
                        ev=extract_ink(roi)
                        prefix=f'file_{file_i+1:02}_{lead}'
                        record={'filename':item.name,'lead':lead,'method':ev['method'],'status':ev['status'],
                            'metrics':ev['metrics'],'size':ev['size'],'version':VERSION,
                            'note':'Geometry is provisional. No QRS detection/clinical validation.'}
                        records.append(record)
                        with st.expander(item.name+' | '+lead+' | '+ev['status']+' | '+ev['method']):
                            st.image(four_panel(roi,ev),caption='Original | Enhanced | Mask | Overlay',width='stretch')
                            st.json(record,expanded=False)
                        b=io.BytesIO();four_panel(roi,ev).save(b,format='PNG');z.writestr(prefix+'_four_panel.png',b.getvalue())
                        for k,m in ev['candidates'].items():
                            b=io.BytesIO();m.save(b,format='PNG');z.writestr(prefix+'_'+k+'.png',b.getvalue())
                        count+=1
                except Exception as exc:
                    records.append({'filename':item.name,'status':'FAIL','error':type(exc).__name__+': '+str(exc)})
                    st.error(item.name+': '+str(exc))
            st.caption(f'Đã xử lý {count} vùng QA. Xem từng vùng trong expander; tải ZIP để đối chiếu.')
            z.writestr('adaptive_v1_validation.json',json.dumps(records,ensure_ascii=False,indent=2))
        st.download_button('Tải toàn bộ ảnh 4-panel + masks + JSON (.zip)',zipbuf.getvalue(),
                           'ecg_core_v1_1_adaptive_validation.zip',mime='application/zip')
