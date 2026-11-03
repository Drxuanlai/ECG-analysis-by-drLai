"""QA-only layout review. Does not change the legacy ECG interpretation pipeline."""
from __future__ import annotations
import io
import json
import zipfile
import re
import hashlib
import streamlit as st
from PIL import Image, ImageOps
from core.lead_layout import detect_layout,resolve_layout,render_overlay,VERSION
from core.waveform_roi import refine_waveform_rois,render_dual_overlay,VERSION as ROI_VERSION
from core.grid_aware_segmentation import segment_grid_aware,diagnostic_sheet


def _read(upload):
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:pass
    im=ImageOps.exif_transpose(Image.open(io.BytesIO(upload.getvalue()))).convert('RGB')
    if max(im.size)>2800:im.thumbnail((2800,2800),Image.Resampling.LANCZOS)
    return im


def _png(im):
    b=io.BytesIO();im.save(b,format='PNG');return b.getvalue()


@st.cache_data(show_spinner=False,ttl=3600)
def _cached_layout(raw, enabled):
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:pass
    im=ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
    if max(im.size)>2800:im.thumbnail((2800,2800),Image.Resampling.LANCZOS)
    return detect_layout(im,ocr=enabled)


def render():
    with st.expander('ECG Core v1.3.1 | Layout + Waveform ROI Review',expanded=False):
        st.caption('Research QA độc lập. KHÔNG thay ROI hay kết quả của app AI cũ; không đo PQRST.')
        upload=st.file_uploader('Chọn ảnh ECG gốc 3×4 (PNG/JPG/HEIC)',type=['png','jpg','jpeg','webp','heic','heif'],key='layout_v13_upload')
        if upload is None:return
        try: im=_read(upload)
        except Exception as exc:st.error('Không mở được ảnh: '+str(exc));return
        w,h=im.size
        file_tag=hashlib.sha256(upload.getvalue()).hexdigest()[:10]
        use_ocr=st.checkbox('Tìm nhãn 12 chuyển đạo bằng OCR trên máy chủ (nếu có Tesseract)',value=True,key='layout_v13_ocr')
        with st.spinner('Đang phân tích bố cục ảnh ECG...'):
            layout=_cached_layout(upload.getvalue(),use_ocr)
        st.write(f"Layout: **{layout['confidence']}** · Cột: `{layout['x_method']}` · Hàng: `{layout['y_method']}` · OCR {len(layout['labels'])} nhãn")
        if layout['reasons']:st.warning('Cần kiểm tra ROI: '+', '.join(layout['reasons']))
        st.info('Vạch xanh/cam là ROI gợi ý, KHÔNG phải bằng chứng nhận dạng lead đã xác thực. Hãy đối chiếu nhãn và các QRS cao/âm sâu trên ảnh gốc.')
        st.markdown('**Điều chỉnh 4 cột cùng lúc** — chỉ sửa những ranh giới đang sai.')
        defs=layout['columns']; bounds=[0]
        cols=st.columns(3)
        for i in range(1,4):
            v=cols[i-1].slider(f'Đầu cột {i+1} (% chiều ngang)',0.,100.,round(100*defs[i]/w,1),.1,key='layout_v13_col_'+file_tag+'_'+str(i))
            bounds.append(round(w*v/100))
        bounds.append(w)
        st.markdown('**Điều chỉnh 3 hàng ECG** — 4 ranh giới chung, không kéo từng lead.')
        row_controls=[]
        cs=st.columns(4)
        for k in range(4):
            v=cs[k].slider('Y'+str(k)+' (%)',0.,100.,round(100*layout['rows'][k]/h,1),.1,key='layout_v13_y_'+file_tag+'_'+str(k))
            row_controls.append(round(h*v/100))
        selected=st.selectbox('Chỉnh riêng một lead (chỉ khi ROI chung chưa đủ)', ['Không']+[v for row in (('I','aVR','V1','V4'),('II','aVL','V2','V5'),('III','aVF','V3','V6')) for v in row],key='layout_v13_single')
        overrides={}
        if selected!='Không':
            st.caption('Trường hợp nếp gấp hoặc QRS vượt biên: chỉ điều chỉnh lead đã chọn.')
            row,col=next((r,c) for r,rr in enumerate((('I','aVR','V1','V4'),('II','aVL','V2','V5'),('III','aVF','V3','V6'))) for c,x in enumerate(rr) if x==selected)
            default=[bounds[col],row_controls[row],bounds[col+1],row_controls[row+1]]
            xx=st.slider('Lead '+selected+' · X trái / phải (%)',0.,100.,(round(100*default[0]/w,1),round(100*default[2]/w,1)),.1,key='layout_v13_manual_x_'+file_tag+'_'+selected)
            yy=st.slider('Lead '+selected+' · Y trên / dưới (%)',0.,100.,(round(100*default[1]/h,1),round(100*default[3]/h,1)),.1,key='layout_v13_manual_y_'+file_tag+'_'+selected)
            overrides[selected]=(round(xx[0]*w/100),round(yy[0]*h/100),round(xx[1]*w/100),round(yy[1]*h/100))
        try:rois=resolve_layout(im,layout,columns=bounds,rows=row_controls,per_lead=overrides)
        except ValueError as e:st.error('ROI chưa hợp lệ: '+str(e));return
        st.markdown('**Waveform ROI (màu cam):** mở rộng dọc an toàn để hạn chế cắt đỉnh R/đáy S. Khung bố cục xanh vẫn giữ nguyên. Vùng trùng giữa các lead là REVIEW, không phải tín hiệu đã xác nhận.')
        pad=st.slider('Đệm dọc waveform ROI (%)',3,16,9,1,key='roi_v131_pad_'+file_tag)
        ext=st.slider('Mức mở tối đa theo bằng chứng biên (%)',9,30,19,1,key='roi_v131_max_'+file_tag)
        selected_wave=st.selectbox('Chỉnh riêng waveform ROI (không làm đổi layout)', ['Không']+[v['lead'] for v in rois if v['lead']!='II-rhythm'],key='roi_v131_override_'+file_tag)
        wave_overrides={}
        if selected_wave!='Không':
            v=next(q for q in rois if q['lead']==selected_wave)
            bx0,by0,bx1,by1=v['box']
            st.caption('Chỉ kéo lên/xuống vùng trích xuất, không đổi nhãn hoặc cột thời gian.')
            py=st.slider('Waveform '+selected_wave+' · trên/dưới (% ảnh)',0.,100.,(round(100*max(0,by0-int((by1-by0)*pad/100))/h,1),round(100*min(h,by1+int((by1-by0)*pad/100))/h,1)),.1,key='roi_v131_wave_y_'+file_tag+'_'+selected_wave)
            wave_overrides[selected_wave]=[bx0,round(py[0]*h/100),bx1,round(py[1]*h/100)]
        try:wave_rois=refine_waveform_rois(im,rois,vertical_pad_ratio=pad/100,max_expand_ratio=ext/100,per_lead=wave_overrides)
        except ValueError as e:st.error(str(e));return
        overlay=render_dual_overlay(im,wave_rois)
        st.image(overlay,caption='XANH: layout box · CAM: waveform extraction box · TÍM: Rhythm II',width='stretch')
        flagged=[v['lead'] for v in wave_rois if any(z.get('possible_trace_collision') for z in v.get('waveform_overlaps',[]))]
        if flagged:st.warning('Waveform ROI chồng lên lead cùng cột: '+', '.join(flagged)+'. Cần kiểm tra nét sóng thực, đặc biệt V1–V3.')
        st.caption('Waveform ROI là đề xuất kỹ thuật. Nếu hai lead chồng sóng, không tự nhận rằng tín hiệu thuộc lead nào; phải REVIEW.')
        st.caption('Vạch ROI có thể chứa calibration và chữ; v1.3 chưa loại hoàn toàn các cấu trúc này. Dữ liệu tọa độ luôn trong hệ ảnh đã EXIF-corrected.')
        proceed=st.checkbox('Kiểm tra thêm segmentation v1.2 trong 12 ROI vừa chọn',value=False,key='layout_v13_seg')
        data={'version':ROI_VERSION,'image_name':upload.name,'layout':layout,
              'roi':[{'lead':v['lead'],'row':v['row'],'col':v['col'],'layout_box':v['layout_box'],'waveform_box':v['waveform_box'],'status':v['refinement_status'],'source':v['waveform_box_source'],'reasons':v['refinement_reasons'],'vertical_edge_evidence':v['vertical_edge_evidence'],'overlaps':v.get('waveform_overlaps',[])} for v in wave_rois],
              'warning':'ROI hypotheses only. No clinical measurements authorized.'}
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('layout_waveform_v131.json',json.dumps(data,ensure_ascii=False,indent=2))
            z.writestr('layout_waveform_overlay_v131.png',_png(overlay))
            if proceed:
                for v in wave_rois:
                    if v['lead']=='II-rhythm':continue
                    roi=im.crop(tuple(v['waveform_box']))
                    ev=segment_grid_aware(roi)
                    name=re.sub('[^A-Za-z0-9]','_',v['lead'])
                    comparison=diagnostic_sheet(ev)
                    z.writestr('lead_'+name+'_roi.png',_png(roi))
                    z.writestr('lead_'+name+'_segmentation.png',_png(comparison))
                    data.setdefault('segmentation',[]).append({'lead':v['lead'],'status':ev['status'],'metrics':ev['metrics']})
                z.writestr('segmentation_v12_metrics.json',json.dumps(data['segmentation'],ensure_ascii=False,indent=2))
        st.download_button('Tải validation Layout + Waveform v1.3.1 (.zip)',buf.getvalue(),
          'ecg_core_v1_3_1_waveform_roi_validation.zip','application/zip',key='layout_v13_zip')
        with st.expander('Xem tọa độ JSON và nhãn OCR',expanded=False):st.json(data)
