"""Research QA panel for multi-layout, independent of legacy AI/failover."""
from __future__ import annotations
import io,json,zipfile,re,hashlib
import streamlit as st
from PIL import Image,ImageOps
from core.multi_layout import suggest_layout,propose_boxes,overlay,VERSION
from core.signal_artifact import separate,sheet


def _image(data):
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:pass
    im=ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert('RGB')
    if max(im.size)>2600:im.thumbnail((2600,2600),Image.Resampling.LANCZOS)
    return im

def _png(im):
    buf=io.BytesIO();im.save(buf,format='PNG');return buf.getvalue()

@st.cache_data(show_spinner=False,ttl=3600)
def _suggest(raw,enabled):
    return suggest_layout(_image(raw),ocr=enabled)

def render():
    with st.expander('ECG Core v1.3.3 | Multi-Layout + Signal/Artifact QA',expanded=False):
        st.caption('QA nghiên cứu; không thay thế kết quả AI/Streamlit legacy, không định lượng PQRST.')
        up=st.file_uploader('ECG 3×4 hoặc 6×2 (PNG/JPEG/HEIC)',type=['png','jpg','jpeg','webp','heic','heif'],key='ml133_upload')
        if not up:return
        raw=up.getvalue();tag=hashlib.sha256(raw).hexdigest()[:10]
        try:im=_image(raw)
        except Exception as e:st.error('Không đọc được ảnh: '+str(e));return
        w,h=im.size
        enabled=st.checkbox('Dò nhãn bằng OCR Tesseract nếu khả dụng',True,key='ml133_ocr')
        evidence=_suggest(raw,enabled)
        st.write(f"Gợi ý: **{evidence['suggested']}** · OCR {len(evidence['labels'])} nhãn · trạng thái REVIEW")
        st.caption('Nếu gợi ý UNKNOWN, phải chọn bố cục theo ảnh gốc. Ảnh 6×2 không được ép thành 3×4.')
        with st.expander('Chi tiết bằng chứng OCR',expanded=False):st.json(evidence)
        choices=['Chưa xác nhận','3x4','6x2','12x1']
        default=choices.index(evidence['suggested']) if evidence['suggested'] in choices else 0
        chosen=st.selectbox('Bố cục ECG (bác sĩ xác nhận)',choices,index=default,key='ml133_mode_'+tag)
        if chosen=='Chưa xác nhận':
            st.warning('Chưa xác định bố cục: dừng tạo lead ROI và mask. Hãy chọn đúng 3×4 hoặc 6×2 từ ảnh gốc.')
            return
        rhythm=False
        if chosen=='3x4':
            rhythm=st.checkbox('Có một dải Rhythm II dài riêng phía dưới',value=True,key='ml133_rhythm_'+tag)
        st.warning('Ảnh overlay chỉ là đề xuất ROI. Chữ/xung chuẩn và hình thái QRS có thể làm lệch ranh giới.')
        initial=propose_boxes(im,chosen,has_rhythm=rhythm)
        xs=initial['column_boundaries'];ys=initial['row_boundaries']
        ncol=len(xs)-1;nrow=len(ys)-1
        st.markdown('**Ranh giới cột** — chỉnh theo vạch chia lead thực, không theo xung chuẩn')
        cols=st.columns(max(1,ncol-1));new_x=[xs[0]]
        for k in range(1,ncol):
            v=cols[k-1].slider(f'X{k} (%)',0.,100.,round(100*xs[k]/w,1),.1,key=f'ml133_x_{tag}_{chosen}_{k}')
            new_x.append(round(v*w/100))
        new_x.append(xs[-1])
        st.markdown('**Ranh giới hàng** — chỉnh theo đường sóng trong từng hàng')
        new_y=[]
        for j in range(0,nrow+1,4):
            controls=st.columns(min(4,nrow+1-j))
            for q,k in enumerate(range(j,min(j+4,nrow+1))):
                v=controls[q].slider(f'Y{k} (%)',0.,100.,round(100*ys[k]/h,1),.1,key=f'ml133_y_{tag}_{chosen}_{k}')
                new_y.append(round(v*h/100))
        lead_names=[v['lead'] for v in initial['rois'] if v['lead']!='II-rhythm']
        target=st.selectbox('Chỉnh riêng ROI sai (tùy chọn)',['Không']+lead_names,key='ml133_lead_'+tag)
        edits={}
        if target!='Không':
            obj=next(v for v in initial['rois'] if v['lead']==target);x0,y0,x1,y1=obj['box']
            rx=st.slider(f'{target} · X trái/phải (%)',0.,100.,(round(100*x0/w,1),round(100*x1/w,1)),.1,key='ml133_single_x_'+tag+target)
            ry=st.slider(f'{target} · Y trên/dưới (%)',0.,100.,(round(100*y0/h,1),round(100*y1/h,1)),.1,key='ml133_single_y_'+tag+target)
            edits[target]=[round(rx[0]*w/100),round(ry[0]*h/100),round(rx[1]*w/100),round(ry[1]*h/100)]
        try:proposal=propose_boxes(im,chosen,has_rhythm=rhythm,column_boundaries=new_x,row_boundaries=new_y,per_lead=edits)
        except ValueError as exc:st.error(str(exc));return
        image_overlay=overlay(im,proposal)
        st.image(image_overlay,caption='12 lead boxes; Rhythm II chỉ có khi được chọn rõ ràng.',width='stretch')
        st.caption('Mask được tính theo ROI đã chọn. Tầng tách nhiễu không coi grid/text/calibration là ECG.')
        do_masks=st.checkbox('Tạo mask riêng 12 chuyển đạo (có thể tốn thời gian)',value=True,key='ml133_masks_'+tag)
        metrics=[];filebuf=io.BytesIO()
        with zipfile.ZipFile(filebuf,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('multi_layout_overlay_v133.png',_png(image_overlay))
            z.writestr('multi_layout_v133.json',json.dumps({'image_name':up.name,'image_size':im.size,'proposal':proposal,
                'layout_evidence':evidence,'disclaimer':'ROI REVIEW. Not clinically verified.'},ensure_ascii=False,indent=2))
            if do_masks:
                for item in proposal['rois']:
                    if item['lead']=='II-rhythm':continue
                    name=re.sub('[^A-Za-z0-9]','_',item['lead']);roi=im.crop(tuple(item['box']))
                    ev=separate(roi)
                    met={'lead':item['lead'],'box':item['box'],'status':'REVIEW','metrics':ev['metrics'],'reasons':ev['reasons']}
                    metrics.append(met)
                    z.writestr(f'masks/{name}_signal_artifact_panel.png',_png(sheet(roi,ev)))
                    for mask_name,arr in ev['masks'].items():
                        z.writestr(f'masks/{name}_{mask_name}.png',_png(Image.fromarray(arr)))
                z.writestr('signal_artifact_v133_metrics.json',json.dumps(metrics,ensure_ascii=False,indent=2))
        if metrics:
            st.dataframe(metrics,hide_index=True,width='stretch')
            for q in metrics:
                with st.expander(f"{q['lead']} · REVIEW · ink {q['metrics']['ecg_candidate_fraction']:.3f}"):
                    roi=im.crop(tuple(q['box']));ev=separate(roi)
                    st.caption('6 ô: original | raw dark | colored grid | suspected text/calibration | uncertain | candidate overlay')
                    st.image(sheet(roi,ev),width='stretch')
        st.download_button('Tải validation v1.3.3 (.zip)',filebuf.getvalue(),
                           'ecg_core_v1_3_3_multilayout_validation.zip','application/zip',key='ml133_download_'+tag)
