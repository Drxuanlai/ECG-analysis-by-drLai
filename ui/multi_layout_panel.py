"""Research QA panel for multi-layout, independent of legacy AI/failover."""
from __future__ import annotations
import io,json,zipfile,re,hashlib
import streamlit as st
from PIL import Image,ImageOps
from core.multi_layout import suggest_layout,propose_boxes,overlay,VERSION
from core.signal_artifact import separate,sheet
from core.row_boundary import propose_row_refinement, mask_qc, qc_overlay, VERSION as ROW_VERSION
from core.waveform_reconstruction import reconstruct, overlay as waveform_overlay, panel as waveform_panel, exportable as waveform_exportable
from core.spike_preservation import reconstruct_spike_aware, comparison_panel, export_comparison


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
    with st.expander('ECG Core v1.3.4 | Row Boundary + Mask QC',expanded=False):
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
        row_review=propose_row_refinement(im,initial)
        st.caption('Ranh giới hàng gợi ý từ hoạt động nét tối; không áp tự động khi chưa xác nhận.')
        with st.expander('Đề xuất ranh giới hàng / độ tin cậy',expanded=False):
            st.json(row_review)
        apply_row=st.checkbox('Thử các ranh giới hàng được đề xuất (vẫn cần kiểm tra bằng mắt)',False,key='ml134_rows_'+tag)
        if apply_row:
            initial=propose_boxes(im,chosen,has_rhythm=rhythm,row_boundaries=row_review['row_boundaries_suggested'])
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
                'layout_evidence':evidence,'row_refinement':row_review,'row_refinement_applied':apply_row,
                'disclaimer':'ROI REVIEW. Not clinically verified.'},ensure_ascii=False,indent=2))
            if do_masks:
                for item in proposal['rois']:
                    if item['lead']=='II-rhythm':continue
                    name=re.sub('[^A-Za-z0-9]','_',item['lead']);roi=im.crop(tuple(item['box']))
                    ev=separate(roi)
                    mq=mask_qc(roi,ev,lead=item['lead'])
                    met={'lead':item['lead'],'box':item['box'],'status':'REVIEW','signal_artifact':ev['metrics'],
                         'mask_qc':mq,'metrics':ev['metrics'],'reasons':list(dict.fromkeys(ev['reasons']+mq['reasons']))}
                    metrics.append(met)
                    z.writestr(f'masks/{name}_signal_artifact_panel.png',_png(sheet(roi,ev)))
                    z.writestr(f'masks/{name}_qc_edge_overlay.png',_png(qc_overlay(roi,ev,mq)))
                    for mask_name,arr in ev['masks'].items():
                        z.writestr(f'masks/{name}_{mask_name}.png',_png(Image.fromarray(arr)))
                z.writestr('signal_artifact_v133_metrics.json',json.dumps(metrics,ensure_ascii=False,indent=2))
                z.writestr('mask_qc_v134.json',json.dumps([x['mask_qc'] for x in metrics],ensure_ascii=False,indent=2))
        if metrics:
            st.dataframe(metrics,hide_index=True,width='stretch')
            for q in metrics:
                with st.expander(f"{q['lead']} · REVIEW · ink {q['metrics']['ecg_candidate_fraction']:.3f}"):
                    roi=im.crop(tuple(q['box']));ev=separate(roi)
                    st.caption('6 ô: original | raw dark | colored grid | suspected text/calibration | uncertain | candidate overlay')
                    st.image(sheet(roi,ev),width='stretch')
                    st.caption('Mask QC v1.3.4: xanh ứng viên mực, cam uncertain, đỏ dải sát rìa ROI.')
                    st.image(qc_overlay(roi,ev,q['mask_qc']),width='stretch')
                    st.json(q['mask_qc'])
        with st.expander('ECG Core v1.4 | Waveform Reconstruction Foundation (nghiên cứu)',expanded=True):
            st.caption('Sử dụng ROI đang hiển thị phía trên, bao gồm mọi chỉnh tay. Chỉ dựng centerline candidate; chưa QRS/PQRST, chưa ms/mV.')
            options=[v['lead'] for v in proposal['rois'] if v['lead']!='II-rhythm']
            rec_lead=st.selectbox('Chọn chuyển đạo kiểm chứng 1-D',options,index=options.index('II') if 'II' in options else 0,key='wf14_lead_'+tag)
            conf_threshold=st.slider('Ngưỡng bằng chứng pixel (cao hơn = bỏ nhiều cột chưa chắc)',0.25,0.75,0.42,0.01,key='wf14_thr_'+tag)
            st.info('Màu xanh: điểm ứng viên có bằng chứng. Màu cam: cột có nhiều mực cạnh tranh. Đoạn đứt: KHÔNG xác định, không được tự nối.')
            run_recon=st.checkbox('Dựng tín hiệu 1-D từ ROI này (QA)',value=False,key='wf14_run_'+tag)
            if run_recon:
                roi_item=next(v for v in proposal['rois'] if v['lead']==rec_lead)
                crop=im.crop(tuple(roi_item['box']))
                try:
                    traced=reconstruct(crop,min_evidence=conf_threshold)
                    st.image(waveform_panel(crop,traced),caption='Trên: ảnh ROI | ảnh tăng tương phản. Dưới: pixel ứng viên | đường centerline candidate (có gaps).',width='stretch')
                    st.write('**Trạng thái: REVIEW — chưa xác minh bằng ground truth.**')
                    st.json({'lead':rec_lead,'box':roi_item['box'],'metrics':traced['metrics'],'reasons':traced['reason_codes']})
                    # Null values stay null in export: not measured, not interpolated.
                    report={'lead':rec_lead,'roi_box':roi_item['box'],'image_name':up.name,
                            'version':'ECG Core v1.4', 'waveform':waveform_exportable(traced),
                            'clinical_use':'RESEARCH_ONLY; NOT CALIBRATED; NOT PQRST'}
                    z14=io.BytesIO()
                    with zipfile.ZipFile(z14,'w',zipfile.ZIP_DEFLATED) as z:
                        z.writestr('waveform_v14.json',json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
                        z.writestr('waveform_overlay_v14.png',_png(waveform_overlay(crop,traced)))
                        z.writestr('waveform_evidence_panel_v14.png',_png(waveform_panel(crop,traced)))
                        z.writestr('waveform_columns_v14.csv','x_pixel,y_pixel,confidence,supported,ambiguous\n'+''.join(
                            f"{x:.2f},{'' if y is None else round(y,2)},{c:.3f},{int(s)},{int(a)}\n"
                            for x,y,c,s,a in zip(traced['x_src'],traced['y_src'],traced['confidence'],traced['supported'],traced['ambiguous'])))
                    st.download_button(f'Tải waveform validation v1.4 · {rec_lead} (.zip)',z14.getvalue(),
                                       'ecg_core_v1_4_waveform_validation.zip','application/zip',key='wf14_dl_'+tag+rec_lead)
                    st.markdown('**ECG Core v1.4.1 | Vertical-Stroke / QRS Spike Preservation QA**')
                    st.caption('Dấu tròn đỏ chỉ nét dốc đứng ứng viên, không khẳng định R/QRS. Không ép đường đi qua chỗ thiếu mực.')
                    run_spike=st.checkbox('So sánh trước/sau bảo toàn nét dốc đứng',value=False,key='wf141_run_'+tag+rec_lead)
                    if run_spike:
                        comparison=reconstruct_spike_aware(crop,min_evidence=conf_threshold)
                        st.image(comparison_panel(crop,comparison),caption='Trên: ROI | mask nét đứng. Dưới: đường v1.4 | đường v1.4.1 + dấu đỏ nét ứng viên.',width='stretch')
                        st.json({'status':'REVIEW','lead':rec_lead,'metrics':comparison['refined']['metrics'],
                                 'spike_candidates':comparison['refined']['spike_candidates']})
                        z141=io.BytesIO()
                        with zipfile.ZipFile(z141,'w',zipfile.ZIP_DEFLATED) as z:
                            z.writestr('waveform_v141.json',json.dumps({'lead':rec_lead,'roi_box':roi_item['box'],
                                'image_name':up.name,'comparison':export_comparison(comparison)},
                                ensure_ascii=False,indent=2,allow_nan=False))
                            z.writestr('waveform_compare_v141.png',_png(comparison_panel(crop,comparison)))
                        st.download_button('Tải validation v1.4.1 · '+rec_lead+' (.zip)',z141.getvalue(),
                            'ecg_core_v1_4_1_spike_validation.zip','application/zip',key='wf141_dl_'+tag+rec_lead)
                except Exception as exc:
                    st.error('Không dựng được waveform: '+str(exc))
        
        st.download_button('Tải validation v1.3.4 (.zip)',filebuf.getvalue(),
                           'ecg_core_v1_3_4_row_mask_qc_validation.zip','application/zip',key='ml133_download_'+tag)
