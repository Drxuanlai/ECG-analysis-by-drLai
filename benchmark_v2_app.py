"""Standalone Streamlit QA app: deploy via benchmark_v2_app.py.
Does not import legacy app, Gemini, or old QA modules.
"""
import io
import json
import zipfile
import pandas as pd
import streamlit as st
from PIL import Image,ImageDraw
from core.benchmark_v2 import import_validation_zip,validate_reference,evaluate
from core.qrs_detector_v2 import detect_qrs_candidates

st.set_page_config(page_title='ECG Core v2 · Benchmark Lab',layout='wide')
st.title('ECG Core v2.0 · Benchmark & QRS Lab')
st.caption('Nghiên cứu / QA — không dùng để ra quyết định lâm sàng. Chỉ chấm hiệu năng khi nhãn tham chiếu đã đầy đủ, được xác minh và thuộc tập TEST độc lập.')

def export_zip(record,roi,pred,report,overlay):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for filename,data in [('reference_v2.json',record),('qrs_predictions_v2.json',pred),('evaluation_v2.json',report)]:
            z.writestr(filename,json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False))
        buf=io.BytesIO();roi.save(buf,format='PNG');z.writestr('roi_original.png',buf.getvalue())
        buf=io.BytesIO();overlay.save(buf,format='PNG');z.writestr('overlay.png',buf.getvalue())
    return out.getvalue()

inp=st.file_uploader('Tải ZIP validation từ v1.5.2–v1.5.9',type=['zip'],key='bench_v2_upload')
if not inp:
    st.info('Tải ZIP validation đã có. App sẽ lấy ROI và các dấu bác sĩ; KHÔNG mặc định các dấu đã được nghiệm thu đầy đủ.')
    st.stop()
try:
    artifact=import_validation_zip(inp.getvalue())
except Exception as exc:
    st.error(f'Không đọc được ZIP: {exc}');st.stop()
record=artifact['record'];roi=artifact['roi_image']
record_key=record['case_id']+'-'+str(hash(inp.getvalue()))
if st.session_state.get('benchmark_case_key')!=record_key:
    st.session_state.benchmark_case_key=record_key
    st.session_state.benchmark_record=record
rec=st.session_state.benchmark_record
st.subheader('1. Dữ liệu tham chiếu')
c1,c2,c3=st.columns(3)
c1.metric('Chuyển đạo',str(rec.get('lead')))
c2.metric('ROI',f"{roi.width} × {roi.height} px")
c3.metric('Mốc người gán',len(rec['reference_points']))
st.warning('Các nhãn cũ được nhập ở trạng thái UNVERIFIED. Nguồn phát triển/đã điều chỉnh thuật toán không được dùng làm tập TEST độc lập.')
with st.expander('Nhãn và dấu bác sĩ (chỉnh sửa)',expanded=False):
    edited=st.data_editor(pd.DataFrame(rec['reference_points']),num_rows='dynamic',hide_index=True,key='v2_editor')
    if st.button('Lưu bảng tham chiếu đã chỉnh'):
        try:
            points=[]
            for a in edited.to_dict('records'):
                if pd.isna(a.get('x')) or pd.isna(a.get('y')):continue
                points.append({'id':str(a.get('id')),'x':float(a.get('x')),'y':float(a.get('y')),
                               'type':str(a.get('type')),'source':'HUMAN','verified':False})
            rec['reference_points']=points
            st.success('Đã lưu bảng trong phiên QA. Cần xuất ZIP để lưu lâu dài.')
        except Exception as e: st.error(str(e))
st.markdown('**Kiểm soát phân chia tập dữ liệu**')
split=st.selectbox('Dataset split',['UNASSIGNED','TRAIN','VALIDATION','TEST'],index=['UNASSIGNED','TRAIN','VALIDATION','TEST'].index(rec['dataset_split']))
rec['dataset_split']=split
used_in_dev=st.checkbox('ECG / ROI này đã được dùng để thiết kế hoặc chỉnh thuật toán',value=rec.get('development_exposed',True))
rec['development_exposed']=bool(used_in_dev)
review_all=st.checkbox('Tôi đã rà soát toàn bộ chiều rộng ROI, bao gồm các nhịp bị bỏ sót',value=rec['all_windows_reviewed'])
complete=st.checkbox('Danh sách mốc QRS/Fiducial đã đầy đủ theo quy ước gán nhãn',value=rec['reference_complete'])
verified=st.checkbox('Tôi xác nhận các dấu đã được bác sĩ kiểm chứng trên ảnh gốc (không phải dự đoán máy)',value=rec['review_status']=='VERIFIED')
rec['all_windows_reviewed']=review_all
rec['reference_complete']=bool(complete and review_all)
rec['review_status']='VERIFIED' if verified else 'UNVERIFIED'
if split=='TEST' and used_in_dev: st.error('Không được dùng ca đã điều chỉnh thuật toán để tính chỉ số TEST độc lập. Hãy chọn TRAIN/VALIDATION.')
if split=='TEST':st.info('Chỉ dùng TEST cho ca chưa từng sử dụng để thiết kế hoặc chỉnh detector. Các ca đã tối ưu v1.x nên thuộc TRAIN/VALIDATION.')
errs=validate_reference(rec)
if errs:st.error('Lỗi nhãn: '+', '.join(errs))
st.subheader('2. QRS detector v2 độc lập')
min_dist=st.slider('Khoảng cách tối thiểu giữa ứng viên (pixel, chỉ là ràng buộc hình học)',8,100,24,2)
result=detect_qrs_candidates(roi,min_distance_px=min_dist)
pred=result['candidates']
st.caption('Detector chỉ nhận ảnh ROI. Không đọc nhãn bác sĩ, không dùng nhịp đều, không nhận định loại QRS hay chẩn đoán.')
preview=roi.copy();draw=ImageDraw.Draw(preview)
for p in pred:
    x=int(round(p['x']));draw.line((x,0,x,roi.height-1),fill=(230,105,0),width=2)
for p in rec['reference_points']:
    x,y=int(round(p['x'])),int(round(p['y']))
    draw.ellipse((x-5,y-5,x+5,y+5),outline=(0,180,70),width=3)
st.image(preview,caption='Cam = QRS event candidate (thời điểm X); xanh = mốc bác sĩ. Chưa suy diễn tọa độ Y của QRS event.',use_container_width=True)
st.metric('Ứng viên sự kiện (chưa xác minh)',len(pred))
with st.expander('Danh sách ứng viên'):st.dataframe(pd.DataFrame(pred),use_container_width=True)
st.subheader('3. Đối chiếu và đánh giá')
x_tol=st.slider('Dung sai thời gian theo pixel (chưa quy đổi ms)',2,30,10)
report=evaluate(rec,pred,x_tol=x_tol,y_tol=None)
if report['status']=='SCORED':
    st.success('Tham chiếu đầy đủ + verified + TEST độc lập: có thể tính chỉ số detection trên tập này.')
    col=st.columns(3)
    col[0].metric('TP',report['tp']);col[1].metric('FP',report['fp']);col[2].metric('FN',report['fn'])
    st.write('Sensitivity:',report['sensitivity'],'Precision:',report['precision'],'F1:',report['f1'])
else:
    st.warning('Chưa chấm sensitivity/precision. '+report.get('reason',''))
    st.write('Số cặp mốc khớp vị trí X phục vụ rà soát:',report['matched_count'])
    st.caption('Khớp X không đồng nghĩa định vị đúng R/S hoặc xác minh QRS. Với nhãn S_NADIR và QS_NADIR, bài toán đánh giá hiện chỉ là thời điểm sự kiện.')
st.download_button('Xuất ZIP benchmark v2 (ảnh + tham chiếu + dự đoán + báo cáo)',data=export_zip(rec,roi,result,report,preview),file_name=f"{rec['case_id']}_benchmark_v2.zip",mime='application/zip')
