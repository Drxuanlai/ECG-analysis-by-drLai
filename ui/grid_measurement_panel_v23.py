"""Optional measurement QA for ECG v2.3. All measurements require manual landmarks."""
from __future__ import annotations
import hashlib
import io
import json
import zipfile
import streamlit as st
from PIL import Image

from core.grid_evidence_v23 import (GridCalibration,LANDMARKS,overlay,paper_values,
    sanitize_ai_landmarks,measure_points,validated_roi)
from core.gemini_visual_v21 import analyze_with_routes
from core.visual_reader_v21 import image_for_model,extract_json


def _state(slot:str,default):
    if slot not in st.session_state:st.session_state[slot]=default
    return st.session_state[slot]


def render(photo:Image.Image,boxes:dict,layout:str,image_hash:str,routes:list[dict]):
    st.subheader('4 · v2.3 · Grid Map và đo mốc ECG (nghiên cứu)')
    st.caption('Ảnh lưới chỉ là lớp phủ hỗ trợ đo. Chưa có bộ nhận dạng giao điểm lưới tự động được xác minh.')
    st.info('Quy trình: chọn lead → đánh dấu 2 đường lớn ngang và 2 đường lớn dọc cách nhau 5 mm → xác nhận hiệu chuẩn → bấm các mốc PQRST → Python tính số đo. Gợi ý của AI KHÔNG phải mốc đã xác minh.')
    selectable=[l for l in ('II-rhythm','I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6') if l in boxes]
    lead=st.selectbox('Lead dùng để kiểm chứng',selectable,key='v23_lead')
    box=validated_roi(boxes[lead],photo.size)
    roi=photo.crop(box)
    fingerprint=hashlib.sha256(f'{image_hash}:{lead}:{box}'.encode()).hexdigest()[:16]
    points_key='v23_points_'+fingerprint
    _state(points_key,[])
    proposals_key='v23_ai_'+fingerprint
    _state(proposals_key,[])
    st.image(roi,caption=f'{lead}: ROI ảnh gốc {box}; kích thước {roi.width}×{roi.height} px.',width='stretch')

    st.markdown('**A. Hiệu chuẩn lưới 5 mm tại chính ROI này**')
    a,b=st.columns(2)
    with a:
        speed=st.selectbox('Tốc độ giấy (mm/s) — theo chữ in', [25.0,50.0,12.5,100.0],key='v23_speed')
    with b:
        gain=st.selectbox('Độ nhạy (mm/mV) — theo chữ in/xung chuẩn', [10.0,5.0,20.0],key='v23_gain')
    st.caption(f"Theo thông số chọn: 1 ô nhỏ = {paper_values(speed,gain)['small_box_ms']:g} ms theo ngang và {paper_values(speed,gain)['small_box_mv']:g} mV theo dọc. Không đồng nghĩa đã xác nhận khoảng px/ô.")
    with st.expander('Chọn 2 đường lưới ĐẬM liên tiếp cách nhau đúng 5 mm',expanded=True):
        st.caption('Các tọa độ sau tương đối so với góc trái trên của ảnh ROI; không phải tọa độ màn hình. Chỉ nhập khi nhìn rõ cả bốn đường lưới lớn.')
        c=st.columns(4)
        max_x=max(1,roi.width-1);max_y=max(1,roi.height-1)
        x0=c[0].number_input('Đường dọc 1: X',min_value=0,max_value=max_x,value=min(20,max_x),key=f'v23_x0_{fingerprint}')
        x5=c[1].number_input('Đường dọc 2: X',min_value=0,max_value=max_x,value=min(70,max_x),key=f'v23_x5_{fingerprint}')
        y0=c[2].number_input('Đường ngang 1: Y',min_value=0,max_value=max_y,value=min(20,max_y),key=f'v23_y0_{fingerprint}')
        y5=c[3].number_input('Đường ngang 2: Y',min_value=0,max_value=max_y,value=min(70,max_y),key=f'v23_y5_{fingerprint}')
        st.caption('Ví dụ x=20 và x=70 chỉ là GIÁ TRỊ NHẬP MẪU, không phải lưới đã tự phát hiện. Hãy chỉnh theo ảnh.')
        accept_grid=st.checkbox('Tôi đã kiểm tra hai cặp đường lưới lớn liên tiếp (mỗi cặp 5 mm) và hiệu chuẩn giấy.',key=f'v23_confirm_{fingerprint}')
    cal=None
    if accept_grid:
        try:
            cal=GridCalibration(float(x0),float(x5),float(y0),float(y5),float(speed),float(gain))
            st.success(f'Tỷ lệ đã xác nhận: ngang {cal.px_per_mm_x:.2f} px/mm; dọc {cal.px_per_mm_y:.2f} px/mm. 1 px = {cal.ms_per_px:.3f} ms hoặc {cal.mv_per_px:.5f} mV theo trục tương ứng.')
        except ValueError as exc: st.error(f'Chưa dùng được tỷ lệ lưới: {exc}')
    else:
        st.warning('Chưa xác nhận hiệu chuẩn lưới → KHÔNG tính ms/mV.')

    st.markdown('**B. Đề xuất vị trí mốc từ AI (chỉ để bác sĩ xem xét)**')
    if st.button('AI gợi ý tọa độ trên lead đang chọn',key=f'v23_infer_{fingerprint}',disabled=not bool(routes)):
        prompt=f'''Đây là ảnh crop của DUY NHẤT chuyển đạo {lead}, kích thước chính xác {roi.width} x {roi.height} pixel, hệ tọa độ gốc (0,0) ở góc trên trái, X sang phải, Y xuống dưới. Nhiệm vụ chỉ định vị các mốc nhìn thấy rõ từ nét ECG trên chính ảnh. ĐỪNG tự kết luận chẩn đoán và ĐỪNG tính độ rộng QRS/ST từ ước lượng. Nếu không xác định được, không trả mốc đó. Đừng suy các nhịp đều. Không xác nhận giả thuyết AI lượt trước. Đưa tối đa 30 mốc có bằng chứng, ưu tiên 1-3 phức bộ QRS đầu tiên rõ nét, xác định QRS_ONSET/QRS_OFFSET, R_PEAK hoặc S_NADIR, J_POINT, T_PEAK và BASELINE nếu thực sự rõ. Nếu là QS không có R, không tạo R_PEAK. Với mỗi mốc phải chỉ vị trí CHÍNH XÁC theo pixel của ảnh hiện tại, nếu không tự tin thì để null. Chỉ trả JSON dạng {{"landmarks":[{{"beat_id":"B1","landmark_type":"QRS_ONSET","x_pixel":123,"y_pixel":34,"evidence":"nét bắt đầu ở ..."}}]}}. KHÔNG gán mốc lên lưới hoặc chữ in, test milivolt.'''
        try:
            with st.spinner('AI đang đề xuất tọa độ ảnh; chưa có số đo xác minh...'):
                raw,used=analyze_with_routes(routes,prompt,[(lead,image_for_model(roi,max_side=max(roi.size)))])
                props=sanitize_ai_landmarks(extract_json(raw),roi.width,roi.height)
            st.session_state[proposals_key]=props
            st.caption(f'{len(props)} mốc gợi ý AI chưa xác minh; {used}. Không dùng tự động để tính.')
        except Exception as exc:
            st.error(f'AI landmark chưa chạy được: {type(exc).__name__}. Kiểm tra Logs và key/model.')
    proposals=st.session_state[proposals_key]
    if proposals:
        with st.expander(f'Các gợi ý AI chưa xác minh ({len(proposals)})',expanded=False):
            st.dataframe(proposals,width='stretch',hide_index=True)

    st.markdown('**C. Bác sĩ đánh dấu mốc trên nét điện tim**')
    rendered=overlay(roi,cal,st.session_state[points_key],proposals)
    display_w=min(950,roi.width)
    display_h=max(1,round(roi.height*display_w/roi.width))
    resized=rendered.resize((display_w,display_h))
    clicked=None
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        clicked=streamlit_image_coordinates(resized,width=display_w,key=f'v23_click_{fingerprint}')
        st.caption('Chạm vào ảnh: hệ thống quy đổi tọa độ ảnh hiển thị về pixel ROI gốc.')
    except ImportError:
        st.image(resized,caption='Chưa cài streamlit-image-coordinates. Có thể nhập tọa độ X/Y thủ công bên dưới.')
        st.caption('Để bấm trực tiếp lên ảnh, bổ sung streamlit-image-coordinates>=0.1.9,<1 vào requirements.txt.')
    if clicked and isinstance(clicked,dict) and isinstance(clicked.get('x'),(int,float)):
        cx=max(0,min(roi.width-1,round(clicked['x']*roi.width/display_w)))
        cy=max(0,min(roi.height-1,round(clicked['y']*roi.height/display_h)))
        st.session_state[f'v23_latest_{fingerprint}']=(cx,cy)
    last=st.session_state.get(f'v23_latest_{fingerprint}',(0,0))
    st.caption(f'Tọa độ bấm gần nhất: x={last[0]}, y={last[1]} pixel trong ROI. X toàn ảnh={box[0]+last[0]}, Y toàn ảnh={box[1]+last[1]}.')
    c=st.columns(4)
    beat=c[0].text_input('Beat ID',value='B1',key=f'v23_beat_{fingerprint}')
    kind=c[1].selectbox('Loại mốc',LANDMARKS,key=f'v23_kind_{fingerprint}')
    source_mode=c[2].radio('Lấy vị trí',('Điểm bấm trên ảnh','Nhập X/Y'),horizontal=False,key=f'v23_coord_mode_{fingerprint}')
    if source_mode=='Nhập X/Y':
        x=c[3].number_input('X trong ROI',min_value=0,max_value=roi.width-1,value=int(last[0]),key=f'v23_x_{fingerprint}')
        y=c[3].number_input('Y trong ROI',min_value=0,max_value=roi.height-1,value=int(last[1]),key=f'v23_y_{fingerprint}')
    else:x,y=last
    ca,cb=st.columns(2)
    with ca:
        if st.button('Lưu mốc bác sĩ',key=f'v23_add_{fingerprint}'):
            n=len(st.session_state[points_key])+1
            st.session_state[points_key].append({'id':f'M{n:03d}', 'lead':lead,'beat_id':beat.strip() or 'B1',
                 'landmark_type':kind,'x_pixel':int(x),'y_pixel':int(y),'source':'HUMAN_CONFIRMED',
                 'x_global_pixel':box[0]+int(x),'y_global_pixel':box[1]+int(y)})
            st.rerun()
    with cb:
        if st.button('Hoàn tác mốc cuối',disabled=not bool(st.session_state[points_key]),key=f'v23_undo_{fingerprint}'):
            st.session_state[points_key].pop();st.rerun()
    points=st.session_state[points_key]
    if points:st.dataframe(points,width='stretch',hide_index=True)
    st.markdown('**D. Phép đo minh bạch từ các mốc đã lưu**')
    if cal and points:
        measured=measure_points(points,cal)
        if measured:st.dataframe(measured,width='stretch',hide_index=True)
        else:st.info('Chưa đủ cặp mốc để tính. Cần ít nhất QRS_ONSET+QRS_OFFSET, R_PEAK+BASELINE, J_POINT+BASELINE hoặc ≥2 R_PEAK ở các beat khác nhau.')
    else:
        measured=[]
        st.warning('Chưa đủ điều kiện đo: phải xác nhận lưới và lưu các mốc bác sĩ.')
    st.caption('Chú ý: số đo phụ thuộc hiệu chuẩn và độ chính xác điểm bấm. ST_AT_J là điện thế tại J so với baseline cùng beat, KHÔNG tự suy ra STEMI/Brugada. Tránh dùng để quyết định điều trị.')
    export={'version':'2.3','image_hash':image_hash,'layout':layout,'lead':lead,'roi_box':list(box),
        'roi_width':roi.width,'roi_height':roi.height,'grid':cal.to_dict() if cal else None,
        'ai_proposals_unverified':proposals,'human_landmarks':points,'measurements_human_landmarks':measured,
        'clinical_status':'RESEARCH_NOT_CLINICALLY_VALIDATED'}
    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('measurement_evidence_v23.json',json.dumps(export,ensure_ascii=False,indent=2))
            z.writestr('README.txt','ECG 2.3 QA: AI landmarks are unverified. Measurements require clinician-confirmed pixel landmarks and a manually calibrated local grid. No raw ECG image included.\n')
        data=buf.getvalue()
    st.download_button('Xuất ZIP Measurement QA v2.3 (không chứa ảnh)',data=data,file_name=f'ecg_core_v2_3_grid_evidence_{lead.replace("-","_")}.zip',mime='application/zip',key=f'v23_export_{fingerprint}')
