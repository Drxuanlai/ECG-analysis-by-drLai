"""Optional measurement QA for ECG v2.3. All measurements require manual landmarks."""
from __future__ import annotations
import hashlib
import io
import json
import zipfile
import streamlit as st
from PIL import Image, ImageDraw

from core.grid_evidence_v23 import (GridCalibration,LANDMARKS,overlay,paper_values,
    sanitize_ai_landmarks,measure_points,validated_roi)
from core.click_coordinates_v232 import window_bounds, display_to_roi
from core.gemini_visual_v21 import analyze_with_routes
from core.visual_reader_v21 import image_for_model,extract_json
from core.beat_anchor_v24 import (windows_for_anchors,image_r_candidates,map_local_landmarks,
    evidence_checks,compare_to_human,beat_overlay)


def _state(slot:str,default):
    if slot not in st.session_state:st.session_state[slot]=default
    return st.session_state[slot]



def _windowed_click(roi, img, prefix, segment, window_px=520, display_width=1040):
    """Clickable zoomed image, returning an absolute coordinate in the full lead ROI.

    Only concrete click events produce coordinates. No (0,0) fallback.
    """
    x0, x1 = window_bounds(roi.width, segment, window_px)
    view = img.crop((x0, 0, x1, roi.height))
    shown_w = min(display_width, max(1, view.width * 2))
    shown_h = max(1, round(view.height * shown_w / view.width))
    view = view.resize((shown_w, shown_h), Image.Resampling.NEAREST)
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
    except ImportError:
        st.error('Thiếu streamlit-image-coordinates trong requirements.txt. Chức năng bấm trực tiếp chưa khả dụng.')
        st.image(view, width='stretch')
        return None
    click = streamlit_image_coordinates(view, width=shown_w, key=prefix)
    if not isinstance(click, dict):
        return None
    px, py = click.get('x'), click.get('y')
    if not isinstance(px, (int,float)) or isinstance(px,bool) or not isinstance(py, (int,float)) or isinstance(py,bool):
        return None
    if not (0 <= px < shown_w and 0 <= py < shown_h):
        return None
    return display_to_roi(px, py, roi.width, roi.height, segment, shown_w, shown_h, window_px)


def _page(roi, prefix, window_px=520):
    total=max(1, (roi.width + window_px-1)//window_px)
    idx=st.selectbox('Cửa sổ phóng to', range(total),
                     format_func=lambda n:f'Đoạn {n+1}/{total} · x={n*window_px}–{min(roi.width,(n+1)*window_px)-1} px',
                     key=prefix)
    return idx


def _grid_marks_image(roi, marks):
    img=roi.convert('RGB').copy()
    d=ImageDraw.Draw(img)
    for name, value in marks.items():
        if not isinstance(value, (int,float)):
            continue
        if name.startswith('X'):
            d.line((int(value),0,int(value),roi.height-1),fill=(20,150,255),width=2)
        else:
            d.line((0,int(value),roi.width-1,int(value)),fill=(20,150,255),width=2)
    return img


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
    st.markdown('**Chọn 4 đường lưới bằng cách bấm trên ảnh phóng to**')
    st.caption('Bấm X1 và X2 lên hai đường ĐẬM đứng liên tiếp (5 mm); Y1 và Y2 lên hai đường ĐẬM ngang liên tiếp (5 mm). Không chọn các đường nhỏ cách nhau 1 mm. Có thể chuyển đoạn khi cần.')
    marks_key=f'v232_gridmarks_{fingerprint}'
    marks=_state(marks_key,{'X1':None,'X2':None,'Y1':None,'Y2':None})
    target=st.radio('Đang chọn đường nào?', ['X1','X2','Y1','Y2'], horizontal=True,
                    key=f'v232_grid_target_{fingerprint}')
    seg=_page(roi,f'v232_grid_seg_{fingerprint}')
    calib_click=_windowed_click(roi, _grid_marks_image(roi,marks),
                     f'v232_gridclick_{fingerprint}_{target}_{seg}',seg)
    if calib_click is not None:
        value=calib_click[0] if target.startswith('X') else calib_click[1]
        if marks[target] != value:
            marks[target]=value
            st.session_state[marks_key]=marks
            st.rerun()
    st.caption('Các đường đã chọn: '+', '.join(f'{k}={v if v is not None else "chưa chọn"}' for k,v in marks.items()))
    if st.button('Xóa 4 điểm hiệu chuẩn',key=f'v232_cleargrid_{fingerprint}'):
        st.session_state[marks_key]={'X1':None,'X2':None,'Y1':None,'Y2':None}
        st.session_state.pop(f'v232_verified_{fingerprint}',None)
        st.rerun()
    have_four=all(marks.get(k) is not None for k in ('X1','X2','Y1','Y2'))
    proposed_cal=None
    if have_four:
        try:
            proposed_cal=GridCalibration(float(marks['X1']),float(marks['X2']),float(marks['Y1']),float(marks['Y2']),float(speed),float(gain))
            st.info(f'Kết quả từ bốn điểm: ngang {proposed_cal.px_per_mm_x:.2f} px/mm; dọc {proposed_cal.px_per_mm_y:.2f} px/mm. Hãy kiểm tra các đường màu xanh trên ảnh có đúng lưới đậm.')
        except ValueError as exc:
            st.error(f'Hiệu chuẩn chưa hợp lệ: {exc}')
    cal_signature=(tuple(marks.get(k) for k in ('X1','X2','Y1','Y2')),speed,gain)
    if st.button('Xác nhận hiệu chuẩn lưới đã chọn',disabled=proposed_cal is None,
                 key=f'v232_acceptcal_{fingerprint}'):
        st.session_state[f'v232_verified_{fingerprint}']=cal_signature
    cal=proposed_cal if proposed_cal is not None and st.session_state.get(f'v232_verified_{fingerprint}')==cal_signature else None
    if cal:
        st.success(f'Đã xác nhận: ngang {cal.px_per_mm_x:.2f} px/mm; dọc {cal.px_per_mm_y:.2f} px/mm. Mỗi px ngang = {cal.ms_per_px:.3f} ms.')
    else:
        st.warning('CHƯA XÁC NHẬN LƯỚI: không tính ms/mV. Các giá trị ví dụ 20/70 của bản cũ không được tự động dùng.')

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
    st.caption('Ảnh phóng to chia thành các đoạn ~520 px, không nén toàn bộ Rhythm II thành dải cao 63 px. Click lên mốc, kiểm tra tọa độ và bấm Lưu.')
    seg_m=_page(roi,f'v232_mark_seg_{fingerprint}')
    rendered=overlay(roi,cal,st.session_state[points_key],proposals)
    click=_windowed_click(roi,rendered,f'v232_mark_click_{fingerprint}_{seg_m}',seg_m)
    latest_key=f'v232_latest_{fingerprint}'
    if click is not None:
        st.session_state[latest_key]=(seg_m, *click)
    remembered=st.session_state.get(latest_key)
    last=(remembered[1],remembered[2]) if remembered and remembered[0]==seg_m else None
    if last is None:
        st.warning('CHƯA NHẬN ĐƯỢC CLICK. Không thể lưu mốc mặc định (0,0). Hãy bấm trực tiếp lên ảnh phóng to; nếu không phản hồi, kiểm tra đã cài streamlit-image-coordinates.')
    else:
        st.success(f'Đã nhận click: X={last[0]}, Y={last[1]} px trong ROI; tọa độ ảnh gốc ({box[0]+last[0]}, {box[1]+last[1]}).')
    c=st.columns(3)
    beat=c[0].text_input('Beat ID',value='B1',key=f'v23_beat_{fingerprint}')
    kind=c[1].selectbox('Loại mốc',LANDMARKS,key=f'v23_kind_{fingerprint}')
    source_mode=c[2].radio('Lấy vị trí',('Điểm bấm trên ảnh','Nhập X/Y'),key=f'v23_coord_mode_{fingerprint}')
    if source_mode=='Nhập X/Y':
        a,b=st.columns(2)
        x=a.number_input('X trong ROI',min_value=0,max_value=roi.width-1,
                         value=int(last[0]) if last else 0,key=f'v232_xmanual_{fingerprint}')
        y=b.number_input('Y trong ROI',min_value=0,max_value=roi.height-1,
                         value=int(last[1]) if last else 0,key=f'v232_ymanual_{fingerprint}')
        valid=st.checkbox('Tôi đã nhập và kiểm tra cả X/Y bằng tay',key=f'v232_manual_ack_{fingerprint}')
    else:
        valid=last is not None
        x,y=last if valid else (None,None)
    ca,cb=st.columns(2)
    with ca:
        if st.button('Lưu mốc bác sĩ',disabled=not valid,key=f'v23_add_{fingerprint}'):
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
    st.markdown('**E. v2.4 · Beat-Anchored Landmark QA (không thay mốc bác sĩ)**')
    st.caption('Tạo ứng viên QRS độc lập từ nét ảnh, chọn từng nhịp và yêu cầu AI đọc CỬA SỔ NHỊP; không dùng mốc bác sĩ làm đầu vào dự đoán.')
    anchors_key=f'v24_anchors_{fingerprint}'
    local_ai_key=f'v24_local_ai_{fingerprint}'
    _state(anchors_key,[])
    _state(local_ai_key,[])
    xa,xb=st.columns(2)
    with xa:
        if st.button('Tìm ứng viên QRS từ ảnh (chưa xác minh)',key=f'v24_candidates_{fingerprint}'):
            st.session_state[anchors_key]=image_r_candidates(roi)
            st.session_state[local_ai_key]=[]
            st.rerun()
    with xb:
        guided_mode=st.checkbox('Chế độ hướng dẫn: tự nhập X mốc QRS',key=f'v24_guided_{fingerprint}',
                                help='Chỉ để kiểm tra giao diện; không dùng để chấm detector độc lập.')
    if guided_mode:
        guided_x=st.number_input('Tọa độ X gợi ý cửa sổ (pixel trong ROI)',0,roi.width-1,
                                 value=min(roi.width-1,roi.width//3),key=f'v24_guided_x_{fingerprint}')
        if st.button('Thêm cửa sổ thủ công (GUIDED, không chấm detector)',key=f'v24_addguided_{fingerprint}'):
            temp=list(st.session_state[anchors_key]);temp.append({'x_pixel':int(guided_x),'source':'GUIDED_MANUAL_ANCHOR'})
            st.session_state[anchors_key]=temp
            st.session_state[local_ai_key]=[]
            st.rerun()
    anchors=st.session_state[anchors_key]
    half=st.slider('Nửa độ rộng cửa sổ quanh QRS (px)',45,180,95,step=5,
                   key=f'v24_half_{fingerprint}')
    windows=windows_for_anchors(anchors,roi.width,half_width=half)
    signature=tuple((w['anchor_x'],w['x0'],w['x1']) for w in windows)
    signature_key=f'v24_window_signature_{fingerprint}'
    if st.session_state.get(signature_key)!=signature:
        st.session_state[local_ai_key]=[]
        st.session_state[signature_key]=signature
    if anchors:
        st.caption(f'{len(anchors)} ứng viên độc lập/được thêm bằng tay; {len(windows)} cửa sổ. Đặc biệt: bộ dò nét đứng có thể nhận nhầm lưới, T hoặc xung chuẩn.')
    if windows:
        selected=st.selectbox('Chọn một cửa sổ nhịp để kiểm tra',range(len(windows)),
            format_func=lambda i:f"{windows[i]['beat_id']} · X≈{windows[i]['anchor_x']} · {windows[i]['anchor_source']}",
            key=f'v24_window_{fingerprint}')
        win=windows[selected]
        stale=[p for p in st.session_state[local_ai_key] if p.get('beat_id')==win['beat_id']]
        current=evidence_checks(roi,stale,windows)
        selected_img=beat_overlay(roi,win,current,[]) # do NOT show human reference before inference
        st.image(selected_img,width='stretch',caption=f"{win['beat_id']} · {win['x0']}–{win['x1']} px · Cam: mốc QRS ứng viên; tím: AI chưa xác minh")
        if win['anchor_source']=='GUIDED_MANUAL_ANCHOR':
            st.warning('GUIDED: cửa sổ được bác sĩ chọn. Không sử dụng đầu ra này để chấm hiệu năng detector tự động.')
        if st.button('AI định vị PQRST chỉ trong cửa sổ này',key=f'v24_ai_{fingerprint}',disabled=not bool(routes)):
            cut=roi.crop((win['x0'],0,win['x1'],roi.height))
            prompt=f"""Quan sát duy nhất một cửa sổ ECG của lead {lead}, kích thước ảnh {cut.width}x{cut.height} pixel. Không có thông tin nhãn bác sĩ. Gốc tọa độ (0,0) tại TRÁI TRÊN của ảnh NHỎ này. Trả từng vị trí P_ONSET,P_PEAK,P_OFFSET,QRS_ONSET,R_PEAK,S_NADIR,QRS_OFFSET,J_POINT,T_PEAK,T_END,BASELINE nếu nhìn rõ. Không ép có R dương; QS có thể không có R_PEAK. Không giả định nhịp đều, không đoán tọa độ từ ảnh toàn dải. Nếu không nhìn đủ nét thì bỏ mốc đó. Chỉ JSON {{"landmarks":[{{"landmark_type":"R_PEAK","x_pixel":30,"y_pixel":25,"evidence":"cực đại dương nét mực"}}]}}. Không nhận đường lưới hoặc chữ in là sóng. Chỉ mô tả vị trí; không chẩn đoán."""
            try:
                with st.spinner('AI đọc độc lập cửa sổ nhịp...'):
                    raw,used=analyze_with_routes(routes,prompt,[(lead,image_for_model(cut,max_side=max(cut.size)))])
                    local=map_local_landmarks(extract_json(raw),win,roi)
                # replace this beat only; never let clinician landmarks steer inference
                st.session_state[local_ai_key]=[p for p in st.session_state[local_ai_key] if p['beat_id']!=win['beat_id']]+local
                st.caption(f'{len(local)} mốc chưa xác minh; route {used}')
                st.rerun()
            except Exception as exc:
                st.error(f'AI không trả kết quả hợp lệ ({type(exc).__name__}); kiểm tra Logs và route.')
        current=evidence_checks(roi,st.session_state[local_ai_key],windows)
        focus=[p for p in current if p['beat_id']==win['beat_id']]
        if focus:
            st.dataframe(focus,width='stretch',hide_index=True)
            st.image(beat_overlay(roi,win,focus,points),width='stretch',
                     caption='Kiểm tra sau suy luận: tím = AI; xanh lá = bác sĩ. Mốc bác sĩ không tham gia prompt.')
    all_predictions=evidence_checks(roi,st.session_state[local_ai_key],windows)
    comparisons=compare_to_human(all_predictions,points,tol_x=8,tol_y=8)
    if comparisons:
        with st.expander('So khớp 2D với mốc bác sĩ (chỉ để QA)',expanded=False):
            st.dataframe(comparisons,width='stretch',hide_index=True)
            st.caption('So khớp một-một theo loại mốc, không dùng tọa độ bác sĩ để tạo dự đoán. Cửa sổ GUIDED phải loại khỏi đánh giá độc lập.')
    st.caption('Không tự dùng tọa độ AI để đo RR/QRS/ST. Chỉ các mốc bác sĩ đã duyệt mới được chuyển sang mục D.')

    export={'version':'2.4-QA','image_hash':image_hash,'layout':layout,'lead':lead,'roi_box':list(box),
        'roi_width':roi.width,'roi_height':roi.height,'grid':cal.to_dict() if cal else None,
        'ai_proposals_unverified':proposals,'human_landmarks':points,'measurements_human_landmarks':measured,
        'clinical_status':'RESEARCH_NOT_CLINICALLY_VALIDATED',
        'beat_v24':{'anchors':anchors,'windows':windows,'ai_predictions_unverified':all_predictions,
                    'comparison_2d_to_human':comparisons,'independent_test_eligible':False}}
    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('measurement_evidence_v23.json',json.dumps(export,ensure_ascii=False,indent=2))
            z.writestr('README.txt','ECG 2.3 QA: AI landmarks are unverified. Measurements require clinician-confirmed pixel landmarks and a manually calibrated local grid. No raw ECG image included.\n')
        data=buf.getvalue()
    st.download_button('Xuất ZIP Measurement QA v2.3 (không chứa ảnh)',data=data,file_name=f'ecg_core_v2_3_grid_evidence_{lead.replace("-","_")}.zip',mime='application/zip',key=f'v23_export_{fingerprint}')
