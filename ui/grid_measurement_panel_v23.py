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
from core.full_beat_v241 import (propose_qrs_candidates, full_beat_windows,
    full_beat_prompt, validate_one_beat_landmarks, candidate_overlay)


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
    st.markdown('**E. v2.4.1 · Full-Beat Window & T-Wave Protection (QA)**')
    st.caption('Mỗi nhịp B1, B2… chỉ hình thành từ một QRS đã được em CHỌN để phân tích. '
               'Mọi ứng viên khác vẫn lưu UNRESOLVED, không tự coi là sóng T hay ARTIFACT.')
    raw_key=f'v241_raw_candidates_{fingerprint}'
    selected_key=f'v241_selected_ids_{fingerprint}'
    local_ai_key=f'v241_local_ai_{fingerprint}'
    _state(raw_key,[]);_state(selected_key,[]);_state(local_ai_key,[])
    left_col,right_col=st.columns(2)
    with left_col:
        if st.button('Tìm ứng viên QRS (chưa xác minh)',key=f'v241_candidates_{fingerprint}'):
            st.session_state[raw_key]=image_r_candidates(roi)
            st.session_state[selected_key]=[]
            st.session_state[local_ai_key]=[]
            st.rerun()
    with right_col:
        guided_mode=st.checkbox('Thêm ứng viên tại X do bác sĩ chỉ',
            key=f'v241_guided_{fingerprint}',
            help='Ứng viên có hướng dẫn; KHÔNG dùng để chấm detector độc lập.')
    if guided_mode:
        guided_x=st.number_input('X cửa sổ cần xem (pixel ROI)',0,roi.width-1,
            value=min(roi.width-1,roi.width//3),key=f'v241_guided_x_{fingerprint}')
        if st.button('Thêm ứng viên GUIDED',key=f'v241_addguided_{fingerprint}'):
            raw=list(st.session_state[raw_key]);raw.append({
                'x_pixel':int(guided_x),'source':'GUIDED_MANUAL_ANCHOR','score':0})
            st.session_state[raw_key]=raw
            st.session_state[selected_key]=[]
            st.session_state[local_ai_key]=[]
            st.rerun()

    candidates=propose_qrs_candidates(st.session_state[raw_key],roi.width)
    if candidates:
        st.info('Chọn đúng các phức bộ QRS thực. Sóng T nghi ngờ KHÔNG được chọn thành nhịp; '
                'các điểm không chọn vẫn giữ UNRESOLVED để tránh bỏ sót ngoại tâm thu.')
        suggested_ids=[c['candidate_id'] for c in candidates if c['suggested_qrs']]
        id_to_c={c['candidate_id']:c for c in candidates}
        current_choice=st.session_state[selected_key]
        default_choice=current_choice if current_choice else suggested_ids
        candidate_signature=hashlib.sha256(str([(c['x_pixel'],c['score']) for c in candidates]).encode()).hexdigest()[:10]
        choices=st.multiselect('Chọn các sự kiện QRS để phân tích P–QRS–ST–T',
            options=[c['candidate_id'] for c in candidates],default=default_choice,
            format_func=lambda cid:f"{cid} · X={id_to_c[cid]['x_pixel']} · {id_to_c[cid]['suggestion_status']}",
            key=f'v241_choice_{fingerprint}_{candidate_signature}')
        st.image(candidate_overlay(roi,candidates,choices),width='stretch',
                 caption='Xanh = đang chọn là QRS; cam = chưa chọn/UNRESOLVED. Tất cả ban đầu chỉ là ứng viên.')
        if st.button('Áp dụng các QRS đã chọn để tạo cửa sổ nhịp',
                     key=f'v241_apply_{fingerprint}'):
            st.session_state[selected_key]=list(choices)
            st.session_state[local_ai_key]=[]
            st.rerun()
        st.caption(f'Có {len(candidates)} ứng viên; {len(suggested_ids)} được xếp ưu tiên theo nét hẹp. '
                   'Đây KHÔNG phải số nhịp đã xác minh. Em phải bấm Áp dụng để tạo cửa sổ.')
    accepted_ids=set(st.session_state[selected_key])
    accepted=[dict(c, source=('USER_SELECTED_GUIDED_QRS_FOR_QA' if
                            c.get('source')=='GUIDED_MANUAL_ANCHOR' else
                            'USER_SELECTED_IMAGE_QRS_FOR_QA'))
              for c in candidates if c['candidate_id'] in accepted_ids]
    unresolved=[dict(c, event_status='UNRESOLVED_NOT_COUNTED_AS_BEAT')
                for c in candidates if c['candidate_id'] not in accepted_ids]
    with st.expander('Điều chỉnh phạm vi cửa sổ P–QRS–ST–T',expanded=False):
        st.caption('Tỷ lệ dựa vào khoảng giữa các QRS ĐÃ CHỌN, không giả định RR đều. '
                   'Cửa sổ được phép chồng lấp để giữ sóng T khi nhịp ngắn.')
        lf=st.slider('Phần khoảng RR trước QRS',.20,.60,.36,step=.02,
                     key=f'v241_left_{fingerprint}')
        rf=st.slider('Phần khoảng RR sau QRS',.55,1.00,.78,step=.02,
                     key=f'v241_right_{fingerprint}')
    windows=full_beat_windows(accepted,roi.width,left_fraction=lf,right_fraction=rf)
    signature=tuple((w['anchor_x'],w['x0'],w['x1']) for w in windows)
    signature_key=f'v241_window_signature_{fingerprint}'
    if st.session_state.get(signature_key)!=signature:
        st.session_state[local_ai_key]=[]
        st.session_state[signature_key]=signature
    if windows:
        st.success(f'{len(windows)} cửa sổ P–QRS–ST–T được tạo từ QRS em đã chọn; '
                   f'{len(unresolved)} ứng viên khác chưa phân định, không được tính là nhịp.')
        selected=st.selectbox('Chọn cửa sổ nhịp để đọc',range(len(windows)),
            format_func=lambda i:f"{windows[i]['beat_id']} · QRS≈{windows[i]['anchor_x']} · X {windows[i]['x0']}–{windows[i]['x1']}",
            key=f'v241_beat_{fingerprint}')
        win=windows[selected]
        old=[p for p in st.session_state[local_ai_key] if p.get('beat_id')==win['beat_id']]
        old=validate_one_beat_landmarks(evidence_checks(roi,old,windows),win)
        st.image(beat_overlay(roi,win,old,[]),width='stretch',
            caption=f"{win['beat_id']} · Full P–QRS–ST–T · X={win['x0']}–{win['x1']} px; "
                    'cam = QRS đã chọn; tím = AI chưa xác minh')
        st.caption('Cửa sổ là vùng quan sát, KHÔNG bảo đảm T đã thấy đủ; nếu T bị cắt '
                   'do ảnh hoặc nhịp sau đến sớm, không được tự điền T_END.')
        if win['coverage_warning']:
            st.warning(f"Cửa sổ sát biên: {win['coverage_warning']} — cần rà soát thành phần bị cắt.")
        if st.button('AI định vị PQRST của nhịp này',key=f'v241_ai_{fingerprint}',
                     disabled=not bool(routes)):
            cut=roi.crop((win['x0'],0,win['x1'],roi.height))
            prompt=full_beat_prompt(lead,win,cut.size)
            try:
                with st.spinner('AI xem toàn bộ P–QRS–ST–T của nhịp...'):
                    raw,used=analyze_with_routes(routes,prompt,
                        [(lead,image_for_model(cut,max_side=max(cut.size)))])
                    local=map_local_landmarks(extract_json(raw),win,roi)
                st.session_state[local_ai_key]=[
                    p for p in st.session_state[local_ai_key]
                    if p.get('beat_id')!=win['beat_id']]+local
                st.caption(f'{len(local)} tọa độ AI chưa xác minh · route {used}')
                st.rerun()
            except Exception as exc:
                st.error(f'AI chưa hoàn tất ({type(exc).__name__}). Kiểm tra Logs.')
        focus=validate_one_beat_landmarks(evidence_checks(roi,
              [p for p in st.session_state[local_ai_key] if p.get('beat_id')==win['beat_id']],windows),win)
        if focus:
            st.dataframe(focus,width='stretch',hide_index=True)
            st.image(beat_overlay(roi,win,focus,points),width='stretch',
                caption='Chỉ QA: tím = AI; xanh lá = bác sĩ. Mốc bác sĩ KHÔNG đưa vào prompt.')
    else:
        st.warning('Chưa có QRS được chọn để phân tích. Chọn ứng viên, bỏ sóng T giả '
                   'rồi bấm “Áp dụng”; hoặc thêm ứng viên GUIDED nếu detector bỏ sót.')
    all_predictions=[]
    for w in windows:
        individual=[p for p in st.session_state[local_ai_key] if p.get('beat_id')==w['beat_id']]
        all_predictions.extend(validate_one_beat_landmarks(evidence_checks(roi,individual,windows),w))
    comparisons=compare_to_human(all_predictions,points,tol_x=8,tol_y=8)
    if comparisons:
        with st.expander('So khớp X–Y với mốc bác sĩ (chỉ QA)',expanded=False):
            st.dataframe(comparisons,width='stretch',hide_index=True)
            st.caption('Không dùng mốc bác sĩ trong suy luận. Chế độ GUIDED không dùng để chấm detector.')
    st.caption('Không tự chuyển tọa độ AI thành số đo RR/QRS/ST. Chỉ các mốc bác sĩ '
               'đã duyệt trong phần C mới được dùng ở phần D.')

    export={'version':'2.4.1-QA','image_hash':image_hash,'layout':layout,'lead':lead,'roi_box':list(box),
        'roi_width':roi.width,'roi_height':roi.height,'grid':cal.to_dict() if cal else None,
        'ai_proposals_unverified':proposals,'human_landmarks':points,'measurements_human_landmarks':measured,
        'clinical_status':'RESEARCH_NOT_CLINICALLY_VALIDATED',
        'beat_v241':{'candidates_unverified':candidates,'selected_qrs_for_review':accepted,
                    'unresolved_candidates_not_beats':unresolved,'full_beat_windows':windows,
                    'ai_predictions_unverified':all_predictions,
                    'comparison_2d_to_human':comparisons,'independent_test_eligible':False}}
    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('measurement_evidence_v23.json',json.dumps(export,ensure_ascii=False,indent=2))
            z.writestr('README.txt','ECG 2.4.1 QA: Unselected candidates are UNRESOLVED, not beats. AI landmarks are unverified. Measurements require clinician-confirmed pixel landmarks and a manually calibrated local grid. No raw ECG image included.\n')
        data=buf.getvalue()
    st.download_button('Xuất ZIP Measurement QA v2.3 (không chứa ảnh)',data=data,file_name=f'ecg_core_v2_3_grid_evidence_{lead.replace("-","_")}.zip',mime='application/zip',key=f'v23_export_{fingerprint}')
