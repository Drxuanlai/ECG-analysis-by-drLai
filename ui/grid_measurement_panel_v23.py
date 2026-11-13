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
from core.qrs_inventory_v242 import (strip_tiles,inventory_prompt,parse_tile,
    merge_tile_inventory,reconcile,inventory_overlay)
from core.integrated_verification_v25 import (tile_coverage, reconcile_proposals, qa_gate)


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
    st.markdown('**E. v2.4.2 · AI QRS Inventory → Reconciliation → Full Beat (QA)**')
    st.caption('Mỗi nhịp B1, B2… chỉ hình thành từ một QRS đã được em CHỌN để phân tích. '
               'Mọi ứng viên khác vẫn lưu UNRESOLVED, không tự coi là sóng T hay ARTIFACT.')
    raw_key=f'v241_raw_candidates_{fingerprint}'
    selected_key=f'v241_selected_ids_{fingerprint}'
    local_ai_key=f'v241_local_ai_{fingerprint}'
    inventory_key=f'v242_inventory_{fingerprint}'
    _state(raw_key,[]);_state(selected_key,[]);_state(local_ai_key,[])
    _state(inventory_key,{})
    st.markdown('**E1 · AI đếm và chỉ vị trí từng QRS trên toàn dải**')
    st.caption('AI đọc nhiều đoạn có vùng chồng lề để giữ độ phân giải. Không dự đoán số QRS từ RR, '
               'không ép nhịp đều và không sử dụng mốc bác sĩ trong prompt.')
    tiles=strip_tiles(roi.width,tile_width=950,overlap=140)
    st.caption(f'{len(tiles)} đoạn ảnh (có chồng lề) · {roi.width} px tổng chiều rộng')
    previous=st.session_state[inventory_key]
    previous_results={r['tile_id']:r for r in previous.get('tile_results',[]) if r.get('tile_id')}
    coverage_before=tile_coverage(tiles,list(previous_results.values()))
    c_all,c_retry=st.columns(2)
    run_all=c_all.button('AI kiểm kê TOÀN BỘ dải',key=f'v25_inventory_all_{fingerprint}',disabled=not routes)
    run_retry=c_retry.button('Chạy lại CHỈ đoạn thiếu/lỗi',key=f'v25_inventory_retry_{fingerprint}',
                             disabled=not routes or not previous or not coverage_before['retry_ids'])
    if run_all or run_retry:
        if run_all:previous_results={}
        target_ids=set(t['tile_id'] for t in tiles) if run_all else set(coverage_before['retry_ids'])
        failures=[]
        for tile in tiles:
            if tile['tile_id'] not in target_ids:continue
            cut=roi.crop((tile['x0'],0,tile['x1'],roi.height))
            prompt=inventory_prompt(lead,cut.width,cut.height,tile)
            try:
                with st.spinner(f"AI kiểm kê {tile['tile_id']} · X {tile['x0']}–{tile['x1']}..."):
                    raw,used=analyze_with_routes(routes,prompt,
                        [(f"{lead}_{tile['tile_id']}",image_for_model(cut,max_side=max(cut.size)))])
                    rep=parse_tile(extract_json(raw),tile,roi.height)
                    rep['api_route_used']=used
                    previous_results[tile['tile_id']]=rep
            except Exception as exc:
                previous_results.pop(tile['tile_id'],None)
                failures.append({'tile_id':tile['tile_id'],'error_type':type(exc).__name__})
        reports=[previous_results[t['tile_id']] for t in tiles if t['tile_id'] in previous_results]
        inventory=merge_tile_inventory(reports)
        cov=tile_coverage(tiles,reports)
        inventory['coverage_v25']=cov
        inventory['failed_tiles']=failures
        inventory['all_tiles_processed']=cov['complete']
        st.session_state[inventory_key]=inventory
        st.session_state[selected_key]=[]
        st.session_state[local_ai_key]=[]
        st.session_state.pop(f'v242_coverage_ack_{fingerprint}',None)
        st.rerun()
    inv=st.session_state[inventory_key]
    coverage=tile_coverage(tiles,inv.get('tile_results',[]))
    st.caption(f"Đã kiểm kê hợp lệ {coverage['n_complete']}/{coverage['n_total']} đoạn; "
               'lỗi/thiếu sẽ không được coi là số QRS bằng 0.')
    st.dataframe(coverage['tiles'],hide_index=True,width='stretch')
    if inv:
        if not coverage['complete']:
            st.error('KIỂM KÊ CHƯA ĐỦ TOÀN DẢI. Chạy lại các đoạn thiếu/lỗi trước khi duyệt QRS.')
        if inv.get('issues'):
            st.warning('Có đoạn khai báo số QRS không khớp tọa độ; cần chạy lại đoạn lỗi.')
        st.info(f"AI đề xuất {inv.get('ai_count_unverified',0)} vị trí (CHƯA PHẢI số QRS xác minh).")
        if inv.get('qrs'):
            st.dataframe([{'id':q['inventory_id'],'x':q['x_pixel'],
                 'tiles':','.join(q['tile_ids']),'overlap_review':q['overlap_merge_review']}
                 for q in inv['qrs']],hide_index=True,width='stretch')
        if inv.get('uncertain_events') or inv.get('partial_events'):
            with st.expander('Sự kiện chưa phân định hoặc bị cắt ở mép ảnh'):
                st.json({'uncertain':inv.get('uncertain_events',[]),
                         'partial':inv.get('partial_events',[])})
    st.markdown('**E2 · Đối chiếu với ứng viên từ nét ảnh**')
    left_col,right_col=st.columns(2)
    with left_col:
        if st.button('Tìm ứng viên QRS (chưa xác minh)',key=f'v241_candidates_{fingerprint}'):
            st.session_state[raw_key]=image_r_candidates(roi)
            st.session_state[selected_key]=[]
            st.session_state[local_ai_key]=[]
            st.session_state.pop(f'v242_coverage_ack_{fingerprint}',None)
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
            st.session_state.pop(f'v242_coverage_ack_{fingerprint}',None)
            st.rerun()

    candidates=propose_qrs_candidates(st.session_state[raw_key],roi.width)
    if inv and candidates:
        comparison=reconcile(inv,candidates,tolerance_px=18)
        st.caption(f"Đối chiếu vị trí: {len(comparison['matches'])} cặp gần nhau; "
                   f"{len(comparison['ai_only'])} AI-only; {len(comparison['image_only'])} Python-only. "
                   'Không phải TP/FP/FN.')
        with st.expander('Danh sách vị trí đồng thuận và bất đồng',expanded=True):
            st.json(comparison)
    else:
        comparison={}
    if inv or candidates:
        st.image(inventory_overlay(roi,inv or {},candidates),width='stretch',
            caption='Tím = AI kiểm kê; cam = ứng viên Python. Cả hai chưa xác minh.')
    st.markdown('**E3 · Duyệt một danh sách HỢP NHẤT (AI + Python)**')
    st.caption('Đồng thuận không phải xác nhận QRS. Mỗi hàng là một VỊ TRÍ cần kiểm tra. '
               'Không có QRS nào được chọn mặc định; ngoại tâm thu sớm không bị loại theo RR.')
    union=reconcile_proposals(inv or {}, candidates, tolerance_px=18)
    selected_key_v25=f'v25_review_choices_{fingerprint}'
    signature_v25=hashlib.sha256(str([(u['review_id'],u['x_pixel']) for u in union]).encode()).hexdigest()[:12]
    selection_widget_key=f'{selected_key_v25}_{signature_v25}'
    if union:
        st.dataframe(union,hide_index=True,width='stretch')
        union_by_id={u['review_id']:u for u in union}
        choices=st.multiselect('Chọn CHỈ các vị trí có QRS thực sự nhìn thấy',
              options=[u['review_id'] for u in union],default=[],
              format_func=lambda rid:f"X={union_by_id[rid]['x_pixel']} · {union_by_id[rid]['origin']}",
              key=selection_widget_key)
        # Use the same overlay but no automatic strong-candidate selection.
        selected_pos=[union_by_id[k]['x_pixel'] for k in choices]
        st.caption('Vị trí được chọn (px ROI): '+(', '.join(str(x) for x in selected_pos) if selected_pos else 'chưa chọn'))
        count_ack=st.checkbox('Tôi đã xem toàn bộ dải II Rhythm và xác nhận từng QRS được chọn',
               key=f'v242_coverage_ack_{fingerprint}')
        gate=qa_gate(coverage,choices,union,count_ack,lead=='II-rhythm')
        if gate['blockers']:
            st.warning('Chưa đủ điều kiện tạo beat: '+', '.join(gate['blockers']))
        if st.button('Áp dụng QRS đã duyệt → tạo cửa sổ PQRST',
                     key=f'v25_apply_{fingerprint}',disabled=not gate['can_build_windows']):
            st.session_state[selected_key]=list(choices)
            st.session_state[local_ai_key]=[]
            st.session_state[f'v25_applied_signature_{fingerprint}']=signature_v25
            st.rerun()
    else:
        choices=[];gate={'can_build_windows':False,'blockers':['NO_PROPOSALS']}
        st.warning('Chưa có ứng viên. Kiểm kê AI hoặc dò nét trước, hoặc thêm mốc GUIDED.')
    applied_ids=st.session_state[selected_key]
    if st.session_state.get(f'v25_applied_signature_{fingerprint}') != signature_v25:
        applied_ids=[]
    selected_rows=[u for u in union if u['review_id'] in applied_ids]
    accepted=[{'x_pixel':u['x_pixel'],'source':'CLINICIAN_SELECTED_UNION_FOR_QA',
               'candidate_id':u['review_id'],'origin':u['origin']} for u in selected_rows]
    unresolved=[dict(u,event_status='UNRESOLVED_NOT_COUNTED_AS_BEAT')
                for u in union if u['review_id'] not in applied_ids]
    if union:
        st.caption(f"Tổng vị trí cần rà soát: {len(union)}; đã áp dụng: {len(accepted)}; "
                   f"chưa phân định: {len(unresolved)}.")
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

    export={'version':'2.5-INTEGRATED-QA','image_hash':image_hash,'layout':layout,'lead':lead,'roi_box':list(box),
        'roi_width':roi.width,'roi_height':roi.height,'grid':cal.to_dict() if cal else None,
        'ai_proposals_unverified':proposals,'human_landmarks':points,'measurements_human_landmarks':measured,
        'clinical_status':'RESEARCH_NOT_CLINICALLY_VALIDATED',
        'qrs_inventory_v242':inv,'qrs_reconciliation_v242':comparison,
        'inventory_coverage_v25':coverage,'unified_review_v25':union,
        'beat_v241':{'candidates_unverified':candidates,'selected_qrs_for_review':accepted,
                    'unresolved_candidates_not_beats':unresolved,'full_beat_windows':windows,
                    'ai_predictions_unverified':all_predictions,
                    'comparison_2d_to_human':comparisons,'independent_test_eligible':False}}
    with io.BytesIO() as buf:
        with zipfile.ZipFile(buf,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('measurement_evidence_v23.json',json.dumps(export,ensure_ascii=False,indent=2))
            z.writestr('README.txt','ECG 2.5 QA: AI QRS Inventory proposals and image candidates are UNVERIFIED. Unselected candidates are UNRESOLVED, not beats. AI landmarks are unverified. Measurements require clinician-confirmed pixel landmarks and a manually calibrated local grid. No raw ECG image included.\n')
        data=buf.getvalue()
    st.download_button('Xuất ZIP Measurement QA v2.5 (không chứa ảnh)',data=data,file_name=f'ecg_core_v2_3_grid_evidence_{lead.replace("-","_")}.zip',mime='application/zip',key=f'v23_export_{fingerprint}')
