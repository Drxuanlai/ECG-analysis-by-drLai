"""Candidate-only Streamlit panel for ECG Core v1.5.

The legacy clinical reader remains unchanged; this is a research QA module.
"""
from __future__ import annotations
import hashlib, io, json, zipfile
from PIL import Image, ImageOps
import streamlit as st
from core.multi_layout import propose_boxes, overlay as layout_overlay
from core.beat_candidates import detect_candidates, evidence_panel, public_report
from core.beat_validation import cluster_validate, attach_review, validation_overlay, REVIEW_LABELS
from core.rpeak_review import (independent_rpeak_proposals, add_reference_point,
                               match_references, mark_overlay, event_rpeak_localization)
from core.rpeak_filter import (filter_rpeak_hints, filter_overlay, compare_with_references)
from core.rpeak_filter_v155 import rank_rpeak_hints
from core.rpeak_windows import (make_windows, local_to_global, global_to_local_markers,
                                review_summary, VERSION as WINDOW_VERSION)


def _load(raw: bytes):
    try:
        from pillow_heif import register_heif_opener
        register_heif_opener()
    except ImportError:
        pass
    im = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert('RGB')
    if max(im.size) > 3000:
        im.thumbnail((3000, 3000), Image.Resampling.LANCZOS)
    return im


def _png(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def render():
    with st.expander('ECG Core v1.5 | Beat-by-Beat Candidate Review', expanded=False):
        st.caption('Công cụ nghiên cứu: mỗi nét đứng là ỨNG VIÊN, không phải QRS xác minh. Không tự tính nhịp tim hoặc PQRST.')
        upload = st.file_uploader('Ảnh ECG (PNG/JPG/HEIC)',
                                  type=['png', 'jpg', 'jpeg', 'webp', 'heic', 'heif'],
                                  key='beat15_upload')
        if upload is None:
            st.info('Ảnh cần được xác nhận bố cục và ROI trước khi xem ứng viên QRS.')
            return
        raw = upload.getvalue()
        tag = hashlib.sha256(raw).hexdigest()[:12]
        try:
            im = _load(raw)
        except Exception as exc:
            st.error('Không đọc được ảnh: ' + str(exc))
            return
        w, h = im.size
        layout = st.selectbox('Bố cục đã kiểm tra trên ECG gốc',
                              ['Chưa xác nhận', '3x4', '6x2'], key='beat15_layout_' + tag)
        if layout == 'Chưa xác nhận':
            st.warning('Hãy xác nhận bố cục trước. Không tự suy ra 3×4 từ ảnh 6×2.')
            return
        rhythm = st.checkbox('Có dải nhịp II dài riêng', value=True,
                             key='beat15_rhythm_' + tag) if layout == '3x4' else False
        try:
            proposal = propose_boxes(im, layout, has_rhythm=rhythm)
        except Exception as exc:
            st.error('Không tạo được ROI: ' + str(exc))
            return
        st.image(layout_overlay(im, proposal), caption='Bố cục đề xuất; xác nhận vị trí trước khi phân tích.', width='stretch')
        names = [i['lead'] for i in proposal['rois']]
        selected = st.selectbox('Chuyển đạo kiểm thử', names,
                                index=names.index('II') if 'II' in names else 0,
                                key='beat15_lead_' + tag)
        row = next(r for r in proposal['rois'] if r['lead'] == selected)
        st.caption('Có thể nhập tọa độ ROI đã chỉnh từ QA cũ bằng cách bật thủ công. Không dùng lại tọa độ của ảnh khác.')
        manual = st.checkbox('Dùng tọa độ ROI thủ công cho lead này', False,
                             key='beat15_manual_' + tag)
        box = list(row['box'])
        if manual:
            c1, c2 = st.columns(2)
            with c1:
                lr = st.slider('Trái / phải (% ảnh)', 0., 100.,
                               (round(box[0] / w * 100, 1), round(box[2] / w * 100, 1)), .1,
                               key='beat15_x_' + tag + selected)
            with c2:
                tb = st.slider('Trên / dưới (% ảnh)', 0., 100.,
                               (round(box[1] / h * 100, 1), round(box[3] / h * 100, 1)), .1,
                               key='beat15_y_' + tag + selected)
            box = [round(w * lr[0] / 100), round(h * tb[0] / 100),
                   round(w * lr[1] / 100), round(h * tb[1] / 100)]
        if box[2] - box[0] < 60 or box[3] - box[1] < 24:
            st.error('ROI quá nhỏ hoặc không hợp lệ.')
            return
        roi = im.crop(tuple(box))
        sensitivity = st.slider('Độ nhạy ứng viên (QA, không phải ngưỡng chẩn đoán)',
                                .15, .85, .55, .05, key='beat15_sens_' + tag)
        if not st.checkbox('Chạy phát hiện ứng viên trên ROI đã xem', False,
                           key='beat15_run_' + tag):
            st.image(roi, caption=f'ROI {selected} · {box}', width='stretch')
            return
        with st.spinner('Phân tích nét mực ứng viên...'):
            try:
                result = detect_candidates(roi, sensitivity=sensitivity)
            except Exception as exc:
                st.error('Không xử lý được ROI: ' + str(exc))
                return
        panel, ink = evidence_panel(roi, result)
        st.image(panel, caption='Trái: ROI gốc · Phải: ứng viên theo từng sự kiện (cam). KHÔNG PHẢI QRS đã xác minh.', width='stretch')
        with st.expander('Xem bản đồ mực thô và metadata', expanded=False):
            st.image(ink, caption='Mực ứng viên vẫn có thể chứa lưới, chữ và xung chuẩn.', width='stretch')
            st.json({'metrics': result['metrics'], 'reason_codes': result['reason_codes']})
        st.dataframe(result['candidates'], hide_index=True, width='stretch')
        st.warning('Mọi dấu cam đều cần đối chiếu ECG gốc. Không dùng số ứng viên để tính tần số tim hoặc nhận định loạn nhịp.')
        st.divider()
        st.subheader('v1.5.1 · Nhóm ứng viên & kiểm tra nhiễu (QA)')
        st.caption('Không ép RR đều. Màu xanh dương: ứng viên hình thái; màu cam: vùng chưa phân giải. Bác sĩ có thể đánh dấu QRS/Artifact và bổ sung QRS bị bỏ sót.')
        # Persist case identity in the event report (v1.5.1 previously lost it).
        result.update({'lead': selected, 'image_name': upload.name, 'roi_box': box})
        grouped = cluster_validate(roi, result)
        st.write(f"{grouped['source_candidate_count']} nét ứng viên → "
                 f"{grouped['event_count']} sự kiện, "
                 f"{grouped['machine_status_counts']['UNRESOLVED']} vùng UNRESOLVED")
        # A reviewer label is NOT inferred from the machine's heuristic score.
        labels = {}
        with st.expander('Gán nhãn từng sự kiện (tùy chọn)', expanded=False):
            st.caption('QRS = bác sĩ xác nhận từ ảnh; ARTIFACT = nhiễu; UNCERTAIN = chưa thể phân định. Mặc định UNREVIEWED.')
            for e in grouped['events']:
                c1, c2, c3 = st.columns([1, 3, 2])
                with c1:
                    st.write(e['event_id'])
                with c2:
                    reasons = ', '.join(e['reason_codes']) if e['reason_codes'] else 'no_specific_flag'
                    st.caption(f"x={e['x_pixel']:.1f}px · {e['machine_status']} · {reasons}")
                with c3:
                    labels[e['event_id']] = st.selectbox(
                        'Nhãn người đọc', REVIEW_LABELS,
                        key='beat151_label_' + tag + selected + '_' + e['event_id'],
                        label_visibility='collapsed')
        missed_text = st.text_input(
            'QRS bị bỏ sót: tọa độ x (pixel tính từ trái ROI), cách nhau bằng dấu phẩy',
            value='', key='beat151_missed_' + tag + selected,
            help='Chỉ ghi sau khi đã kiểm tra ảnh ECG gốc. Để trống nếu chưa rà soát đầy đủ.')
        missed = []
        if missed_text.strip():
            try:
                missed = [float(t.strip()) for t in missed_text.split(',') if t.strip()]
                if any(not 0 <= x < roi.width for x in missed):
                    raise ValueError('Tọa độ nằm ngoài ROI')
            except ValueError as exc:
                st.error('Tọa độ QRS bỏ sót chưa hợp lệ: ' + str(exc))
                return
        reviewed = attach_review(grouped, labels, missed)
        st.image(validation_overlay(roi, reviewed),
                 caption='Đường màu: sự kiện ứng viên; chấm tím dưới: QRS bác sĩ bổ sung. Không phải ECG đã đo.',
                 width='stretch')
        with st.expander('Bảng sự kiện và lý do QC', expanded=False):
            st.dataframe(reviewed['events'], hide_index=True, width='stretch')
        st.warning('Máy không xác minh QRS. Những sự kiện UNRESOLVED không được tự loại bỏ vì có thể là nhịp bệnh lý thực.')
        st.divider()
        st.subheader('v1.5.3 · Đánh dấu đỉnh R theo từng đoạn (QA)')
        st.caption('Chấm xanh lá = vị trí bác sĩ đặt; chấm tím = gợi ý ảnh độc lập (không phải R xác minh). Không ép RR đều. Nhãn ARTIFACT đã lưu không bị thay đổi.')
        hint_report = independent_rpeak_proposals(roi)
        st.caption('v1.5.5: ưu tiên cực trị và nét mực tại đỉnh; không dùng RR đều hoặc nhãn bác sĩ để xếp hạng.')
        fraction = st.slider('Tỷ lệ gợi ý đưa vào nhóm xem trước (QA)',
                             min_value=.20, max_value=.80, value=.35, step=.05,
                             key='v154_fraction_'+tag+selected)
        filter_report = rank_rpeak_hints(roi, hint_report['hints'],
                                          shortlist_fraction=fraction)
        st.caption(f"{filter_report['raw_hint_count']} gợi ý gốc → "
                   f"{filter_report['priority_count']} cần xem trước; "
                   f"{filter_report['unresolved_count']} vẫn lưu ở trạng thái UNRESOLVED. "
                   'Không coi gợi ý ưu tiên là QRS đã xác nhận.')
        reference_key = 'v152_reference_' + tag + '_' + selected + '_' + hashlib.sha256(str(box).encode()).hexdigest()[:8]
        if reference_key not in st.session_state:
            st.session_state[reference_key] = []
        references = st.session_state[reference_key]
        with st.expander('Khôi phục điểm R đã lưu từ bản v1.5.2 hoặc v1.5.3',expanded=False):
            st.caption('Tải ZIP validation hoặc JSON của cùng ảnh, lead và ROI. Điểm cũ sẽ không bị tự ghi đè.')
            restore_file=st.file_uploader('Tệp validation R-peak (ZIP/JSON)',
                                          type=['zip','json'],key=reference_key+'_restore_file')
            if restore_file is not None and st.button('Khôi phục điểm R từ file',
                                                    key=reference_key+'_restore_button'):
                try:
                    data=restore_file.getvalue()
                    if restore_file.name.lower().endswith('.zip'):
                        with zipfile.ZipFile(io.BytesIO(data)) as zz:
                            found=next((n for n in ['rpeak_reference_v153.json','rpeak_reference_v152.json']
                                        if n in zz.namelist()),None)
                            if not found: raise ValueError('Không tìm thấy file R-peak reference trong ZIP')
                            payload_restore=json.loads(zz.read(found))
                    else:
                        payload_restore=json.loads(data)
                    if (payload_restore.get('image_sha256_12') != tag or
                        payload_restore.get('lead') != selected or
                        list(payload_restore.get('roi_box',[])) != box):
                        raise ValueError('Ảnh / chuyển đạo / ROI không khớp; không thể nạp nhãn cũ.')
                    imported=[]
                    for q in payload_restore.get('reference_points',[]):
                        imported=add_reference_point(imported,q['x_pixel'],q['y_pixel'],
                                                     width=roi.width,height=roi.height)
                    if references:
                        raise ValueError('Đang có điểm R trong phiên này; hãy dùng vùng hiện tại để tránh ghi đè.')
                    st.session_state[reference_key]=imported
                    st.session_state[reference_key+'_reviewed_windows']={}
                    st.session_state[reference_key+'_confirm_complete']=False
                    st.success(f'Đã khôi phục {len(imported)} điểm. Cần rà soát lại từng đoạn trước khi khóa bộ tham chiếu.')
                    st.rerun()
                except (ValueError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
                    st.error('Không thể khôi phục: '+str(exc))
        can_click = False
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
            can_click = True
        except ImportError:
            st.info('Chưa có streamlit-image-coordinates. Vẫn có thể nhập x/y; thêm dependency theo PATCH để bấm trực tiếp trên ảnh.')
        # v1.5.3 - annotation happens in short, non-overlapping windows.
        # All marks are stored in full ORIGINAL-ROI coordinates, never display pixels.
        windows = make_windows(roi.width, target_width=640)
        review_key = reference_key + '_reviewed_windows'
        select_key = reference_key + '_selected_window'
        confirm_key = reference_key + '_confirm_complete'
        if review_key not in st.session_state:
            st.session_state[review_key] = {}
        if select_key not in st.session_state:
            st.session_state[select_key] = 0
        win_idx = max(0, min(int(st.session_state[select_key]), len(windows)-1))
        st.session_state[select_key] = win_idx
        state = review_summary(windows, st.session_state[review_key], references)
        st.progress(state['reviewed_count'] / state['total_count'],
                    text=f"Đã rà soát {state['reviewed_count']}/{state['total_count']} đoạn · "
                         f"{len(references)} điểm R đã lưu")
        nav_prev, nav_sel, nav_next = st.columns([1,3,1])
        with nav_prev:
            st.button('◀ Đoạn trước', disabled=win_idx==0, key=reference_key+'_prev',
                      on_click=lambda: st.session_state.update({select_key:win_idx-1}))
        with nav_sel:
            st.selectbox('Chọn đoạn cần đánh dấu', options=list(range(len(windows))),
                         format_func=lambda i: (f"Đoạn {i+1}/{len(windows)} · "
                             f"x={windows[i]['x_start']}–{windows[i]['x_end']-1} px "
                             + ('✓ Đã xem' if st.session_state[review_key].get(str(i),False) else '• Chưa rà soát')),
                         key=select_key)
        with nav_next:
            st.button('Đoạn sau ▶', disabled=win_idx>=len(windows)-1,
                      key=reference_key+'_next',
                      on_click=lambda: st.session_state.update({select_key:win_idx+1}))
        win_idx = int(st.session_state[select_key])
        window = windows[win_idx]
        window_roi = roi.crop((window['x_start'], 0, window['x_end'], roi.height))
        window_events = global_to_local_markers(reviewed['events'], window)
        window_refs = global_to_local_markers(references, window)
        window_hints = global_to_local_markers(hint_report['hints'], window)
        img_click = mark_overlay(window_roi, window_events, window_refs, window_hints)
        display_width = min(850, window['width'])
        if can_click:
            coords = streamlit_image_coordinates(img_click, width=display_width,
                key=reference_key+'_click_window_'+str(win_idx))
            if coords and 'x' in coords and 'y' in coords:
                try:
                    clicked_x,clicked_y = local_to_global(float(coords['x']), float(coords['y']),
                        window, display_width, roi.height)
                    st.caption(f"Điểm vừa bấm: X toàn ROI={clicked_x:.1f}, Y={clicked_y:.1f} px. "
                               "Nhấn 'Thêm điểm R tham chiếu' để lưu.")
                except ValueError:
                    clicked_x,clicked_y = None,None
            else:
                clicked_x,clicked_y = None,None
        else:
            st.image(img_click, caption='Đoạn ECG hiện tại; chưa cài thư viện nhận tọa độ.',
                     width='stretch')
            clicked_x,clicked_y = None,None
        st.caption(f"Đoạn {win_idx+1}: x toàn ROI {window['x_start']}–{window['x_end']-1} px. "
                   "Đường cam là gợi ý máy; dấu xanh là R do bác sĩ đánh dấu.")
        cx,cy = st.columns(2)
        with cx:
            ref_x=st.number_input('X của đỉnh R trong toàn ROI (pixel)',
                min_value=float(window['x_start']), max_value=float(window['x_end']-1),
                value=float(window['x_start']), step=1.,
                key=reference_key+'_manual_x_w'+str(win_idx))
        with cy:
            ref_y=st.number_input('Y của đỉnh R (pixel)', 0., float(roi.height-1),
                value=float(roi.height/2), step=1.,
                key=reference_key+'_manual_y_w'+str(win_idx))
        mark_source=st.radio('Lấy tọa độ từ',
            ['Điểm bấm trên ảnh', 'Ô nhập x/y'],horizontal=True,
            index=0 if can_click else 1,key=reference_key+'_coord_source_w'+str(win_idx))
        ax,bx,cx = st.columns(3)
        with ax:
            if st.button('Thêm điểm R tham chiếu',key=reference_key+'_add_w'+str(win_idx)):
                if mark_source=='Điểm bấm trên ảnh' and clicked_x is None:
                    st.warning('Hãy bấm vào đỉnh R trong đoạn hiện tại trước khi nhấn Thêm.')
                else:
                    x_add=clicked_x if mark_source=='Điểm bấm trên ảnh' else ref_x
                    y_add=clicked_y if mark_source=='Điểm bấm trên ảnh' else ref_y
                    try:
                        st.session_state[reference_key]=add_reference_point(references,x_add,y_add,
                                                    width=roi.width,height=roi.height)
                        # Any edit makes current window require a fresh review confirmation.
                        st.session_state[review_key][str(win_idx)]=False
                        st.session_state[confirm_key]=False
                        st.rerun()
                    except ValueError as exc:
                        st.warning(str(exc))
        with bx:
            if st.button('Hoàn tác điểm cuối',disabled=not bool(references),
                         key=reference_key+'_undo'):
                last = references[-1]
                affected = next((w['index'] for w in windows
                                 if w['x_start']<=last['x_pixel']<w['x_end']),win_idx)
                st.session_state[reference_key]=references[:-1]
                st.session_state[review_key][str(affected)]=False
                st.session_state[confirm_key]=False
                st.rerun()
        with cx:
            if st.button('Xóa điểm trong đoạn này',
                         disabled=not bool(window_refs),key=reference_key+'_clear_w'+str(win_idx)):
                st.session_state[reference_key]=[p for p in references
                    if not window['x_start']<=p['x_pixel']<window['x_end']]
                # Renumber reference IDs after deletion, retaining coordinates.
                for i,p in enumerate(st.session_state[reference_key],1):
                    p['reference_id']=f'R{i:03d}'
                st.session_state[review_key][str(win_idx)]=False
                st.session_state[confirm_key]=False
                st.rerun()
        reviewed_current = bool(st.session_state[review_key].get(str(win_idx),False))
        if st.button(('Mở lại đoạn này để rà soát' if reviewed_current else
                      'Xác nhận đã rà soát HẾT đoạn này'),
                     key=reference_key+'_mark_window_'+str(win_idx)):
            st.session_state[review_key][str(win_idx)] = not reviewed_current
            if reviewed_current:
                st.session_state[confirm_key]=False
            st.rerun()
        references = st.session_state[reference_key]
        with st.expander('Danh sách điểm R bác sĩ đã đặt', expanded=bool(references)):
            st.dataframe(references, hide_index=True, width='stretch')
        st.caption(f"{len(hint_report['hints'])} gợi ý ảnh độc lập (chưa xác minh); {len(references)} điểm R bác sĩ đã đánh dấu.")
        with st.expander('v1.5.5 · So sánh gợi ý tím và thứ tự ưu tiên (QA)', expanded=False):
            st.image(filter_overlay(roi, filter_report, references),
                     caption='Xanh lam = ưu tiên xem; tím = chưa giải quyết; xanh lá = R bác sĩ. KHÔNG phải QRS xác minh.',
                     width='stretch')
            st.dataframe(filter_report['hints'],hide_index=True,width='stretch')
            st.warning('Cả nhóm ưu tiên và nhóm UNRESOLVED đều có thể chứa QRS thực. Không tự xóa ứng viên.')
        localized = event_rpeak_localization(roi, reviewed['events'])
        with st.expander('So sánh x của sự kiện và gợi ý đỉnh R tại chỗ', expanded=False):
            st.dataframe(localized, hide_index=True, width='stretch')
        state=review_summary(windows, st.session_state[review_key], references)
        st.dataframe(state['windows'],hide_index=True,width='stretch')
        if not state['all_windows_reviewed']:
            st.warning(f"Còn {state['total_count']-state['reviewed_count']} đoạn chưa rà soát. "
                       "Không thể xác nhận bộ R đầy đủ hoặc tính sensitivity/precision.")
            st.session_state[confirm_key]=False
        completed_reference = st.checkbox(
            'Tôi xác nhận đã đánh dấu ĐẦY ĐỦ các đỉnh R trên TOÀN BỘ ROI',
            value=False,disabled=not state['all_windows_reviewed'],key=confirm_key)
        completed_reference = bool(completed_reference and state['all_windows_reviewed'])
        tolerance = st.slider('Dung sai so khớp vị trí (pixel; QA)', 3, 30, 10,
                              key='v152_tolerance_'+tag+selected)
        comparison = match_references(reviewed['events'], references,
                                      tolerance_px=tolerance,review_complete=completed_reference)
        filtered_comparison = compare_with_references(filter_report, references,
                    tolerance_px=tolerance, reference_complete=completed_reference)
        with st.expander('v1.5.5 · Đối chiếu danh sách ưu tiên với R bác sĩ',expanded=False):
            st.json({'raw_hints':filter_report['raw_hint_count'],
                'priority_review':filter_report['priority_count'],
                'references':len(references),
                'matched_x':filtered_comparison['spatially_matched_count'],
                'metrics':filtered_comparison['metrics'],
                'unmatched_references':filtered_comparison['unmatched_reference_ids']})
            if not completed_reference:
                st.caption('Chưa xác nhận đầy đủ toàn ROI: chỉ được đối chiếu vị trí, KHÔNG tính sensitivity/precision.')
        st.json({'counts':comparison['counts'],
                 'reference_complete':comparison['reference_complete'],
                 'performance_metrics':comparison['performance_metrics'],
                 'artifact_label_conflicts':comparison['artifact_label_conflicts']})
        if not comparison['exhaustive_scoring_allowed']:
            st.warning('Chưa đủ nhãn tham chiếu để tính sensitivity/precision. Gợi ý ảnh và ứng viên máy vẫn chỉ là CANDIDATE.')
        else:
            st.info('Các chỉ số hiện tại chỉ phản ánh khớp tọa độ theo pixel trên ROI đã đánh dấu đầy đủ; không phải xác minh QRS lâm sàng.')
        mark_img = mark_overlay(roi, reviewed['events'], references, hint_report['hints'])
        reference_payload = {'version':'ecg-core-v1.5.3', 'image_name':upload.name,
             'image_sha256_12':tag, 'lead':selected,'layout':layout,'roi_box':box,
             'reference_points':references, 'reference_complete':completed_reference,
             'annotation_windows':state, 'window_annotation_version':WINDOW_VERSION,
             'reference_source':'HUMAN_DIRECT_MARK',
             'candidate_event_labels':[{ 'event_id':e['event_id'], 'label':e['review_label']} for e in reviewed['events']],
             'comparison':comparison, 'note':'Do not convert px to ms without verified time calibration.'}
        payload = public_report(result)
        payload.update({'lead': selected, 'roi_box': box, 'layout_confirmed_by_user': layout,
                        'image_name': upload.name, 'review_status': 'UNVERIFIED'})
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('beat_candidates_v15.json', json.dumps(payload, ensure_ascii=False, indent=2))
            z.writestr('beat_candidates_overlay_v15.png', _png(panel))
            z.writestr('candidate_ink_mask_v15.png', _png(ink))
            z.writestr('roi_original_v15.png', _png(roi))
            z.writestr('beat_events_v151.json', json.dumps(reviewed, ensure_ascii=False, indent=2))
            z.writestr('beat_events_overlay_v151.png', _png(validation_overlay(roi, reviewed)))
            z.writestr('rpeak_image_hints_v152.json', json.dumps(hint_report, ensure_ascii=False, indent=2))
            old_filter_report=filter_rpeak_hints(roi,hint_report['hints'],shortlist_fraction=fraction)
            old_comp=compare_with_references(old_filter_report,references,tolerance_px=tolerance,reference_complete=completed_reference)
            z.writestr('rpeak_filtered_v154.json', json.dumps(old_filter_report, ensure_ascii=False, indent=2))
            z.writestr('rpeak_filtered_comparison_v154.json', json.dumps(old_comp, ensure_ascii=False, indent=2))
            z.writestr('rpeak_filtered_overlay_v154.png',_png(filter_overlay(roi,old_filter_report,references)))
            z.writestr('rpeak_ranked_v155.json',json.dumps(filter_report,ensure_ascii=False,indent=2))
            z.writestr('rpeak_ranked_comparison_v155.json',json.dumps(filtered_comparison,ensure_ascii=False,indent=2))
            z.writestr('rpeak_ranked_overlay_v155.png',_png(filter_overlay(roi,filter_report,references)))
            z.writestr('event_rpeak_localization_v152.json', json.dumps(localized, ensure_ascii=False, indent=2))
            z.writestr('rpeak_reference_v152.json', json.dumps(reference_payload, ensure_ascii=False, indent=2))
            z.writestr('rpeak_reference_v153.json', json.dumps(reference_payload, ensure_ascii=False, indent=2))
            z.writestr('rpeak_reference_overlay_v152.png', _png(mark_img))
        st.download_button('Tải validation v1.5.5 (ZIP)', buf.getvalue(),
                           'ecg_core_v1_5_5_rpeak_ranking_validation.zip',
                           'application/zip', key='beat15_download_' + tag)
