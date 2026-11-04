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
        st.subheader('v1.5.2 · Đánh dấu đỉnh R tham chiếu độc lập (QA)')
        st.caption('Chấm xanh lá = vị trí bác sĩ đặt; chấm tím = gợi ý ảnh độc lập (không phải R xác minh). Không ép RR đều. Nhãn ARTIFACT đã lưu không bị thay đổi.')
        hint_report = independent_rpeak_proposals(roi)
        reference_key = 'v152_reference_' + tag + '_' + selected + '_' + hashlib.sha256(str(box).encode()).hexdigest()[:8]
        if reference_key not in st.session_state:
            st.session_state[reference_key] = []
        references = st.session_state[reference_key]
        can_click = False
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates
            can_click = True
        except ImportError:
            st.info('Chưa có streamlit-image-coordinates. Vẫn có thể nhập x/y; thêm dependency theo PATCH để bấm trực tiếp trên ảnh.')
        img_click = mark_overlay(roi, reviewed['events'], references, hint_report['hints'])
        if can_click:
            coords = streamlit_image_coordinates(img_click, width=roi.width,
                        key='v152_click_'+tag+selected+hashlib.sha256(str(box).encode()).hexdigest()[:8])
            if coords and 'x' in coords and 'y' in coords:
                st.caption(f"Điểm vừa bấm: x={coords['x']} px, y={coords['y']} px. Nhấn 'Thêm điểm R' để lưu.")
                clicked_x, clicked_y = float(coords['x']), float(coords['y'])
            else:
                clicked_x, clicked_y = None, None
        else:
            st.image(img_click, caption='Overlay nhãn và gợi ý ảnh', width='stretch')
            clicked_x, clicked_y = None, None
        cx,cy = st.columns(2)
        with cx:
            ref_x=st.number_input('Đỉnh R tham chiếu x (pixel trong ROI)', 0., float(roi.width-1),
                                  value=float(clicked_x) if clicked_x is not None else 0., step=1.,
                                  key='v152_manual_x_'+tag+selected)
        with cy:
            ref_y=st.number_input('Đỉnh R tham chiếu y (pixel trong ROI)', 0., float(roi.height-1),
                                  value=float(clicked_y) if clicked_y is not None else float(roi.height/2), step=1.,
                                  key='v152_manual_y_'+tag+selected)
        # Click coordinates are preferred when a new point is clicked; the numeric
        # fields remain a fallback and can be used when clicking is unavailable.
        mark_source=st.radio('Lấy tọa độ từ', ['Điểm bấm trên ảnh', 'Ô nhập x/y'], horizontal=True,
             index=0 if clicked_x is not None else 1, key='v152_coord_source_'+tag+selected)
        ax,bx,cx=st.columns(3)
        with ax:
            if st.button('Thêm điểm R tham chiếu', key='v152_add_'+tag+selected):
                x_add=clicked_x if mark_source=='Điểm bấm trên ảnh' and clicked_x is not None else ref_x
                y_add=clicked_y if mark_source=='Điểm bấm trên ảnh' and clicked_y is not None else ref_y
                try:
                    st.session_state[reference_key]=add_reference_point(references,x_add,y_add,
                                                  width=roi.width,height=roi.height)
                    st.rerun()
                except ValueError as exc:
                    st.warning(str(exc))
        with bx:
            if st.button('Hoàn tác điểm cuối', disabled=not bool(references), key='v152_undo_'+tag+selected):
                st.session_state[reference_key] = references[:-1]
                st.rerun()
        with cx:
            if st.button('Xóa điểm tham chiếu', disabled=not bool(references), key='v152_clear_'+tag+selected):
                st.session_state[reference_key] = []
                st.rerun()
        references = st.session_state[reference_key]
        with st.expander('Danh sách điểm R bác sĩ đã đặt', expanded=bool(references)):
            st.dataframe(references, hide_index=True, width='stretch')
        st.caption(f"{len(hint_report['hints'])} gợi ý ảnh độc lập (chưa xác minh); {len(references)} điểm R bác sĩ đã đánh dấu.")
        localized = event_rpeak_localization(roi, reviewed['events'])
        with st.expander('So sánh x của sự kiện và gợi ý đỉnh R tại chỗ', expanded=False):
            st.dataframe(localized, hide_index=True, width='stretch')
        completed_reference = st.checkbox('Tôi đã đánh dấu ĐẦY ĐỦ mọi đỉnh R thấy được trong ROI này',
                                           value=False,key='v152_complete_'+tag+selected)
        tolerance = st.slider('Dung sai so khớp vị trí (pixel; QA)', 3, 30, 10,
                              key='v152_tolerance_'+tag+selected)
        comparison = match_references(reviewed['events'], references,
                                      tolerance_px=tolerance,review_complete=completed_reference)
        st.json({'counts':comparison['counts'],
                 'reference_complete':comparison['reference_complete'],
                 'performance_metrics':comparison['performance_metrics'],
                 'artifact_label_conflicts':comparison['artifact_label_conflicts']})
        if not comparison['exhaustive_scoring_allowed']:
            st.warning('Chưa đủ nhãn tham chiếu để tính sensitivity/precision. Gợi ý ảnh và ứng viên máy vẫn chỉ là CANDIDATE.')
        else:
            st.info('Các chỉ số hiện tại chỉ phản ánh khớp tọa độ theo pixel trên ROI đã đánh dấu đầy đủ; không phải xác minh QRS lâm sàng.')
        mark_img = mark_overlay(roi, reviewed['events'], references, hint_report['hints'])
        reference_payload = {'version':'ecg-core-v1.5.2', 'image_name':upload.name,
             'image_sha256_12':tag, 'lead':selected,'layout':layout,'roi_box':box,
             'reference_points':references, 'reference_complete':completed_reference,
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
            z.writestr('event_rpeak_localization_v152.json', json.dumps(localized, ensure_ascii=False, indent=2))
            z.writestr('rpeak_reference_v152.json', json.dumps(reference_payload, ensure_ascii=False, indent=2))
            z.writestr('rpeak_reference_overlay_v152.png', _png(mark_img))
        st.download_button('Tải validation v1.5.2 (ZIP)', buf.getvalue(),
                           'ecg_core_v1_5_2_rpeak_validation.zip',
                           'application/zip', key='beat15_download_' + tag)
