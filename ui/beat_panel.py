"""Candidate-only Streamlit panel for ECG Core v1.5.

The legacy clinical reader remains unchanged; this is a research QA module.
"""
from __future__ import annotations
import hashlib, io, json, zipfile
from PIL import Image, ImageOps
import streamlit as st
from core.multi_layout import propose_boxes, overlay as layout_overlay
from core.beat_candidates import detect_candidates, evidence_panel, public_report


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
        payload = public_report(result)
        payload.update({'lead': selected, 'roi_box': box, 'layout_confirmed_by_user': layout,
                        'image_name': upload.name, 'review_status': 'UNVERIFIED'})
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('beat_candidates_v15.json', json.dumps(payload, ensure_ascii=False, indent=2))
            z.writestr('beat_candidates_overlay_v15.png', _png(panel))
            z.writestr('candidate_ink_mask_v15.png', _png(ink))
            z.writestr('roi_original_v15.png', _png(roi))
        st.download_button('Tải validation v1.5 (ZIP)', buf.getvalue(),
                           'ecg_core_v1_5_beat_candidate_validation.zip',
                           'application/zip', key='beat15_download_' + tag)
