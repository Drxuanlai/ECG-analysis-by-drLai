"""ECG Core v2.1 — AI Visual Reader + Adaptive Cropping. Research prototype.

Separate Streamlit entrypoint, preserving app.py, legacy Gemini/failover, Benchmark v2.0.
"""
from __future__ import annotations
import io
import json
import zipfile
from PIL import Image,ImageDraw
import streamlit as st
from core.visual_reader_v21 import (LEADS,FIRST_PASS_PROMPT,SECOND_PASS_PROMPT,
    image_from_bytes,image_identity,image_for_model,layout_boxes,crop_images,extract_json,
    sanitized_observations,normalize_second,choose_leads,reconciliation,report_export)
from core.gemini_visual_v21 import routes_from_secrets, analyze_with_routes

st.set_page_config(page_title='ECG Core v2.1 | AI Visual Evidence',layout='wide')
st.title('ECG Core v2.1 · AI Visual Evidence Lab')
st.caption('Một ảnh gốc → AI đọc 12 lead → crop thích ứng → đọc lại độc lập → đối chiếu. NGHIÊN CỨU, bác sĩ phải duyệt.')
st.warning('Bản nghiên cứu. Không dùng làm công cụ duy nhất để loại trừ nhồi máu cơ tim, rối loạn nhịp nguy hiểm hoặc quyết định điều trị. Chưa có phép đo định lượng ECG đã xác minh.')

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    heic_ready=True
except ImportError:
    heic_ready=False

with st.sidebar:
    st.subheader('Thiết lập AI')
    try:
        routes = routes_from_secrets(st.secrets)
    except Exception:
        routes = []
    st.caption(f'Đã nhận {len(routes)} route Gemini từ Secrets.' if routes else
               'Chưa nhận được ECG_API_ROUTES hoặc GEMINI_API_KEY từ Secrets của app này.')
    if routes:
        st.caption('Route ưu tiên: ' + str(routes[0]['name']) + ' · ' + str(routes[0]['model']))
    st.caption('Chuyển route khi API lỗi. Không ghi API key vào báo cáo.')
    st.caption('API key đặt trong Streamlit Secrets; không đưa vào GitHub.')
    st.caption('Các ảnh gửi đến dịch vụ AI của Google. Chỉ dùng dữ liệu đã khử định danh và có quyền xử lý.')

st.subheader('1 · Tải một ảnh ECG gốc')
upload=st.file_uploader('Ảnh 12 chuyển đạo',type=['jpg','jpeg','png','webp']+(['heic','heif'] if heic_ready else []))
if not heic_ready: st.caption('Ảnh HEIC: cần pillow-heif; nếu không có hãy chuyển sang JPG/PNG.')
if upload is None:
    st.info('Tải một ảnh ECG để bắt đầu. Chưa cần chuẩn bị 5 ảnh crop.')
    st.stop()
try:
    photo=image_from_bytes(upload.getvalue())
except Exception as exc:
    st.error(f'Không mở được ảnh: {exc}')
    st.stop()
image_hash=image_identity(photo)
if st.session_state.get('v21_image_hash')!=image_hash:
    for state_key in ('v21_first','v21_second','v21_selected','v21_completed_layout'):
        st.session_state.pop(state_key,None)
    st.session_state['v21_image_hash']=image_hash
st.image(photo,caption=f'Ảnh gốc · {photo.width}×{photo.height} pixel',width='stretch')

st.subheader('2 · Đọc toàn cảnh bằng AI (lượt 1)')
if st.button('AI đọc cấu trúc 12 chuyển đạo',type='primary'):
    try:
        with st.spinner('AI đang mô tả hình ảnh ECG gốc...'):
            raw, used_route=analyze_with_routes(routes,FIRST_PASS_PROMPT,[('ECG gốc, 12 chuyển đạo',image_for_model(photo,max_side=3200))])
            first=sanitized_observations(extract_json(raw))
        st.session_state['v21_first']=first
        st.session_state.pop('v21_second',None)
        st.success(f'Đã nhận mô tả AI lượt 1 qua {used_route}. Đây là giả thuyết, chưa kiểm chứng định lượng.')
    except Exception as exc:
        st.error(f'Không đọc được AI lượt 1: {exc}')
first=st.session_state.get('v21_first')
if not first: st.stop()

st.markdown('**Nhận xét toàn cảnh**')
st.write(first['rhythm_observation'])
st.caption('Hiệu chuẩn: '+first['calibration'])
if first['image_limitations']: st.caption('Giới hạn: '+first['image_limitations'])
with st.expander('Checklist đầy đủ 12 chuyển đạo (AI, chưa xác minh)',expanded=False):
    for lead in LEADS:
        x=first['lead_observations'][lead]
        st.markdown(f"**{lead}** · {x['visibility']} — {x['observation']}")
        st.caption(f"P: {x['p']} | QRS: {x['qrs']} | ST–T: {x['st_t']}")
if first['hypotheses']:
    st.markdown('**Giả thuyết cần kiểm tra**')
    for f in first['hypotheses']:
        st.write(f"{f['type']} · {', '.join(f['leads']) or 'Chưa rõ lead'} — {f['evidence']}")
        if f['uncertainty']: st.caption(f"Chưa chắc: {f['uncertainty']}")
else:
    st.info('AI chưa đưa ra giả thuyết. Vẫn phải rà soát các lead cơ bản; không suy ra ECG bình thường.')

st.subheader('3 · Bố cục và crop thích ứng (bác sĩ xác nhận)')
col1,col2,col3=st.columns(3)
with col1: layout=st.selectbox('Bố cục trên ảnh gốc',['3x4','6x2'])
with col2: rhythm=st.checkbox('Có dải II kéo dài riêng',value=(layout=='3x4'))
with col3: margin=st.slider('Lề tránh nhiễm lead liền kề',0.0,.07,.02,.005)
if rhythm:
    rhythm_start=st.slider('Ranh giới bắt đầu dải nhịp (%)',65,90,78)/100
else: rhythm_start=.78
try:
    boxes=layout_boxes(photo.size,layout,rhythm=rhythm,rhythm_start=rhythm_start,margin=margin)
except ValueError as exc:
    st.error(str(exc)); st.stop()
# Per-lead ROI overrides are bound to the source image and selected layout.
override_identity=f'{image_hash}:{layout}:{rhythm}'
if st.session_state.get('v21_override_identity')!=override_identity:
    st.session_state['v21_override_identity']=override_identity
    st.session_state['v21_roi_overrides']={}
for lead, box in st.session_state['v21_roi_overrides'].items():
    if lead in boxes: boxes[lead]=tuple(box)
with st.expander('Sửa ROI riêng khi S âm sâu / QRS cao bị cắt (tùy chọn)'):
    edit_lead=st.selectbox('Chuyển đạo muốn chỉnh',list(boxes),key='v21_edit_lead')
    bx=boxes[edit_lead]
    st.caption(f'Khung hiện tại của {edit_lead}: {bx}. Có thể mở rộng Y khi thấy đỉnh/đáy QRS bị cắt.')
    ed_cols=st.columns(4)
    x0=ed_cols[0].number_input('X trái',0,photo.width-2,value=int(bx[0]),step=5,key=f'v21_x0_{override_identity}_{edit_lead}')
    y0=ed_cols[1].number_input('Y trên',0,photo.height-2,value=int(bx[1]),step=5,key=f'v21_y0_{override_identity}_{edit_lead}')
    x1=ed_cols[2].number_input('X phải',1,photo.width,value=int(bx[2]),step=5,key=f'v21_x1_{override_identity}_{edit_lead}')
    y1=ed_cols[3].number_input('Y dưới',1,photo.height,value=int(bx[3]),step=5,key=f'v21_y1_{override_identity}_{edit_lead}')
    if st.button('Áp dụng ROI cho chuyển đạo này'):
        if x0>=x1 or y0>=y1:
            st.error('Khung phải có chiều rộng và chiều cao dương.')
        else:
            st.session_state['v21_roi_overrides'][edit_lead]=(int(x0),int(y0),int(x1),int(y1))
            st.session_state.pop('v21_second',None)
            st.rerun()
    if st.button('Khôi phục tất cả ROI theo bố cục'):
        st.session_state['v21_roi_overrides']={}
        st.session_state.pop('v21_second',None)
        for coord in ('x0','y0','x1','y1'):
            st.session_state.pop(f'v21_{coord}_{override_identity}_{edit_lead}',None)
        st.rerun()
preview=photo.copy();draw=ImageDraw.Draw(preview)
for lead,(x0,y0,x1,y1) in boxes.items():
    color='#9C27B0' if lead=='II-rhythm' else '#04A5D6'
    draw.rectangle((x0,y0,x1,y1),outline=color,width=max(2,photo.width//750))
    draw.text((x0+6,y0+5),lead,fill=color,stroke_width=1,stroke_fill='#ffffff')
st.image(preview,caption='Khung ROI chỉ là đề xuất hình học. Nếu sai nhãn, cắt QRS hoặc chồng lead, KHÔNG dùng crop để kết luận.',width='stretch')

available=set(boxes)
proposed=choose_leads(first,available,max_leads=6)
selected=st.multiselect('Lead gửi đọc kỹ (tối đa 6 ảnh, có checklist cố định)',options=list(boxes),default=proposed)
if len(selected)>6:
    st.error('Chỉ chọn tối đa 6 ảnh trong một lượt; phân tích thêm ở lượt khác.')
    st.stop()
if not selected:
    st.warning('Chọn ít nhất một chuyển đạo trước khi đọc lại.');st.stop()
with st.expander('Xem các ảnh crop thực tế trước khi gửi AI',expanded=True):
    cropped=crop_images(photo,boxes,selected)
    columns=st.columns(3)
    for i,(lead,buf) in enumerate(cropped):
        with columns[i%3]:
            st.image(buf,caption=f'{lead} · {boxes[lead]}',width='stretch')
    st.warning('Nếu crop bị lệch nhãn, mất đỉnh R/S hoặc chứa lead khác: dừng tại đây, chỉnh ROI ở công cụ layout v1 trước khi đọc lại.')

confirmed=st.checkbox('Tôi đã kiểm tra: bố cục, nhãn lead và các vùng crop phù hợp để AI xem lại',value=False)
if st.button('AI xem lại các lead đã chọn (lượt 2)',disabled=not confirmed):
    try:
        with st.spinner('Đang đọc lại các ROI theo bằng chứng hình ảnh...'):
            # No initial AI hypotheses supplied: limits confirmation bias.
            raw, used_route=analyze_with_routes(routes,SECOND_PASS_PROMPT,cropped)
            second=normalize_second(extract_json(raw),selected)
        st.session_state['v21_second']=second
        st.session_state['v21_selected']=list(selected)
        st.session_state['v21_completed_layout']={'layout':layout,'rhythm':rhythm,'boxes':boxes}
        st.success(f'Đã có lượt đọc lại qua {used_route}. Đây vẫn là AI quan sát hình ảnh, không phải đo độc lập.')
    except Exception as exc:
        st.error(f'Không đọc được AI lượt 2: {exc}')

second=st.session_state.get('v21_second')
if second:
    st.subheader('4 · Đối chiếu hai lượt đọc')
    prev=st.session_state.get('v21_completed_layout',{})
    if prev.get('boxes')!=boxes or st.session_state.get('v21_selected')!=list(selected):
        st.warning('ROI/lựa chọn lead đã đổi sau khi chạy lượt 2. Kết quả bên dưới thuộc cấu hình cũ; cần chạy lại lượt 2.')
        st.stop()
    rows=reconciliation(first,second,selected)
    st.dataframe(rows,width='stretch',hide_index=True)
    st.caption('Hai lượt AI đồng ý nhau KHÔNG đồng nghĩa đã xác minh bằng thuật toán đo ECG độc lập.')
    st.warning('Xác minh định lượng: NOT_AVAILABLE. Không xuất QRS duration, QTc, độ chênh ST hoặc kết luận bệnh lý được máy đo xác nhận.')
    if second.get('conflicts_or_limits'):st.write('Giới hạn lượt 2:',second['conflicts_or_limits'])
    report=report_export(first,second,rows,boxes,layout,selected,image_hash)
    with io.BytesIO() as stream:
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('evidence_report_v21.json',json.dumps(report,ensure_ascii=False,indent=2))
            z.writestr('README.txt','ECG v2.1 AI Visual QA prototype. Research only. No verified quantitative measurements. Do not publish patient identifiers.\n')
            # No original image in export by default, reduces accidental personal-data leaks.
        data=stream.getvalue()
    st.download_button('Tải ZIP báo cáo AI Visual v2.1 (không kèm ảnh gốc)',data=data,
                       file_name='ecg_core_v2_1_visual_validation.zip',mime='application/zip')
