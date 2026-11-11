"""ECG Core v2.3 — AI Visual Reader + Adaptive Cropping. Research prototype.

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
from core.group_crops_v22 import panel_as_jpeg, v22_report, GROUPS
from core.group_crops_v231 import default_direct_boxes, direct_panels, validate_direct_group_boxes
from ui.grid_measurement_panel_v23 import render as render_grid_measurement_v23

st.set_page_config(page_title='ECG Core v2.3.1 | AI Visual Evidence',layout='wide')
st.title('ECG Core v2.3.1 · AI Visual Evidence Lab')
st.caption('Một ảnh gốc → AI đọc đủ 12 lead → 4 nhóm cố định + DII rhythm nếu có → AI đọc lại → rà soát bằng chứng. NGHIÊN CỨU.')
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
    st.info('Tải một ảnh ECG để bắt đầu. Ứng dụng tự tạo 4 hoặc 5 ảnh nhóm từ ROI.')
    st.stop()
try:
    photo=image_from_bytes(upload.getvalue())
except Exception as exc:
    st.error(f'Không mở được ảnh: {exc}')
    st.stop()
image_hash=image_identity(photo)
if st.session_state.get('v21_image_hash')!=image_hash:
    for state_key in ('v21_first','v21_second','v21_selected','v21_completed_layout','v22_second','v22_config_hash'):
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

st.subheader('3 · Xác nhận bố cục và 5 nhóm ảnh ECG')
col1,col2,col3=st.columns(3)
with col1: layout=st.selectbox('Bố cục trên ảnh gốc',['3x4','6x2'])
with col2: rhythm=st.checkbox('Ảnh thật có dải DII rhythm riêng',value=(layout=='3x4'))
with col3: margin=st.slider('Lề tránh tín hiệu liền kề',0.0,.06,.02,.005)
if rhythm:
    rhythm_start=st.slider('Ranh giới bắt đầu dải rhythm (%)',65,92,78)/100
else: rhythm_start=.78
if layout=='6x2' and not rhythm:
    st.info('Bố cục 6×2 không có dải rhythm dài: chỉ tạo 4 nhóm. DII ngắn không thể thay cho DII rhythm.')
try:
    boxes=layout_boxes(photo.size,layout,rhythm=rhythm,rhythm_start=rhythm_start,margin=margin)
except ValueError as exc:
    st.error(str(exc)); st.stop()

# Group crop and per-lead measurement ROI are intentionally separate.
import hashlib
config_tag=f'{image_hash}:{layout}:{rhythm}:{rhythm_start:.4f}:{margin:.4f}'
if st.session_state.get('v231_for')!=config_tag:
    st.session_state['v231_for']=config_tag
    st.session_state['v231_group_override']={}
    st.session_state['v231_lead_override']={}
    st.session_state.pop('v22_second',None)
    st.session_state.pop('v22_config_hash',None)

# Existing lead boxes continue to serve the optional v2.3 measurement QA.
for lead,box in st.session_state['v231_lead_override'].items():
    if lead in boxes: boxes[lead]=tuple(box)
group_boxes=default_direct_boxes(photo.size,layout,rhythm,rhythm_start,margin)
for name,box in st.session_state['v231_group_override'].items():
    if name in group_boxes:group_boxes[name]=tuple(box)

with st.expander('Chỉnh trực tiếp 4/5 KHUNG NHÓM (không phải 12 lead)',expanded=False):
    group_id=st.selectbox('Nhóm muốn điều chỉnh',list(group_boxes),
                          format_func=lambda k:f"{k} · {', '.join(GROUPS[k])}",key='v231_edit_group')
    current=group_boxes[group_id]
    editor=st.columns(4)
    coords=[]
    for j,(label,minimum,maximum) in enumerate(zip(('X trái','Y trên','X phải','Y dưới'),
                                                  (0,0,1,1),
                                                  (photo.width-1,photo.height-1,photo.width,photo.height))):
        coords.append(editor[j].number_input(label,min_value=minimum,max_value=maximum,
                     value=int(current[j]),step=5,key=f'v231_{config_tag}_{group_id}_{j}'))
    if st.button('Lưu vùng NHÓM',key='v231_save_group'):
        try:
            trial=dict(group_boxes)
            trial[group_id]=tuple(coords)
            validate_direct_group_boxes(trial,photo.size,layout,rhythm)
            st.session_state['v231_group_override'][group_id]=tuple(coords)
            st.session_state.pop('v22_second',None)
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
    if st.button('Khôi phục 5 khung mặc định',key='v231_reset_group'):
        st.session_state['v231_group_override']={}
        st.session_state.pop('v22_second',None)
        st.rerun()
    st.caption('Mỗi nhóm được lấy trực tiếp từ một hình chữ nhật trong ảnh gốc. Không cắt/ghép riêng ba hàng.')

with st.expander('Chỉnh ROI TỪNG LEAD chỉ dành cho Measurement QA v2.3 (nâng cao)',expanded=False):
    lead=st.selectbox('Lead để đo',list(boxes),key='v231_lead')
    old=boxes[lead]
    editor=st.columns(4)
    xy=[]
    for j,(label,lo,hi) in enumerate(zip(('X trái','Y trên','X phải','Y dưới'),
                                         (0,0,1,1),(photo.width-1,photo.height-1,photo.width,photo.height))):
        xy.append(editor[j].number_input(label,min_value=lo,max_value=hi,value=int(old[j]),
                     step=5,key=f'v231_lead_{config_tag}_{lead}_{j}'))
    if st.button('Lưu ROI lead đo',key='v231_save_lead'):
        from core.group_crops_v22 import validate_roi
        try:
            st.session_state['v231_lead_override'][lead]=validate_roi(tuple(xy),photo.size)
            st.rerun()
        except ValueError as exc:st.error(str(exc))
    st.caption('Chỉnh lead QA KHÔNG thay đổi năm ảnh nhóm gửi AI.')

try:
    group_boxes=validate_direct_group_boxes(group_boxes,photo.size,layout,rhythm)
    panels=direct_panels(photo,group_boxes,boxes)
except ValueError as exc:
    st.error('Khung nhóm không hợp lệ: '+str(exc))
    st.stop()

preview=photo.copy()
draw=ImageDraw.Draw(preview)
colors=('#dc3535','#2dae5c','#246de3','#a06a42','#9f4db2')
for index,(name,(x0,y0,x1,y1)) in enumerate(group_boxes.items()):
    color=colors[index]
    draw.rectangle((x0,y0,x1,y1),outline=color,width=max(3,photo.width//400))
    draw.text((x0+5,y0+4),name,fill=color,stroke_width=2,stroke_fill='white')
st.image(preview,caption='Chỉ 4/5 khung NHÓM trực tiếp. Ranh giới là đề xuất hình học; kiểm tra không cắt R/S, P/T và nhãn.',width='stretch')
st.caption('Measurement QA v2.3 vẫn giữ 12 ROI riêng bên trong; chúng KHÔNG được dùng để tạo ảnh nhóm.')

st.markdown('**Nhóm gửi lượt 2 (đủ 12 lead, không chọn lọc theo kết luận AI):**')
for p in panels:
    st.caption(f"{p.group_id}: {', '.join(p.leads)}")
if len(panels)==4:
    st.warning('Không có dải DII rhythm riêng. Không thể đánh giá nhịp kéo dài từ nhóm DII ngắn; báo cáo sẽ ghi giới hạn này.')
with st.expander('Xem trước 4/5 ảnh nhóm thật sẽ gửi Gemini',expanded=True):
    cols=st.columns(2)
    for i,p in enumerate(panels):
        with cols[i%2]:
            st.image(p.image,caption=f"{p.group_id} · {', '.join(p.leads)}",width='stretch')
    st.warning('Nếu bất kỳ ảnh nhóm nào mất R/S, chứa lead khác hoặc sai nhãn, hãy sửa ROI và xem lại. Không gửi AI khi crop chưa đúng.')

confirmed=st.checkbox('Tôi xác nhận các nhóm crop có đúng chuyển đạo và không cắt mất nét sóng quan trọng',value=False)
second_prompt="""Đọc các ảnh NHÓM ECG được cung cấp theo tên nhóm và nhãn từng hàng trong ảnh. Mỗi nhóm chứa ba chuyển đạo riêng (hoặc một dải II rhythm); KHÔNG giả định chúng được ghi đồng thời hoặc là cùng một nhịp. Đây là lượt mô tả độc lập; không có kết luận lượt 1. Đối với từng lead nhìn thấy, ghi visibility, mô tả quan sát, P, hình thái QRS, J/ST và T; nếu không chắc hãy ghi UNKNOWN, không tự suy từ chẩn đoán in trên ảnh. Nếu có II-rhythm, mô tả nhịp và chuỗi P-QRS. Bắt buộc ghi riêng TẤT CẢ các lead được gửi và không tự tạo lead thiếu. Không xuất độ chênh ST hoặc khoảng QRS chính xác khi chưa có đo lường đã xác minh. Trả JSON duy nhất: {"lead_observations":[{"lead":"I","visibility":"CLEAR|LIMITED|UNKNOWN","observation":"...","p":"...","qrs":"...","st_t":"..."}],"conflicts_or_limits":"..."}"""
config_fingerprint=hashlib.sha256(json.dumps({'layout':layout,'group_boxes':group_boxes,'groups':[p.group_id for p in panels]},sort_keys=True).encode()).hexdigest()[:20]
if st.button('AI đọc chi tiết toàn bộ nhóm (lượt 2)',disabled=not confirmed,type='primary'):
    try:
        with st.spinner('Gemini đang quan sát 4/5 nhóm ECG độc lập...'):
            images=[(f"{p.group_id}: {', '.join(p.leads)}",panel_as_jpeg(p)) for p in panels]
            raw,used_route=analyze_with_routes(routes,second_prompt,images)
            expected=[lead for p in panels for lead in p.leads]
            second=normalize_second(extract_json(raw),expected)
        st.session_state['v22_second']=second
        st.session_state['v22_config_hash']=config_fingerprint
        st.success(f'AI lượt 2 đã chạy qua {used_route}; các nhận xét vẫn cần bác sĩ xác minh.')
    except Exception as exc:
        st.error(f'Không đọc được lượt 2: {exc}')

second=st.session_state.get('v22_second')
if second:
    if st.session_state.get('v22_config_hash')!=config_fingerprint:
        st.warning('Bố cục/ROI đã thay đổi: lượt đọc 2 không còn khớp với ảnh đang xem. Hãy chạy lại.');st.stop()
    st.subheader('4 · Đối chiếu bằng chứng và cảnh báo logic')
    rows=[]
    for p in panels:
        for lead in p.leads:
            first_ob=first.get('rhythm_observation','UNKNOWN') if lead=='II-rhythm' else first.get('lead_observations',{}).get(lead,{}).get('observation','UNKNOWN')
            second_ob=second.get('lead_observations',{}).get(lead,{}).get('observation','NOT_REVIEWED')
            rows.append({'group':p.group_id,'lead':lead,'AI_original':first_ob,'AI_group_crop':second_ob,'status':'AI_VISUAL_ONLY_NOT_QUANTITATIVE_VERIFICATION'})
    st.dataframe(rows,width='stretch',hide_index=True)
    report=v22_report(first,second,panels,boxes,layout,image_hash)
    report['schema_version']='2.3.1'
    report['group_crop_method']='DIRECT_SINGLE_RECTANGLE_FROM_ORIGINAL'
    report['direct_group_boxes']={k:list(v) for k,v in group_boxes.items()}
    for entry,panel in zip(report['group_crops'],panels):
        entry['group_box']=list(panel.group_box)
    flags=report['consistency_flags']
    if flags:
        st.warning(f'{len(flags)} mục cần rà soát logic hoặc thiếu quan sát ở lượt 2:')
        st.dataframe(flags,width='stretch',hide_index=True)
    else:
        st.info('Không phát hiện mâu thuẫn theo bộ quy tắc hiện có. Điều này KHÔNG xác nhận kết luận AI là đúng.')
    st.warning('ĐỊNH LƯỢNG: NOT_AVAILABLE. Chưa có bộ đo độc lập được nghiệm thu để xác minh QRS, RR, ST hoặc QTc.')
    if second.get('conflicts_or_limits'): st.caption('Giới hạn lượt 2: '+second['conflicts_or_limits'])
    include_crops=st.checkbox('Kèm ảnh crop trong ZIP validation (chỉ khi đã khử định danh)',value=False)
    with io.BytesIO() as stream:
        with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
            z.writestr('evidence_report_v231.json',json.dumps(report,ensure_ascii=False,indent=2))
            z.writestr('README.txt','ECG v2.3.1 direct-group visual QA; measurement QA remains separate. Research only. Original image omitted.\n')
            if include_crops:
                for p in panels:
                    z.writestr(f'groups/{p.group_id}.jpg',panel_as_jpeg(p))
        data=stream.getvalue()
    st.download_button('Xuất ZIP validation v2.3.1',data=data,
        file_name='ecg_core_v2_3_1_direct_group_visual_validation.zip',mime='application/zip')


st.divider()
render_grid_measurement_v23(photo,boxes,layout,image_hash,routes)
