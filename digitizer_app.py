"""Single-lead trace extraction prototype. No API, learned weights or ECG diagnosis.
Deploy separately from app.py. Uses existing Streamlit/Pillow/NumPy/OpenCV only.
"""
import csv
import hashlib
import io
import json
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

VERSION = "Digitizer 0.3"
METHOD = "ink-graph-review-v3"
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_SOURCE_PIXELS = 25_000_000
MAX_WORK_WIDTH = 2000
MAX_WORK_HEIGHT = 800


def png_bytes(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def orient(image, angle):
    turns = {90:Image.Transpose.ROTATE_90,180:Image.Transpose.ROTATE_180,270:Image.Transpose.ROTATE_270}
    return image.transpose(turns[angle]) if angle in turns else image.copy()


def crop_region(image, x_range, y_range):
    left,right = x_range
    top,bottom = y_range
    if not (0 <= left < right <= 100 and 0 <= top < bottom <= 100):
        raise ValueError("Vùng chọn cần có chiều rộng và chiều cao lớn hơn 0.")
    box = (int(image.width*left/100),int(image.height*top/100),
           int(image.width*right/100),int(image.height*bottom/100))
    if box[2]-box[0] < 32 or box[3]-box[1] < 16:
        raise ValueError("Vùng quá nhỏ. Chọn một đoạn có nhiều nhát tim và đủ toàn bộ chiều cao sóng.")
    return image.crop(box),box


def roi_preview(image, box):
    preview = image.copy()
    preview.thumbnail((1500,1000),Image.Resampling.LANCZOS)
    sx,sy = preview.width/image.width,preview.height/image.height
    draw = ImageDraw.Draw(preview)
    draw.rectangle(tuple((v*sx if i%2==0 else v*sy) for i,v in enumerate(box)),outline="#e53935",width=3)
    return preview


def prepare_working_image(crop):
    work = crop.convert("RGB").copy()
    work.thumbnail((MAX_WORK_WIDTH,MAX_WORK_HEIGHT),Image.Resampling.LANCZOS)
    return work


def ink_mask(image, max_gray=185, local_contrast=10, suppress_color=True, remove_grid=True):
    rgb = np.asarray(image.convert("RGB"))
    # A black trace stays dark while colored grid pixels are lightened.
    gray = np.max(rgb,axis=2).astype(np.uint8) if suppress_color else cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    h,w = gray.shape
    block = min(41,min(h,w) if min(h,w)%2 else min(h,w)-1)
    block = max(3,block)
    adaptive = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,block,float(local_contrast)) > 0
    original = adaptive & (gray < max_gray)
    removed = np.zeros_like(original)
    if remove_grid:
        # Remove only very long straight structures. May also remove ECG segments;
        # those pixels are not silently reconstructed/interpolated.
        horizontal = cv2.morphologyEx(original.astype(np.uint8),cv2.MORPH_OPEN,
                                      np.ones((1,max(24,int(w*.65))),np.uint8)) > 0
        vertical = cv2.morphologyEx(original.astype(np.uint8),cv2.MORPH_OPEN,
                                    np.ones((max(12,int(h*.88)),1),np.uint8)) > 0
        removed = horizontal | vertical
    return original & ~removed,gray,removed


def trace_candidates(mask, gray, continuity=.12, seed_percent=50.):
    """Find a low-cost ink path, then reject unsupported/ambiguous samples.

    This path is an image-processing candidate, not a confirmed ECG signal.
    No spline, gap fill, waveform synthesis or physical unit conversion.
    """
    h,w = mask.shape
    if not np.any(mask):
        raise ValueError("Không có đủ nét tối để dò. Chọn vùng khác hoặc chỉnh ngưỡng tối.")
    rows = np.arange(h,dtype=np.float32)
    indices = np.arange(h,dtype=np.int32)
    parent = np.empty((w,h),dtype=np.uint16)
    intensity = gray.astype(np.float32)/255
    cost = np.where(mask[:,0],intensity[:,0],20.) + continuity*.2*np.abs(rows-seed_percent*(h-1)/100)
    parent[0] = indices
    for x in range(1,w):
        forward_raw = cost-continuity*rows
        forward_min = np.minimum.accumulate(forward_raw)
        forward_parent = np.maximum.accumulate(np.where(forward_raw <= forward_min,indices,0))
        forward = forward_min+continuity*rows
        backward_raw = (cost+continuity*rows)[::-1]
        backward_min = np.minimum.accumulate(backward_raw)
        backward_parent_rev = np.maximum.accumulate(np.where(backward_raw <= backward_min,indices,0))
        backward_parent = (h-1-backward_parent_rev)[::-1]
        backward = backward_min[::-1]-continuity*rows
        use_forward = forward <= backward
        parent[x] = np.where(use_forward,forward_parent,backward_parent)
        cost = np.minimum(forward,backward)+np.where(mask[:,x],intensity[:,x],20.)
    path = np.empty(w,dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for x in range(w-1,0,-1):
        path[x-1] = parent[x,path[x]]
    supported = mask[path,np.arange(w)]
    run_count = mask[0].astype(np.int32)+np.sum(mask[1:] & ~mask[:-1],axis=0)
    # A nearly vertical ink stroke may contain many possible y values at one x.
    # Do not pretend one chosen y uniquely recovers its peak.
    vertical_span = np.zeros(w,dtype=np.int32)
    for x,y in enumerate(path):
        if not supported[x]:
            continue
        lo,hi = y,y
        while lo > 0 and mask[lo-1,x]: lo -= 1
        while hi+1 < h and mask[hi+1,x]: hi += 1
        vertical_span[x] = hi-lo+1
        # Center of the selected connected ink run reduces contour-edge bias.
        # Long near-vertical runs remain ambiguous and are rejected below.
        path[x] = (lo+hi)//2
    status = np.full(w,"ung_vien_net_toi",dtype=object)
    status[~supported] = "khong_co_net_tai_diem_do"
    status[supported & (run_count > 3)] = "nhieu_net_trong_cot"
    status[supported & (vertical_span > max(10,int(h*.14)))] = "net_doc_mo_ho"
    status[supported & ((path <= 1) | (path >= h-2))] = "cham_bien_vung_cat"
    accepted = status == "ung_vien_net_toi"
    values = path.astype(float)
    values[~accepted] = np.nan
    return {"y":values,"status":status,"accepted":accepted,
            "support_fraction":float(np.mean(accepted)),"run_count":run_count,
            "vertical_span":vertical_span}


def trace_overlay(image, values):
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    previous = None
    for x,y in enumerate(values):
        if not np.isfinite(y):
            previous = None
            continue
        point = (x,float(y))
        if previous is not None and abs(y-previous[1]) <= max(12,image.height*.12):
            draw.line((previous,point),fill="#e60073",width=2)
        else:
            draw.ellipse((x-1,y-1,x+1,y+1),fill="#e60073")
        previous = point
    return overlay


def pixels_csv(result, crop_box, work_size, crop_size):
    # Map to the oriented image; NOT back to the camera's pre-EXIF coordinates.
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(["x_work_px","y_work_px_candidate","x_oriented_px","y_oriented_px_candidate","status"])
    ww,wh = work_size
    cw,ch = crop_size
    for x,y in enumerate(result["y"]):
        x_source = crop_box[0]+(x+.5)*cw/ww-.5
        y_source = crop_box[1]+(y+.5)*ch/wh-.5 if np.isfinite(y) else None
        writer.writerow([x,round(float(y),3) if np.isfinite(y) else "",round(x_source,3),
                         round(float(y_source),3) if y_source is not None else "",result["status"][x]])
    return out.getvalue().encode("utf-8-sig")


def analyze_crop(crop, params):
    work = prepare_working_image(crop)
    mask,gray,removed = ink_mask(work,params["max_gray"],params["local_contrast"],params["suppress_color"],params["remove_grid"])
    trace = trace_candidates(mask,gray,params["continuity"],params["seed_percent"])
    graph = trace_ink_graph(mask,trace,include_unanchored=True)
    statuses,counts = np.unique(trace["status"],return_counts=True)
    return {"work":work,"mask":mask,"removed":removed,"graph":graph,**trace,
            "counts":dict(zip(statuses.tolist(),counts.tolist()))}


def thin_ink(mask):
    """Zhang-Suen thinning; output pixels are a subset of input ink, no gap fill."""
    a = np.pad(mask.astype(np.uint8),1)
    for _ in range(100):
        changed = False
        for phase in (0,1):
            p = a[1:-1,1:-1]
            n = [a[:-2,1:-1],a[:-2,2:],a[1:-1,2:],a[2:,2:],
                 a[2:,1:-1],a[2:,:-2],a[1:-1,:-2],a[:-2,:-2]]
            count = sum(n)
            turns = sum((n[i]==0)&(n[(i+1)%8]==1) for i in range(8))
            if phase == 0:
                c1,c2 = n[0]*n[2]*n[4],n[2]*n[4]*n[6]
            else:
                c1,c2 = n[0]*n[2]*n[6],n[0]*n[4]*n[6]
            drop = (p==1)&(count>=2)&(count<=6)&(turns==1)&(c1==0)&(c2==0)
            if np.any(drop):
                p[drop]=0;changed=True
        if not changed:
            break
    return a[1:-1,1:-1].astype(bool)


def trace_ink_graph(mask, column_trace, include_unanchored=False):
    # Select connected ink components supported by the old path; detached text
    # and tiny specks are less likely to survive. Attached grid/text can remain.
    h,w = mask.shape
    number,labels,stats,_ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    votes = {}
    for x,y in enumerate(column_trace['y']):
        if np.isfinite(y):
            label = int(labels[int(round(y)),x])
            if label: votes.setdefault(label,set()).add(x)
    keep = [lab for lab,xs in votes.items() if len(xs)>=3 and stats[lab,cv2.CC_STAT_WIDTH]>=5]
    anchored_labels = set(keep)
    if include_unanchored:
        # Expose previously rejected upright strokes for explicit review.
        # This also exposes text/grid: these components are NOT auto-selected.
        keep = [lab for lab in range(1,number)
                if stats[lab,cv2.CC_STAT_AREA]>=3 and
                (stats[lab,cv2.CC_STAT_WIDTH]>=2 or stats[lab,cv2.CC_STAT_HEIGHT]>=3)]
    selected = np.isin(labels,keep) & mask
    if not np.any(selected):
        raise ValueError('Chưa tìm được thành phần nét đủ dài; chọn vùng sát một chuyển đạo hơn.')
    skeleton = thin_ink(selected)
    ys,xs = np.where(skeleton)
    if len(xs)>100000:
        raise ValueError('Quá nhiều nét ứng viên; thu hẹp vùng hoặc giảm lưới/nhiễu trước.')
    nodes = set(zip(xs.tolist(),ys.tolist()))
    neighbors = {}
    for x,y in sorted(nodes):
        adjacent=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                q=(x+dx,y+dy)
                if (dx==0 and dy==0) or q not in nodes: continue
                # Avoid triangular edges around a rasterized right-angle corner.
                if dx and dy and ((x+dx,y) in nodes or (x,y+dy) in nodes): continue
                adjacent.append(q)
        neighbors[(x,y)] = sorted(adjacent)
    component={};component_info={}
    for start in sorted(nodes):
        if start in component:continue
        cid=len(component_info)+1;stack=[start];component[start]=cid;points=[]
        while stack:
            p=stack.pop();points.append(p)
            for q in neighbors[p]:
                if q not in component:component[q]=cid;stack.append(q)
        branched=any(len(neighbors[p])>2 for p in points)
        loop=all(len(neighbors[p])==2 for p in points)
        boundary=any(p[0] in (0,w-1) or p[1] in (0,h-1) for p in points)
        component_info[cid]={'branch':branched,'loop':loop,'edge':boundary,'pixels':len(points),
                             'anchored':any(int(labels[y,x]) in anchored_labels for x,y in points)}
    visited=set();segments=[]
    def edge(a,b):return tuple(sorted((a,b)))
    def follow(start,next_point):
        points=[start,next_point];visited.add(edge(start,next_point))
        prev,current=start,next_point
        while len(neighbors[current])==2:
            choices=[q for q in neighbors[current] if q!=prev]
            if not choices:break
            q=choices[0]
            if edge(current,q) in visited:break
            visited.add(edge(current,q));points.append(q)
            prev,current=current,q
            if current==start:break
        cid=component[start];info=component_info[cid]
        segments.append({'component_id':cid,'points':points,
                         'ambiguous':info['branch'] or info['loop'] or info['edge'],
                         'suggested':info['anchored']})
    for p in sorted(nodes):
        if len(neighbors[p])!=2:
            for q in neighbors[p]:
                if edge(p,q) not in visited:follow(p,q)
    for p in sorted(nodes):
        for q in neighbors[p]:
            if edge(p,q) not in visited:follow(p,q)
    for p in sorted(nodes):
        if not neighbors[p]:
            cid=component[p];info=component_info[cid]
            segments.append({'component_id':cid,'points':[p],
                             'ambiguous':True,'suggested':info['anchored']})
    return {'selected_mask':selected,'skeleton':skeleton,'segments':segments,
            'component_info':component_info,'junctions':[p for p in sorted(nodes) if len(neighbors[p])>2],
            'stats':{'selected_components':len(keep),'graph_components':len(component_info),
                     'skeleton_pixels':int(skeleton.sum()),'segments':len(segments),
                     'junctions':sum(len(v)>2 for v in neighbors.values()),
                     'ambiguous_components':sum(v['branch'] or v['loop'] or v['edge'] for v in component_info.values())}}


def graph_overlay(image,graph):
    overlay=image.convert('RGB').copy();draw=ImageDraw.Draw(overlay)
    # Draw individual retained skeleton pixels, never connect across a missing pixel.
    for seg in graph['segments']:
        color='#ff8c00' if seg['ambiguous'] else '#e60073'
        for x,y in seg['points']:draw.point((x,y),fill=color)
    for x,y in graph['junctions']:
        draw.ellipse((x-2,y-2,x+2,y+2),outline='#ff8c00',width=1)
    return overlay


def graph_csv(graph,crop_box,work_size,crop_size):
    out=io.StringIO(newline='');writer=csv.writer(out)
    writer.writerow(['component_id','segment_id','point_index','x_work_px','y_work_px',
                     'x_oriented_px','y_oriented_px','status'])
    ww,wh=work_size;cw,ch=crop_size
    for sid,seg in enumerate(graph['segments'],1):
        for i,(x,y) in enumerate(seg['points']):
            writer.writerow([seg['component_id'],seg.get('source_segment_id',sid),i,x,y,
                round(crop_box[0]+(x+.5)*cw/ww-.5,3),round(crop_box[1]+(y+.5)*ch/wh-.5,3),
                'nhanh_vong_hoac_cham_bien' if seg['ambiguous'] else 'doan_net_ung_vien'])
    return out.getvalue().encode('utf-8-sig')


def segment_bounds(seg):
    pts=np.asarray(seg['points'])
    return tuple(int(v) for v in (pts[:,0].min(),pts[:,1].min(),pts[:,0].max(),pts[:,1].max()))


def select_graph(graph, selected_ids):
    selected=set(selected_ids)
    segments=[dict(seg,source_segment_id=i) for i,seg in enumerate(graph['segments'],1) if i in selected]
    return dict(graph,segments=segments)


def review_image(image, graph, selected_ids, focus_id=None, window=None):
    out=image.convert('RGB').copy();draw=ImageDraw.Draw(out)
    selected=set(selected_ids)
    for sid,seg in enumerate(graph['segments'],1):
        color='#008c95' if sid in selected else '#999999'
        if sid==focus_id:color='#e60073'
        for point in seg['points']:draw.point(tuple(point),fill=color)
    # Show endpoints at nearby disconnected components. Markers are annotations,
    # not recovered pixels; there is deliberately no line across the gap.
    endpoints=[]
    for sid,seg in enumerate(graph['segments'],1):
        if sid in selected and len(seg['points'])>1:
            endpoints.extend([(sid,seg['component_id'],seg['points'][0]),
                              (sid,seg['component_id'],seg['points'][-1])])
    tolerance=max(12,int(image.height*.12));buckets={}
    for _,cid,(x,y) in endpoints:
        buckets.setdefault((x//9,y//(tolerance+1)),[]).append((cid,x,y))
    for sid,cid,(x,y) in endpoints:
        nearby=(q for bx in range(x//9-1,x//9+2)
                for by in range(y//(tolerance+1)-1,y//(tolerance+1)+2)
                for q in buckets.get((bx,by),[]))
        if any(cid!=other and 0<abs(x-qx)<=8 and abs(y-qy)<=tolerance
               for other,qx,qy in nearby):
            draw.rectangle((x-2,y-2,x+2,y+2),outline='#d00000')
    if window:
        left,right=window
        out=out.crop((left,0,right,image.height))
        scale=max(1,min(6,1800//max(1,right-left),1200//max(1,image.height)))
        out=out.resize((out.width*scale,out.height*scale),Image.Resampling.NEAREST)
    return out


def review_panel(result):
    graph=result['graph'];image=Image.open(io.BytesIO(result['original_png'])).convert('RGB')
    key='dg_review_'+result['signature'][:16]
    ids=list(range(1,len(graph['segments'])+1))
    defaults=[i for i,seg in enumerate(graph['segments'],1) if seg.get('suggested',True)]
    def label(sid):
        seg=graph['segments'][sid-1];a,b,c,d=segment_bounds(seg)
        return f"#{sid} · x {a}–{c}, y {b}–{d} · {len(seg['points'])} điểm" + (" · nét bổ sung" if not seg.get('suggested',True) else "")
    st.caption("Xanh: đoạn đang chọn; xám: chưa chọn; hồng: đoạn đang xem. Ô đỏ: đầu mút gần thành phần rời khác, chỉ là vị trí cần kiểm tra. Màu không biểu thị độ chính xác.")
    # All choices use work-image coordinates, with a signature-specific state.
    def edit_selection(sid,keep):
        current=set(st.session_state.get(key+'_ids',defaults))
        if keep:current.add(sid)
        else:current.discard(sid)
        st.session_state[key+'_ids']=sorted(current)
    if key+'_ids' not in st.session_state:
        st.session_state[key+'_ids']=defaults
    selected=st.multiselect("Các đoạn giữ trong bản xuất",ids,format_func=label,key=key+'_ids')
    mode=st.radio("Vùng phóng to",['Chọn cửa sổ ngang','Theo đoạn nét'],horizontal=True,key=key+'_mode')
    focus=None
    if mode=='Theo đoạn nét' and ids:
        focus=st.selectbox("Đoạn cần xem / thêm / bỏ",ids,format_func=label,key=key+'_focus')
        a,b,c,d=segment_bounds(graph['segments'][focus-1])
        left=max(0,a-15);right=min(image.width,max(a+1,c)+16)
        st.button("Giữ đoạn đang xem",key=key+'_keep',on_click=edit_selection,args=(focus,True))
        st.button("Bỏ đoạn đang xem",key=key+'_drop',on_click=edit_selection,args=(focus,False))
        st.caption("Đoạn đang xem được tô hồng dù có hoặc chưa có trong danh sách giữ. Dùng danh sách phía trên để thêm/bỏ đúng số đoạn.")
    else:
        percent=st.slider("Cửa sổ QRS: trái — phải (%)",0.,100.,(5.,15.),.1,key=key+'_window')
        left=int(percent[0]*image.width/100);right=max(left+1,int(percent[1]*image.width/100))
        right=min(image.width,right);left=min(left,right-1)
    shown=[i for i,seg in enumerate(graph['segments'],1) if segment_bounds(seg)[0]<right and segment_bounds(seg)[2]>=left]
    st.write("Đoạn trong cửa sổ: "+(', '.join('#'+str(i) for i in shown) or 'Không có'))
    st.image(review_image(image,graph,selected,focus,(left,right)),caption="Phóng to pixel gốc; không làm mượt hay vẽ bù sóng.",width='stretch')
    original=image.crop((left,0,right,image.height))
    factor=max(1,min(6,1800//max(1,right-left),1200//image.height))
    st.image(original.resize((original.width*factor,original.height*factor),Image.Resampling.NEAREST),caption="Ảnh gốc cùng cửa sổ để kiểm tra đỉnh/đáy.",width='stretch')
    reviewed=select_graph(graph,selected)
    overview=review_image(image,graph,selected)
    st.image(overview,caption="Tổng quan lựa chọn hiện tại. Màu xám vẫn hiện để đối chiếu nhưng không nằm trong CSV đã chọn.",width='stretch')
    notes=st.text_area("Ghi chú đoạn còn sai hoặc chưa rõ",key=key+'_notes')
    if not selected:st.warning("Chưa chọn đoạn nào: CSV xuất chỉ có tiêu đề; không có đường tín hiệu.")
    review={'selection_type':'user_editable_segments_not_validation',
            'initial_suggested_segment_ids':defaults,'selected_segment_ids':selected,
            'excluded_segment_ids':[sid for sid in ids if sid not in selected],
            'all_segments':[{'segment_id':sid,'component_id':seg['component_id'],
                'bounds_work_px':segment_bounds(seg),'point_count':len(seg['points']),
                'initial_suggestion':seg.get('suggested',True)} for sid,seg in enumerate(graph['segments'],1)],
            'notes':notes,'zoom_window_work_px':[left,right]}
    return reviewed,review,overview


def clear_result():
    st.session_state.pop("digitizer_result",None)


def main():
    st.set_page_config(page_title="ECG Digitizer — by dr. Lai",page_icon="📈",layout="wide")
    st.title("📈 ECG Digitizer — by dr. Lai")
    st.caption(VERSION+" · Thử nghiệm trích nét của MỘT chuyển đạo · Không cần API")
    st.info("Đường màu là nét tối ứng viên do xử lý ảnh tìm được, chưa được xác nhận là tín hiệu ECG. Bản này chưa đo PR/QRS/ST, chưa chẩn đoán và chưa dùng mô hình số hóa có trọng số.")
    st.caption("Ảnh được xử lý trên máy chủ Streamlit của app. Code này không gửi ảnh đến Gemini/GPT hay dịch vụ số hóa bên ngoài; không ghi ảnh vào đĩa hay bộ nhớ dùng chung giữa các phiên.")
    uploaded = st.file_uploader("Tải ảnh ECG",type=["jpg","jpeg","png","webp","heic","heif"],key="dg_upload",on_change=clear_result)
    if uploaded is None:
        st.write("Bắt đầu với ảnh thẳng, rõ. Chọn một chuyển đạo, bỏ chữ và xung chuẩn khỏi vùng dò, giữ đủ chiều cao P–QRS–T.")
        return
    blob = uploaded.getvalue()
    if len(blob) > MAX_FILE_BYTES:
        st.error("Ảnh vượt 20 MB. Xuất lại JPEG/PNG nhỏ hơn rồi tải lại.")
        return
    try:
        source = Image.open(io.BytesIO(blob))
        if source.width*source.height > MAX_SOURCE_PIXELS:
            raise ValueError("Ảnh vượt 25 triệu pixel; xuất lại ảnh nhỏ hơn.")
        source = ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:
        st.error("Không đọc được ảnh. Thử xuất JPEG/PNG. " + (str(exc) if isinstance(exc,ValueError) else ""))
        return
    sha = hashlib.sha256(blob).hexdigest()
    suffix = sha[:12]
    st.subheader("1. Chọn đúng một chuyển đạo")
    rotation = st.selectbox("Xoay ngược chiều kim đồng hồ",[0,90,180,270],key="dg_rot_"+suffix)
    oriented = orient(source,rotation)
    frame = suffix+str(rotation)
    lead = st.selectbox("Nhãn chuyển đạo do em chọn",["II","I","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6","Chưa xác định"],key="dg_lead_"+frame)
    x_range = st.slider("Vùng dò: trái — phải (%)",0.,100.,(0.,95.),.1,key="dg_roi_x_"+frame)
    y_range = st.slider("Vùng dò: trên — dưới (%)",0.,100.,(40.,58.),.1,key="dg_roi_y_"+frame)
    try:
        crop,box = crop_region(oriented,x_range,y_range)
    except ValueError as exc:
        st.error(str(exc));return
    st.image(roi_preview(oriented,box),caption="Khung đỏ là vùng sẽ dò. Kiểm tra cả đỉnh R và đáy S không bị cắt.",width="stretch")
    st.image(crop,caption="Ảnh gốc của vùng chọn. Nên có 2–5 nhát, một hàng sóng; tránh chữ, hàng sóng khác và xung chuẩn.",width="stretch")
    st.caption("Nhãn do người dùng chọn, app chưa tự xác nhận đúng chuyển đạo. Vùng quá cao có thể khiến thuật toán bám nhầm hàng khác.")
    st.subheader("2. Trích nét thử nghiệm")
    with st.expander("Thiết lập xử lý ảnh (chỉ chỉnh khi nét bị sai)"):
        suppress_color = st.checkbox("Nét đen: giảm lưới màu",value=True,key="dg_color_"+frame)
        remove_grid = st.checkbox("Thử gỡ đường lưới thẳng dài",value=True,key="dg_grid_"+frame)
        st.caption("Gỡ lưới có thể làm mất cả đoạn ECG trùng lưới. Hãy thử tắt nếu đường màu bị đứt; không tự điền đoạn đã mất.")
        max_gray = st.slider("Ngưỡng tối tối đa",30,245,185,key="dg_gray_"+frame)
        local_contrast = st.slider("Tương phản cục bộ tối thiểu",2,35,10,key="dg_contrast_"+frame)
        continuity = st.slider("Ưu tiên đường liên tục",.02,.4,.12,.01,key="dg_cont_"+frame)
        seed_percent = st.slider("Gợi ý độ cao bắt đầu trong vùng (%)",0.,100.,50.,1.,key="dg_seed_"+frame)
    params = dict(max_gray=max_gray,local_contrast=local_contrast,suppress_color=suppress_color,
                  remove_grid=remove_grid,continuity=continuity,seed_percent=seed_percent)
    signature = hashlib.sha256(json.dumps([sha,rotation,lead,box,params,VERSION],sort_keys=True).encode()).hexdigest()
    if st.button("Trích đường ứng viên",type="primary",key="dg_run"):
        clear_result()
        started = time.monotonic()
        try:
            with st.spinner("Đang xử lý ảnh tại app, không gọi API..."):
                candidate = analyze_crop(crop,params)
            work = candidate["work"]
            metadata = {"version":VERSION,"method":METHOD,"validation_status":"unverified_candidate",
                "time_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":sha,
                "original_exif_corrected_size":list(source.size),"rotation_ccw":rotation,
                "oriented_size":list(oriented.size),"crop_box_oriented_pixels":list(box),
                "crop_size":list(crop.size),"working_size":list(work.size),"chosen_lead":lead,
                "parameters":params,"column_status_counts":candidate["counts"],
                "candidate_column_fraction":candidate["support_fraction"],"graph_stats":candidate["graph"]["stats"],
                "seconds":round(time.monotonic()-started,3),
                "coordinates":"CSV contains 2D segment points. Repeated x and backwards x are allowed. Segment order is graph traversal, NOT time order. No ms/mV.",
                "limits":"Dark ink is not proven to be ECG. No gap filling. No P/QRS/T landmarks. No clinical validation."}
            st.session_state["digitizer_result"] = {"signature":signature,"metadata":metadata,
                "original_png":png_bytes(work),"overlay_png":png_bytes(graph_overlay(work,candidate["graph"])),
                "old_overlay_png":png_bytes(trace_overlay(work,candidate["y"])),
                "mask_png":png_bytes(Image.fromarray(candidate["mask"].astype(np.uint8)*255)),
                "graph":candidate["graph"],
                "csv":graph_csv(candidate["graph"],box,work.size,crop.size)}
        except (ValueError,cv2.error) as exc:
            st.error(str(exc))
    result = st.session_state.get("digitizer_result")
    if not result:
        return
    if result["signature"] != signature:
        st.info("Ảnh, vùng chọn hoặc thiết lập đã đổi. Kết quả cũ không áp dụng; bấm trích lại.")
        return
    st.subheader("3. Đối chiếu với ảnh gốc")
    tabs = st.tabs(["Tất cả nét 2D","Ảnh gốc","Mặt nạ nét tối","Cách cũ để đối chiếu"])
    with tabs[0]:st.image(result["overlay_png"],caption="Gồm cả nét bổ sung chưa chọn; hãy chọn ở bước 4. Hồng: đoạn ứng viên. Cam: thành phần có nhánh/vòng/chạm biên. Cả hai chưa xác minh. Đường được vẽ từ pixel nét giữ lại, không nối khoảng mất.",width="stretch")
    with tabs[1]:st.image(result["original_png"],width="stretch")
    with tabs[2]:st.image(result["mask_png"],caption="Mặt nạ có thể còn chữ/lưới hoặc mất sóng thật.",width="stretch")
    with tabs[3]:st.image(result["old_overlay_png"],caption="Cách 0.1: một điểm mỗi cột; giữ để thấy đoạn dọc bị bỏ hoặc co về giữa.",width="stretch")
    meta=result["metadata"];gs=meta["graph_stats"]
    st.write(f"Thời gian: {meta['seconds']:.2f} giây · {gs['segments']} đoạn nét · {gs['junctions']} điểm phân nhánh")
    st.caption("Số đoạn/pixel không phải độ chính xác. Giữ thêm nét dọc có thể đồng thời giữ thêm lưới hoặc chữ.")
    if gs["ambiguous_components"]:
        st.warning("Có thành phần mơ hồ màu cam. Chưa thể quyết định nhánh nào là ECG; không dùng để đo.")
    st.warning("Chưa có tín hiệu thời gian hay các mốc P/QRS/T. Đường nằm trên mực thật vẫn có thể là lưới/chữ. Kiểm tra riêng đỉnh R và đáy S.")
    st.subheader("4. Phóng to và chọn nhánh nét")
    reviewed_graph,review_meta,review_overlay = review_panel(result)
    meta = dict(meta)
    meta["review"] = review_meta
    meta["validation_status"] = "manual_selection_unverified"
    meta["export_graph_stats"] = {"segments":len(reviewed_graph["segments"]),
        "unique_points":len({tuple(p) for seg in reviewed_graph["segments"] for p in seg["points"]})}
    meta["limits"] += " Manual selection does not establish a time signal or clinical validity."
    review_csv = graph_csv(reviewed_graph,box,tuple(meta["working_size"]),tuple(meta["crop_size"]))
    st.subheader("5. Tải kết quả thử nghiệm — từng file riêng")
    st.caption("CSV mới chứa các đoạn nét 2D: có thể nhiều điểm cùng x, chưa phải tín hiệu theo thời gian. Các đoạn khác nhau không được tự nối hoặc lấy trung bình để đo. Ảnh có thể chứa thông tin ca bệnh; che định danh trước khi tải lên.")
    st.download_button("Tải ảnh phủ màu đã chọn",png_bytes(review_overlay),"digitizer_overlay.png","image/png",key="dg_overlay_download")
    st.download_button("Tải tọa độ CSV đã chọn",review_csv,"digitizer_trace_points.csv","text/csv",key="dg_csv_download")
    st.download_button("Tải cấu hình JSON",json.dumps(meta,ensure_ascii=False,indent=2),"digitizer_record.json","application/json",key="dg_json_download")
    st.download_button("Tải mặt nạ PNG",result["mask_png"],"digitizer_mask.png","image/png",key="dg_mask_download")
    st.subheader("6. Bước tiếp theo")
    st.write("1. Dùng lại ảnh III và vùng cắt cũ. Mở từng QRS, đối chiếu ảnh gốc với tất cả nét ứng viên.")
    st.write("2. Chọn thêm đoạn bị bỏ và bỏ nhánh chữ/lưới nếu nhìn chắc. Chưa rõ thì giữ nguyên và ghi chú; không đo PR/QRS/ST.")
    st.write("3. Tải ảnh phủ màu đã chọn, CSV đã chọn, JSON và mặt nạ. Gửi đủ 4 file để người phát triển kiểm tra; chưa cần ECG mới.")
    st.caption("Chọn nhánh chỉ là thao tác đối chiếu của người dùng, không phải xác nhận tín hiệu đúng. Khoảng mất không được nối tự động.")



if __name__ == "__main__":
    main()
