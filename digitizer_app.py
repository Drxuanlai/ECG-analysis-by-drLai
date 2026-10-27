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

VERSION = "Digitizer 0.5"
METHOD = "ink-graph-selection-review-v5"
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


def photo_ink_mask(image, contrast=55, red_tolerance=6):
    """Experimental RGB mask, not enhancement of the source or trace synthesis.

    Local channel envelopes reduce uneven lighting; a red-excess gate rejects
    some warm grid pixels. A very-dark override retains some ink/grid crossings.
    None of these tests proves a pixel belongs to ECG. Colored/weak ECG may vanish.
    """
    rgb=np.asarray(image.convert('RGB')).astype(np.float32)
    bg=cv2.morphologyEx(rgb,cv2.MORPH_CLOSE,np.ones((21,21),np.uint8))
    bg=cv2.GaussianBlur(bg,(21,21),0)
    normalized=np.clip(rgb/np.maximum(bg,1)*255,0,255).max(axis=2)
    red_excess=rgb[:,:,0]-np.maximum(rgb[:,:,1],rgb[:,:,2])
    darkness=normalized < 255-int(contrast)
    color_ok=(red_excess < float(red_tolerance)) | (normalized < 150)
    mask=darkness & color_ok
    removed=darkness & ~color_ok
    return mask,normalized.astype(np.uint8),removed


def extraction_quality(mask, graph):
    """Engineering triage only. Thresholds are heuristics, not clinical validation."""
    default=[seg for seg in graph['segments'] if seg.get('suggested',True)]
    xs={x for seg in default for x,y in seg['points']}
    coverage=len(xs)/mask.shape[1]
    density=float(mask.mean());stats=graph['stats'];reasons=[]
    if not default or coverage < .65:
        reasons.append('Lựa chọn ban đầu bỏ nhiều cột ảnh; có thể mất đường ECG.')
    if stats['segments']>1500 or stats['junctions']>500:
        reasons.append('Quá nhiều đoạn/nhánh; có thể lẫn mạng lưới hoặc chữ.')
    if density>.30:
        reasons.append('Mặt nạ quá dày; có thể chứa nhiều nền/lưới.')
    blocked=bool(reasons)
    if not blocked and (stats['segments']>250 or stats['junctions']>100):
        reasons.append('Còn nhiều nét rời/phân nhánh; cần xem kỹ từng đỉnh và đáy.')
    return {'status':'not_ready' if blocked else 'needs_visual_review',
            'reasons':reasons,'mask_ink_fraction':density,
            'initial_selected_x_fraction':round(coverage,6),
            'meaning':'Heuristic image triage only, never accuracy or measurement approval.'}


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
    if params.get('mask_mode','legacy') == 'photo_red':
        mask,gray,removed=photo_ink_mask(work,params.get('photo_contrast',55),params.get('red_tolerance',6))
    else:
        mask,gray,removed = ink_mask(work,params["max_gray"],params["local_contrast"],params["suppress_color"],params["remove_grid"])
    trace = trace_candidates(mask,gray,params["continuity"],params["seed_percent"])
    graph = trace_ink_graph(mask,trace,include_unanchored=True)
    statuses,counts = np.unique(trace["status"],return_counts=True)
    return {"work":work,"mask":mask,"removed":removed,"graph":graph,**trace,
            "counts":dict(zip(statuses.tolist(),counts.tolist())),
            "quality":extraction_quality(mask,graph)}


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


def selected_quality(graph, selected_ids, size):
    """Describe the current selection only. No waveform/medical accuracy score."""
    width,height=size;chosen=set(selected_ids)
    nodes={tuple(p) for sid,seg in enumerate(graph['segments'],1) if sid in chosen for p in seg['points']}
    neighbors={}
    for x,y in nodes:
        adjacent=[]
        for dy in (-1,0,1):
            for dx in (-1,0,1):
                q=(x+dx,y+dy)
                if (not dx and not dy) or q not in nodes:continue
                if dx and dy and ((x+dx,y) in nodes or (x,y+dy) in nodes):continue
                adjacent.append(q)
        neighbors[(x,y)]=adjacent
    seen=set();components=0
    for start in nodes:
        if start in seen:continue
        components+=1;stack=[start];seen.add(start)
        while stack:
            for q in neighbors[stack.pop()]:
                if q not in seen:seen.add(q);stack.append(q)
    columns={}
    for x,y in nodes:columns.setdefault(x,[]).append(y)
    xs=sorted(columns);gap_ranges=[]
    if xs:
        for a,b in zip(xs,xs[1:]):
            if b-a>1:gap_ranges.append([a+1,b-1])
    split_columns=0
    for ys in columns.values():
        ys=sorted(ys)
        # A connected vertical stroke has many y but only one run.
        if any(b-a>1 for a,b in zip(ys,ys[1:])):split_columns+=1
    branches=sum(len(q)>2 for q in neighbors.values())
    longest=max((b-a+1 for a,b in gap_ranges),default=0)
    fraction=len(columns)/max(1,width);split_fraction=split_columns/max(1,len(columns))
    reasons=[]
    if not nodes:reasons.append('Chưa giữ đoạn nào.')
    elif fraction<.65:reasons.append('Lựa chọn hiện tại bỏ nhiều cột ảnh.')
    if longest>max(10,int(width*.02)):
        reasons.append('Có khoảng thiếu cột dài bên trong lựa chọn.')
    if split_fraction>.25:
        reasons.append('Nhiều cột có các cụm nét tách rời; có thể giữ nhiều nhánh/hàng hoặc lưới.')
    if branches>30 and branches/max(1,len(nodes))>.12:
        reasons.append('Mật độ phân nhánh của các điểm đang giữ còn cao.')
    # Absence of flags never promotes the selection to validated ECG.
    return {'status':'not_ready' if reasons else 'needs_visual_review',
        'reasons':reasons,'selected_segments':len(chosen),'unique_points':len(nodes),
        'selected_x_columns':len(columns),'image_width_px':width,
        'x_column_fraction':round(fraction,6),'connected_components':components,
        'junctions':branches,'endpoints':sum(len(q)==1 for q in neighbors.values()),
        'split_y_columns':split_columns,'split_y_column_fraction':round(split_fraction,6),
        'longest_internal_missing_x_px':longest,'internal_missing_x_ranges':gap_ranges,
        'meaning':'Image-graph heuristics only. Coverage is NOT accuracy. Multiple y are allowed; no time signal, no clinical approval.'}


def segment_hints(graph, size):
    """Short attached segments are review hints only, never auto-deleted."""
    endpoint_counts={}
    for seg in graph['segments']:
        for p in {tuple(seg['points'][0]),tuple(seg['points'][-1])}:
            endpoint_counts[p]=endpoint_counts.get(p,0)+1
    limit=max(5,min(12,round(size[1]*.04)))
    hints={}
    for sid,seg in enumerate(graph['segments'],1):
        a,b,c,d=segment_bounds(seg)
        attached=any(endpoint_counts[tuple(p)]>1 for p in (seg['points'][0],seg['points'][-1]))
        short=len(seg['points'])<=limit and attached
        hints[sid]={'short_attached_review_hint':short,'point_count':len(seg['points']),
                    'bounds_work_px':[a,b,c,d]}
    return hints


def review_panel(result):
    graph=result['graph'];image=Image.open(io.BytesIO(result['original_png'])).convert('RGB')
    key='dg_review_'+result['signature'][:16]
    ids=list(range(1,len(graph['segments'])+1))
    defaults=[i for i,seg in enumerate(graph['segments'],1) if seg.get('suggested',True)]
    hints=segment_hints(graph,image.size)
    state_key=key+'_selected'
    if state_key not in st.session_state:st.session_state[state_key]=defaults
    def edit_selection(sid,keep):
        current=set(st.session_state.get(state_key,defaults))
        if keep:current.add(sid)
        else:current.discard(sid)
        st.session_state[state_key]=sorted(current)
    def restore_selection():st.session_state[state_key]=list(defaults)
    selected=list(st.session_state[state_key]);chosen=set(selected)
    st.caption('Xanh: đang giữ; xám: chưa giữ; hồng: đoạn đang xem. Gợi ý nhánh ngắn chỉ để kiểm tra, không xác định đó là nhiễu. Không tự xóa hoặc nối sóng.')
    st.button('Khôi phục lựa chọn ban đầu',key=key+'_reset',on_click=restore_selection)
    mode=st.radio('Vùng phóng to',['Chọn cửa sổ ngang','Theo đoạn nét'],horizontal=True,key=key+'_mode')
    focus=None
    def label(sid):
        hint=hints[sid];a,b,c,d=hint['bounds_work_px']
        flags=(' · đang giữ' if sid in chosen else ' · chưa giữ')
        if hint['short_attached_review_hint']:flags+=' · nhánh ngắn cần xem'
        return f'#{sid} · x {a}–{c}, y {b}–{d} · {hint["point_count"]} điểm'+flags
    if mode=='Theo đoạn nét' and ids:
        focus=st.selectbox('Đoạn cần xem / thêm / bỏ',ids,format_func=label,key=key+'_focus_all')
        a,b,c,d=hints[focus]['bounds_work_px']
        left=max(0,a-15);right=min(image.width,max(a+1,c)+16)
    else:
        percent=st.slider('Cửa sổ QRS: trái — phải (%)',0.,100.,(5.,15.),.1,key=key+'_window')
        left=min(image.width-1,int(percent[0]*image.width/100))
        right=min(image.width,max(left+1,int(percent[1]*image.width/100)))
    shown=[sid for sid in ids if hints[sid]['bounds_work_px'][0]<right and hints[sid]['bounds_work_px'][2]>=left]
    # Use short local lists; whole-image graph may contain thousands of background fragments.
    filter_label=st.selectbox('Lọc đoạn trong cửa sổ',['Tất cả','Đang giữ','Chưa giữ','Nhánh ngắn cần xem'],key=key+'_filter')
    local=[sid for sid in shown if filter_label=='Tất cả' or
           (filter_label=='Đang giữ' and sid in chosen) or
           (filter_label=='Chưa giữ' and sid not in chosen) or
           (filter_label=='Nhánh ngắn cần xem' and hints[sid]['short_attached_review_hint'])]
    st.write(f'{len(shown)} đoạn trong cửa sổ · {len(local)} đoạn theo bộ lọc.')
    if mode=='Chọn cửa sổ ngang' and local:
        # Page at 60 items without silently discarding candidates or changing IDs.
        pages=max(1,(len(local)+59)//60)
        page=st.selectbox('Trang danh sách đoạn',list(range(1,pages+1)),key=key+f'_page_{left}_{right}_{filter_label}')
        options=local[(page-1)*60:page*60]
        focus=st.selectbox('Đoạn trong cửa sổ cần xem',options,format_func=label,key=key+'_local_'+hashlib.sha256(json.dumps(options).encode()).hexdigest()[:12])
    if focus is not None:
        st.button('Giữ đoạn đang xem',key=key+'_keep',on_click=edit_selection,args=(focus,True))
        st.button('Bỏ đoạn đang xem',key=key+'_drop',on_click=edit_selection,args=(focus,False))
        st.caption(label(focus)+'. Hồng chỉ là đoạn đang xem, kể cả khi chưa được giữ trong CSV.')
    st.image(review_image(image,graph,selected,focus,(left,right)),caption='Phóng to các nét trong cửa sổ; dấu đỏ là đầu mút gần thành phần rời, không phải mốc ECG.',width='stretch')
    original=image.crop((left,0,right,image.height))
    factor=max(1,min(6,1800//max(1,right-left),1200//image.height))
    st.image(original.resize((original.width*factor,original.height*factor),Image.Resampling.NEAREST),caption='Ảnh gốc cùng cửa sổ, không vẽ lại hoặc làm mượt sóng.',width='stretch')
    reviewed=select_graph(graph,selected);overview=review_image(image,graph,selected)
    st.image(overview,caption='Lựa chọn đang xuất: xanh được giữ; xám chỉ để đối chiếu, không có trong CSV.',width='stretch')
    notes=st.text_area('Ghi chú đoạn còn sai hoặc chưa rõ',key=key+'_notes')
    review={'selection_type':'user_editable_segments_not_validation',
        'initial_suggested_segment_ids':defaults,'selected_segment_ids':selected,
        'excluded_segment_ids':[sid for sid in ids if sid not in chosen],
        'all_segments':[{'segment_id':sid,'component_id':seg['component_id'],
            **hints[sid],'initial_suggestion':seg.get('suggested',True)} for sid,seg in enumerate(graph['segments'],1)],
        'notes':notes,'zoom_window_work_px':[left,right],'local_filter':filter_label,
        'hint_meaning':'Short attached segment is only a review hint; no ECG/noise classification, no automatic removal.'}
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
    mask_label=st.selectbox("Cách tách nét",["Ảnh sạch / cách 0.3","Ảnh chụp lưới đỏ (thử nghiệm)"],key="dg_mask_mode_"+frame)
    mask_mode='photo_red' if mask_label.startswith('Ảnh chụp') else 'legacy'
    photo_contrast,red_tolerance=55,6
    if mask_mode=='photo_red':
        st.caption("Chế độ này xử lý màu và nền cục bộ bên trong bộ dò; không sửa ảnh gốc, không bật tăng tương phản cho ảnh gửi AI. Có thể làm mất nét mảnh hoặc nét có màu.")
        with st.expander("Thiết lập ảnh chụp lưới đỏ"):
            photo_contrast=st.slider("Mức tối tương đối tối thiểu",25,100,55,key="dg_photo_dark_"+frame)
            red_tolerance=st.slider("Dung sai sắc đỏ",2,20,6,key="dg_photo_red_"+frame)
            st.caption("Giữ mặc định cho lần thử đầu. Nới dung sai có thể giữ thêm sóng và thêm lưới. Không tự nối đoạn mất.")
    with st.expander("Thiết lập cách 0.3 (chỉ áp dụng khi chọn ảnh sạch)"):

        suppress_color = st.checkbox("Nét đen: giảm lưới màu",value=True,key="dg_color_"+frame)
        remove_grid = st.checkbox("Thử gỡ đường lưới thẳng dài",value=True,key="dg_grid_"+frame)
        st.caption("Gỡ lưới có thể làm mất cả đoạn ECG trùng lưới. Hãy thử tắt nếu đường màu bị đứt; không tự điền đoạn đã mất.")
        max_gray = st.slider("Ngưỡng tối tối đa",30,245,185,key="dg_gray_"+frame)
        local_contrast = st.slider("Tương phản cục bộ tối thiểu",2,35,10,key="dg_contrast_"+frame)
        continuity = st.slider("Ưu tiên đường liên tục",.02,.4,.12,.01,key="dg_cont_"+frame)
        seed_percent = st.slider("Gợi ý độ cao bắt đầu trong vùng (%)",0.,100.,50.,1.,key="dg_seed_"+frame)
    params = dict(max_gray=max_gray,local_contrast=local_contrast,suppress_color=suppress_color,
                  remove_grid=remove_grid,continuity=continuity,seed_percent=seed_percent,
                  mask_mode=mask_mode,photo_contrast=photo_contrast,red_tolerance=red_tolerance)
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
                "quality":candidate["quality"],
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
    background=meta.get('quality',{})
    if background.get('reasons'):
        st.info('Nền và toàn bộ nét ứng viên: '+ ' '.join(background['reasons'])+
                ' Đây là thông tin toàn vùng; đánh giá lựa chọn xuất nằm dưới bước 4.')
    st.write(f"Thời gian: {meta['seconds']:.2f} giây · {gs['segments']} đoạn nét · {gs['junctions']} điểm phân nhánh")
    st.caption("Số đoạn/pixel không phải độ chính xác. Giữ thêm nét dọc có thể đồng thời giữ thêm lưới hoặc chữ.")
    if gs["ambiguous_components"]:
        st.warning("Có thành phần mơ hồ màu cam. Chưa thể quyết định nhánh nào là ECG; không dùng để đo.")
    st.warning("Chưa có tín hiệu thời gian hay các mốc P/QRS/T. Đường nằm trên mực thật vẫn có thể là lưới/chữ. Kiểm tra riêng đỉnh R và đáy S.")
    st.subheader("4. Phóng to và chọn nhánh nét")
    reviewed_graph,review_meta,review_overlay = review_panel(result)
    meta = dict(meta)
    meta["review"] = review_meta
    quality=selected_quality(result['graph'],review_meta['selected_segment_ids'],tuple(meta['working_size']))
    meta['background_quality_v04']=meta.pop('quality',{})
    meta['quality']=quality
    meta['selection_quality']=quality
    meta['extraction_baseline']='Digitizer 0.4 masks and graph; unchanged extraction'
    st.subheader('Đánh giá các đoạn đang giữ')
    if quality['status']=='not_ready':
        st.error('Lựa chọn chưa đạt để tiếp tục đo: '+' '.join(quality['reasons']))
    else:
        st.warning('Chưa thấy lỗi lớn theo các ngưỡng kỹ thuật đang dùng; vẫn cần đối chiếu từng sóng. Không xác nhận đúng ECG hoặc cho phép đo.')
    st.write(f"{quality['selected_segments']} đoạn giữ · {quality['connected_components']} thành phần rời · {quality['junctions']} điểm phân nhánh · khoảng thiếu cột dài nhất {quality['longest_internal_missing_x_px']} px")
    st.caption('Nhiều điểm cùng x có thể là sườn QRS dốc. Các chỉ số này không đánh giá đầy đủ nét sóng hoặc thay thế người kiểm tra.')
    meta["validation_status"] = "not_ready_diagnostic_only" if quality.get("status")=="not_ready" else "manual_selection_unverified"
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
    st.write("1. Ảnh sạch: dùng cách 0.3 để đối chiếu. Ảnh chụp lưới đỏ: chọn chế độ tương ứng, thử mặc định với 3–5 nhát của một chuyển đạo.")
    st.write("2. Chọn cửa sổ quanh QRS. Dùng bộ lọc Nhánh ngắn cần xem; giữ/bỏ chỉ khi thấy rõ trên ảnh gốc. Đánh giá các đoạn đang giữ cập nhật sau mỗi thao tác.")
    st.write("3. Thử lại đúng vùng cắt và tham số của 0.4, xuất đủ 4 file trước khi chỉnh nhánh. Nếu chỉnh tay, xuất lượt riêng sau chỉnh. Chưa cần ảnh mới hoặc đo PR/QRS/ST.")
    st.caption("Chọn nhánh chỉ là thao tác đối chiếu của người dùng, không phải xác nhận tín hiệu đúng. Khoảng mất không được nối tự động.")



if __name__ == "__main__":
    main()
