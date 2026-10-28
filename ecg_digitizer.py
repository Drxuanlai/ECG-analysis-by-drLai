"""ECG digitization engine for ECG-analysis-by-drLai.

Phase 2 / v1.0:
- image orientation + deskew
- red-grid region detection
- standard 3x4 lead layout with waveform-energy refinement
- optional label anchors hook (OCR/vision can provide these later)
- per-lead centerline extraction
- automatic grid-spacing calibration (geometry only)
- PASS/REVIEW/FAIL engineering QC
- no clinical diagnosis and no unproven ms/mV conversion

Dependencies: numpy, opencv-python-headless, Pillow.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import cv2
import numpy as np
from PIL import Image, ImageOps

VERSION = "ECG Digitizer Engine 1.2-cross-axis-calibration"
STANDARD_12_LAYOUT = [
    ["I", "aVR", "V1", "V4"],
    ["II", "aVL", "V2", "V5"],
    ["III", "aVF", "V3", "V6"],
]
LEADS_12 = [x for row in STANDARD_12_LAYOUT for x in row]

def _as_rgb(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    if isinstance(image, np.ndarray):
        a=image
        if a.ndim==2: return Image.fromarray(a.astype(np.uint8)).convert("RGB")
        if a.shape[2]==4: a=cv2.cvtColor(a,cv2.COLOR_BGRA2RGB)
        elif a.shape[2]==3: a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB)
        return Image.fromarray(a.astype(np.uint8))
    raise TypeError("image must be PIL.Image or numpy array")

def orient(image: Image.Image, angle: int) -> Image.Image:
    turns={90:Image.Transpose.ROTATE_90,180:Image.Transpose.ROTATE_180,270:Image.Transpose.ROTATE_270}
    return image.transpose(turns[angle]) if angle in turns else image.copy()

def detect_paper_bbox(image: Image.Image) -> Tuple[int,int,int,int]:
    rgb=np.asarray(image.convert("RGB")); h,w=rgb.shape[:2]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    sat=rgb.max(2).astype(np.int16)-rgb.min(2).astype(np.int16)
    mask=((gray>140)&(sat<90)).astype(np.uint8)*255
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((max(7,h//100),max(7,w//100)),np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    best=None; score=-1
    for c in contours:
        x,y,bw,bh=cv2.boundingRect(c); area=bw*bh
        if area<.08*w*h: continue
        fill=float((mask[y:y+bh,x:x+bw]>0).mean())
        if area*fill>score: score=area*fill; best=(x,y,x+bw,y+bh)
    return best or (0,0,w,h)

def auto_orient_page(image: Image.Image):
    cand=[]
    for angle in (0,90,180,270):
        im=orient(image,angle); b=detect_paper_bbox(im)
        pw=max(1,b[2]-b[0]); ph=max(1,b[3]-b[1])
        aspect=pw/ph; area=pw*ph/(im.width*im.height)
        landscape=min(2.5,max(0.,aspect))/2.5
        s=2*landscape+1.2*area-{0:0,90:.01,180:.04,270:.02}[angle]
        cand.append((s,angle,im,b))
    cand.sort(reverse=True,key=lambda z:z[0])
    _,a,im,b=cand[0]
    return im,a,b

def auto_deskew(image: Image.Image):
    rgb=np.asarray(image.convert("RGB")); gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    scale=min(1.,1600/max(1,gray.shape[1]))
    small=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA) if scale<1 else gray
    edges=cv2.Canny(small,60,160)
    lines=cv2.HoughLinesP(edges,1,np.pi/180,threshold=max(70,small.shape[1]//9),
                          minLineLength=max(50,small.shape[1]//5),maxLineGap=25)
    angles=[]
    if lines is not None:
        for x1,y1,x2,y2 in np.asarray(lines).reshape(-1,4):
            a=np.degrees(np.arctan2(float(y2-y1),float(x2-x1)))
            if -10<=a<=10: angles.append(a)
    angle=float(np.median(angles)) if len(angles)>=2 else 0.
    if abs(angle)<.2: return image.copy(),0.
    h,w=gray.shape; m=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    out=cv2.warpAffine(rgb,m,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(out),angle

def detect_grid_bbox(image: Image.Image, paper_box=None):
    """Find the red/pink ECG-grid region. Falls back to paper bbox with low confidence."""
    rgb=np.asarray(image.convert("RGB")); h,w=rgb.shape[:2]
    x0,y0,x1,y1=paper_box or detect_paper_bbox(image)
    roi=rgb[y0:y1,x0:x1]
    r=roi[:,:,0].astype(np.int16); g=roi[:,:,1].astype(np.int16); b=roi[:,:,2].astype(np.int16)
    # Pink/red grid: red channel exceeds green/blue modestly; avoid very dark trace/text.
    red=((r-g)>8)&((r-b)>5)&(r>105)&(g>70)
    m=(red.astype(np.uint8)*255)
    kh=max(5,roi.shape[0]//120); kw=max(5,roi.shape[1]//120)
    m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((kh,kw),np.uint8))
    # Projection is robust to folds and black ECG ink.
    xs=np.where((m>0).mean(0)>.035)[0]; ys=np.where((m>0).mean(1)>.035)[0]
    if xs.size>20 and ys.size>20:
        gx0,gx1=int(np.percentile(xs,1)),int(np.percentile(xs,99))
        gy0,gy1=int(np.percentile(ys,1)),int(np.percentile(ys,99))
        if (gx1-gx0)>.45*roi.shape[1] and (gy1-gy0)>.35*roi.shape[0]:
            box=(x0+gx0,y0+gy0,x0+gx1+1,y0+gy1+1)
            return box, float(min(1., (m[gy0:gy1+1,gx0:gx1+1]>0).mean()*8))
    return (x0,y0,x1,y1),0.15

def _dark_signal_score(gray):
    # suppress slowly varying background; reward dark trace-like pixels
    blur=cv2.GaussianBlur(gray,(0,0),3)
    d=np.clip(blur.astype(np.float32)-gray.astype(np.float32),0,None)
    return cv2.GaussianBlur(d,(3,3),0)

def _refine_row_centers(image: Image.Image, grid_box):
    rgb=np.asarray(image.convert("RGB")); x0,y0,x1,y1=grid_box
    gray=cv2.cvtColor(rgb[y0:y1,x0:x1],cv2.COLOR_RGB2GRAY)
    sig=_dark_signal_score(gray)
    prof=np.percentile(sig,85,axis=1)
    prof=cv2.GaussianBlur(prof.reshape(-1,1),(1,max(9,(len(prof)//35)|1)),0).ravel()
    # Expected centers for 3 diagnostic rows + rhythm row; search locally.
    expected=np.array([.20,.42,.64,.84])*(len(prof)-1)
    centers=[]
    radius=max(10,int(len(prof)*.10))
    for e in expected:
        a=max(0,int(e)-radius); b=min(len(prof),int(e)+radius+1)
        centers.append(int(a+np.argmax(prof[a:b])) if b>a else int(e))
    # enforce order/min separation; fallback to expected if unstable
    minsep=max(8,int(len(prof)*.11))
    if any(centers[i+1]-centers[i]<minsep for i in range(3)):
        centers=[int(v) for v in expected]
    return [y0+c for c in centers]

def _refine_col_edges(image: Image.Image, grid_box):
    # Standard print is 4 equal time blocks; refine only a small global offset using vertical activity.
    x0,y0,x1,y1=grid_box; width=x1-x0
    left=x0+int(.02*width); right=x1-int(.015*width)
    return [int(left+i*(right-left)/4) for i in range(5)]

def make_lead_boxes(image: Image.Image, grid_box, label_anchors: Optional[Mapping[str,Tuple[float,float]]]=None):
    """Create lead boxes from grid geometry and waveform row energy.
    label_anchors may contain lead -> (x,y) from OCR/vision. Anchors refine row/column centers.
    """
    x0,y0,x1,y1=grid_box; gh=y1-y0
    rows=_refine_row_centers(image,grid_box); cols=_refine_col_edges(image,grid_box)
    # OCR/vision hook: median anchor positions by known row/col.
    if label_anchors:
        for r,names in enumerate(STANDARD_12_LAYOUT):
            ys=[label_anchors[n][1] for n in names if n in label_anchors]
            if ys: rows[r]=int(np.median(ys)+.08*gh)  # label normally sits slightly above baseline
        for c in range(4):
            xs=[label_anchors[STANDARD_12_LAYOUT[r][c]][0] for r in range(3)
                if STANDARD_12_LAYOUT[r][c] in label_anchors]
            if xs and c<4:
                # anchor marks left side of cell; preserve width from geometric edges
                shift=int(np.median(xs))-cols[c]
                if abs(shift)<.10*(x1-x0): cols[c]+=shift
    # derive row boundaries halfway between centers; top/bottom conservatively around rows
    mids=[int((rows[i]+rows[i+1])/2) for i in range(3)]
    top=max(y0, rows[0]-int(.105*gh)); rbottom=min(y1, rows[2]+int(.105*gh))
    rb=[top,mids[0],mids[1],rbottom]
    boxes=[]
    for r,names in enumerate(STANDARD_12_LAYOUT):
        cy0,cy1=rb[r],rb[r+1]
        for c,name in enumerate(names):
            cw=cols[c+1]-cols[c]
            bx0=cols[c]+int(.04*cw); bx1=cols[c+1]-int(.015*cw)
            pad=int(.04*(cy1-cy0))
            boxes.append({"lead":name,"box":(bx0,cy0+pad,bx1,cy1-pad),
                          "row":r,"col":c,"kind":"diagnostic"})
    # rhythm strip uses fourth refined row
    half=max(int(.075*gh), int((rows[2]-rows[1])*.42))
    boxes.append({"lead":"II-rhythm","box":(cols[0],max(y0,rows[3]-half),cols[-1],min(y1,rows[3]+half)),
                  "row":3,"col":0,"kind":"rhythm"})
    return boxes

def photo_ink_mask(image: Image.Image, contrast=55, red_tolerance=6):
    rgb=np.asarray(image.convert("RGB")).astype(np.float32)
    bg=cv2.morphologyEx(rgb,cv2.MORPH_CLOSE,np.ones((21,21),np.uint8))
    bg=cv2.GaussianBlur(bg,(21,21),0)
    normalized=np.clip(rgb/np.maximum(bg,1)*255,0,255).max(2)
    red_excess=rgb[:,:,0]-np.maximum(rgb[:,:,1],rgb[:,:,2])
    darkness=normalized<255-int(contrast)
    color_ok=(red_excess<float(red_tolerance))|(normalized<150)
    return darkness&color_ok,normalized.astype(np.uint8)

def trace_centerline(mask, gray, continuity=.12, seed_percent=50.):
    h,w=mask.shape
    if not np.any(mask): raise ValueError("no dark ink for centerline")
    ink=mask.astype(bool); rows=np.arange(h,dtype=np.float32); idx=np.arange(h,dtype=np.int32)
    parent=np.empty((w,h),dtype=np.uint16 if h<65535 else np.uint32)
    hk=max(15,min(w//18,61)); vk=max(9,min(h//4,51))
    horizontal=cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_OPEN,np.ones((1,hk),np.uint8)).astype(bool)
    vertical=cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_OPEN,np.ones((vk,1),np.uint8)).astype(bool)
    grid_like=horizontal|vertical
    darkness=1-gray.astype(np.float32)/255
    local=cv2.boxFilter(ink.astype(np.float32),-1,(5,3),normalize=True)
    seed=float(seed_percent)*(h-1)/100
    def emission(x):
        e=np.where(ink[:,x],1.05-.85*darkness[:,x]-.25*local[:,x],9.)
        return (e+.55*grid_like[:,x]).astype(np.float32)
    cost=emission(0)+continuity*.15*np.abs(rows-seed); parent[0]=idx
    for x in range(1,w):
        fr=cost-continuity*rows; fm=np.minimum.accumulate(fr)
        fp=np.maximum.accumulate(np.where(fr<=fm,idx,0)); f=fm+continuity*rows
        br=(cost+continuity*rows)[::-1]; bm=np.minimum.accumulate(br)
        bp_rev=np.maximum.accumulate(np.where(br<=bm,idx,0)); bp=(h-1-bp_rev)[::-1]
        b=bm[::-1]-continuity*rows; use=f<=b
        parent[x]=np.where(use,fp,bp); cost=np.minimum(f,b)+emission(x)
    path=np.empty(w,dtype=np.int32); path[-1]=int(np.argmin(cost))
    for x in range(w-1,0,-1): path[x-1]=int(parent[x,path[x]])
    run_count=ink[0].astype(np.int16)+np.sum(ink[1:]&~ink[:-1],axis=0).astype(np.int16)
    span=np.zeros(w,np.int32); snapped=path.copy(); supported=np.zeros(w,bool)
    rad=max(2,min(5,h//80+2))
    for x,y in enumerate(path):
        ys=np.flatnonzero(ink[:,x])
        if not ys.size: continue
        near=int(ys[np.argmin(np.abs(ys-y))])
        if abs(near-y)>rad: continue
        supported[x]=True; lo=hi=near
        while lo>0 and ink[lo-1,x]: lo-=1
        while hi+1<h and ink[hi+1,x]: hi+=1
        span[x]=hi-lo+1; snapped[x]=(lo+hi)//2
    base=float(np.median(run_count[supported])) if supported.any() else 0
    mad=float(np.median(np.abs(run_count[supported]-base))) if supported.any() else 0
    mt=max(5.,base+max(2.,2.5*mad))
    unusual=supported&(run_count>mt); vert=supported&(span>max(14,int(h*.22)))
    status=np.full(w,"centerline_candidate",object); status[~supported]="no_ink_support"
    status[unusual]="ambiguous_multiple_ink_runs"; status[vert]="ambiguous_near_vertical_stroke"
    edge=supported&((snapped<=1)|(snapped>=h-2)); status[edge]="touches_crop_edge"
    conf=np.zeros(w,np.float32); good=np.flatnonzero(supported)
    if good.size:
        yy=snapped[good]; c=.35+.55*darkness[yy,good]
        c-=.025*np.minimum(6,np.maximum(0,run_count[good]-mt))
        c-=.30*(span[good]>max(14,int(h*.22))); c-=.12*grid_like[yy,good]
        conf[good]=np.clip(c,.05,.99)
    values=snapped.astype(float); values[~supported]=np.nan
    missing=np.flatnonzero(~supported); longest=0
    if missing.size:
        longest=max(map(len,np.split(missing,np.where(np.diff(missing)>1)[0]+1)))
    stats={"supported_fraction":float(supported.mean()),
           "low_confidence_fraction":float(np.mean(supported&(conf<.35))),
           "median_confidence_supported":float(np.median(conf[supported])) if supported.any() else 0.,
           "longest_missing_run_px":int(longest),"near_vertical_columns":int(vert.sum()),
           "touches_edge_columns":int(edge.sum())}
    return {"y":values,"confidence":conf,"status":status,"supported":supported,"stats":stats}

def _autocorr_peaks(profile, min_lag=3, max_lag=None):
    p=np.asarray(profile,float); p=p-np.mean(p)
    if np.std(p)<1e-6: return []
    max_lag=min(max_lag or len(p)//6,len(p)//3)
    if max_lag<=min_lag: return []
    ac=np.correlate(p,p,mode="full")[len(p)-1:]; ac/=max(ac[0],1e-9)
    out=[]
    for lag in range(max(min_lag,2),max_lag):
        if ac[lag]>ac[lag-1] and ac[lag]>=ac[lag+1] and ac[lag]>=.08:
            out.append((float(ac[lag]),float(lag)))
    return sorted(out,reverse=True)

def _fundamental_from_peaks(peaks):
    """Infer 1-mm grid spacing when autocorrelation may prefer 5-mm bold lines."""
    if not peaks: return None,0.,[]
    candidates=[]
    for score,lag in peaks[:12]:
        for div in (1,2,3,4,5):
            f=lag/div
            if f>=3: candidates.append(f)
    best=None
    for f in candidates:
        support=0.; matches=[]
        for score,lag in peaks[:16]:
            k=max(1,round(lag/f)); err=abs(lag-k*f)/max(f,1e-6)
            if k<=8 and err<.16:
                support+=score*(1-err)/np.sqrt(k); matches.append((lag,k,score))
        # prefer a fundamental that explains several harmonics, not a tiny alias
        if len(matches)>=2:
            value=(support+.08*min(f,20),f,matches)
            if best is None or value[0]>best[0]: best=value
    if best is None:
        score,lag=max(peaks); return lag,score,[(lag,1,score)]
    return float(best[1]),float(min(1,best[0]/2)),best[2]

def _direct_cross_axis_period(px, py):
    """Prefer a strong observed period shared by X and Y before harmonic division."""
    matches=[]
    for lx,sx in px:
        for ly,sy in py:
            ratio=max(lx,ly)/max(1e-6,min(lx,ly))
            if ratio<=1.12:
                strength=min(float(sx),float(sy))
                period=(float(lx)+float(ly))/2.0
                score=2.5*strength + 1.2*(1-(ratio-1)/.12)
                matches.append((score,period,float(lx),float(ly),strength,ratio))
    if not matches:return None
    strong=[m for m in matches if m[4]>=.18]
    return max(strong or matches,key=lambda z:z[0])

def estimate_grid_calibration(image: Image.Image, grid_box):
    """Robust 1-mm grid estimate with X/Y consensus before harmonic assignment."""
    rgb=np.asarray(image.convert("RGB")); x0,y0,x1,y1=grid_box
    roi=rgb[y0:y1,x0:x1]
    r=roi[:,:,0].astype(float); g=roi[:,:,1].astype(float); b=roi[:,:,2].astype(float)
    red=np.clip(r-(g+b)/2,0,None)
    xp=np.percentile(red,78,axis=0); yp=np.percentile(red,78,axis=1)
    maxx=min(160,max(10,len(xp)//10)); maxy=min(160,max(10,len(yp)//6))
    px=_autocorr_peaks(xp,3,maxx); py=_autocorr_peaks(yp,3,maxy)
    fx,cx,mx=_fundamental_from_peaks(px); fy,cy,my=_fundamental_from_peaks(py)

    status="FAIL"; pxmm=None; conf=0.; ratio=None; method=None
    observed_x=fx; observed_y=fy
    direct=_direct_cross_axis_period(px,py)
    if direct is not None:
        _,period,dx,dy,dconf,dratio=direct
        if 3<=period<=80:
            pxmm=period; observed_x=dx; observed_y=dy
            ratio=dratio; conf=dconf; method="direct_cross_axis"
            status="PASS" if dratio<=1.08 and dconf>=.20 else "REVIEW"

    # Only if no credible shared observed period exists, fall back to harmonic families.
    if pxmm is None and fx and fy:
        pairs=[]
        for ax in (1,2,3,4,5,6):
            for ay in (1,2,3,4,5,6):
                aa=fx/ax; bb=fy/ay
                if aa<3 or bb<3: continue
                rr=max(aa,bb)/max(1e-6,min(aa,bb))
                complexity=(ax-1)+(ay-1)
                score=rr + .025*complexity
                pairs.append((score,rr,aa,bb,ax,ay))
        if pairs:
            _,rr,aa,bb,ax,ay=min(pairs,key=lambda z:z[0])
            conf=float(min(cx,cy))
            if rr<=1.18 and 3<=np.mean([aa,bb])<=80:
                pxmm=float(np.mean([aa,bb])); ratio=rr; method="harmonic_cross_axis"
                status="PASS" if rr<=1.10 and conf>=.22 else "REVIEW"
            else:
                status="REVIEW"

    return {"status":status,"small_grid_px_x":observed_x,"small_grid_px_y":observed_y,
            "px_per_mm":pxmm,"axis_ratio":ratio,"confidence":conf,
            "consensus_method":method,"x_harmonics":mx,"y_harmonics":my,
            "speed_mm_s":None,"gain_mm_mV":None,"gain_by_lead":{},
            "time_ms_per_px":None,"mV_per_px_by_lead":{},
            "note":"1-mm grid geometry with X/Y consensus; speed/gain require OCR, calibration pulse, or user confirmation."}

def _normalize_recording_hints(recording_hints=None, speed_mm_s=None, gain_mm_mV=None):
    h=dict(recording_hints or {})
    if speed_mm_s is not None: h["speed_mm_s"]=speed_mm_s
    if gain_mm_mV is not None: h["gain_mm_mV"]=gain_mm_mV
    return h

def _gain_map_from_hints(hints):
    gains={}
    src={}
    by=hints.get("gain_by_lead") or {}
    for lead,v in by.items():
        if lead in LEADS_12 and v in (2.5,5,10,20): gains[lead]=float(v); src[lead]="ocr_or_user_per_lead"
    limb=hints.get("limb_gain_mm_mV"); chest=hints.get("precordial_gain_mm_mV")
    if limb in (2.5,5,10,20):
        for lead in ("I","II","III","aVR","aVL","aVF"): gains.setdefault(lead,float(limb)); src.setdefault(lead,"ocr_or_user_group")
    if chest in (2.5,5,10,20):
        for lead in ("V1","V2","V3","V4","V5","V6"): gains.setdefault(lead,float(chest)); src.setdefault(lead,"ocr_or_user_group")
    global_gain=hints.get("gain_mm_mV")
    if global_gain in (2.5,5,10,20):
        for lead in LEADS_12: gains.setdefault(lead,float(global_gain)); src.setdefault(lead,"user_global")
    return gains,src

def detect_calibration_pulse_gain(image, boxes, px_per_mm):
    """Conservative pulse-height detector. It proposes gains only when a rectangular
    calibration-like vertical excursion near a cell edge has a plausible 1-mV height.
    This is evidence, not proof; callers should retain source/confidence.
    """
    if not px_per_mm: return {},{}
    rgb=np.asarray(image.convert("RGB")); proposals={}; confs={}
    for item in boxes:
        if item.get("kind")!="diagnostic": continue
        lead=item["lead"]; x0,y0,x1,y1=item["box"]; w=x1-x0; h=y1-y0
        # Search edge zones because calibration pulses usually border lead blocks.
        zones=[(max(0,x0-int(.05*w)),min(rgb.shape[1],x0+int(.16*w))),
               (max(0,x1-int(.16*w)),min(rgb.shape[1],x1+int(.05*w)))]
        best=None
        for a,b in zones:
            if b-a<6: continue
            gray=cv2.cvtColor(rgb[y0:y1,a:b],cv2.COLOR_RGB2GRAY)
            dark=(gray<135).astype(np.uint8)
            # vertical dark runs; text tends to be shorter and fragmented
            col=dark.sum(0); j=int(np.argmax(col)); ys=np.flatnonzero(dark[:,j])
            if ys.size<5: continue
            height=float(ys.max()-ys.min()+1); mm=height/px_per_mm
            # Candidate gains are pulse height in mm for a standard 1-mV pulse.
            nearest=min((2.5,5.,10.,20.),key=lambda g:abs(g-mm))
            rel=abs(nearest-mm)/nearest
            score=max(0.,1-rel/.35)*min(1.,ys.size/max(8,height*.45))
            if rel<=.35 and (best is None or score>best[0]): best=(score,nearest,mm)
        if best and best[0]>=.45:
            proposals[lead]=float(best[1]); confs[lead]=float(best[0])
    return proposals,confs

def apply_recording_standard(calibration, recording_hints=None, pulse_gains=None, pulse_confidence=None):
    c=dict(calibration); hints=dict(recording_hints or {})
    speed=hints.get("speed_mm_s")
    if speed not in (None,25,50): raise ValueError("speed must be 25, 50, or None")
    gains,sources=_gain_map_from_hints(hints)
    for lead,g in (pulse_gains or {}).items():
        # OCR/user wins; pulse fills only missing leads.
        if lead not in gains:
            gains[lead]=float(g); sources[lead]="calibration_pulse"
    pxmm=c.get("px_per_mm"); c["speed_mm_s"]=speed
    c["speed_source"]=hints.get("speed_source","ocr_or_user" if speed else None)
    c["gain_by_lead"]=gains; c["gain_source_by_lead"]=sources
    c["gain_confidence_by_lead"]={lead:float((pulse_confidence or {}).get(lead,1.0 if sources.get(lead," ").startswith("ocr_or_user") else 0.0)) for lead in gains}
    c["time_ms_per_px"]=(1000/(pxmm*speed)) if pxmm and speed else None
    c["mV_per_px_by_lead"]={lead:1/(pxmm*g) for lead,g in gains.items()} if pxmm else {}
    c["recording_standard_status"]=("PASS" if pxmm and speed and len(gains)>=12 else
                                    "REVIEW" if pxmm and (speed or gains) else "UNRESOLVED")
    return c

def lead_qc(centerline, box, grid_box):
    """Engineering QC. FAIL is reserved for gross extraction failure.
    Photographed thin traces can have intermittent pixel support even when the DP
    centerline is visually coherent, so support alone triggers REVIEW, not FAIL.
    """
    s=centerline["stats"]; reasons=[]
    if s["supported_fraction"]<.78: reasons.append("low_ink_support")
    if s["median_confidence_supported"]<.45: reasons.append("low_median_confidence")
    gap_limit=max(12,int(len(centerline["y"])*.035))
    if s["longest_missing_run_px"]>gap_limit: reasons.append("long_missing_run")
    if s["touches_edge_columns"]>max(10,int(len(centerline["y"])*.02)): reasons.append("crop_edge_contact")
    severe=(s["supported_fraction"]<.45 or
            s["median_confidence_supported"]<.30 or
            s["longest_missing_run_px"]>max(30,int(len(centerline["y"])*.10)))
    status="FAIL" if severe else ("REVIEW" if reasons else "PASS")
    return {"status":status,"reasons":reasons,**s}

def overall_qc(leads, calibration, layout_confidence):
    statuses=[v["qc"]["status"] for v in leads.values()]
    fail=statuses.count("FAIL"); review=statuses.count("REVIEW")
    reasons=[]
    if fail: reasons.append(f"{fail}_lead_failed")
    if review: reasons.append(f"{review}_lead_review")
    if layout_confidence<.35: reasons.append("layout_low_confidence")
    if calibration["status"]=="FAIL": reasons.append("grid_calibration_failed")
    # Digitization can still be usable in pixel space when calibration fails.
    status="FAIL" if fail>=3 else ("REVIEW" if fail or review or layout_confidence<.5 or calibration["status"]!="PASS" else "PASS")
    return {"status":status,"reasons":reasons,"lead_pass":statuses.count("PASS"),
            "lead_review":review,"lead_fail":fail}

def digitize_ecg(image: Any, *, label_anchors=None, recording_hints=None,
                 speed_mm_s=None, gain_mm_mV=None, include_rhythm=True, deskew=True) -> Dict[str,Any]:
    """Main API used by app.py. Never invents speed/gain.
    label_anchors: optional dict {lead:(x,y)} from a real OCR/vision layer.
    """
    src=_as_rgb(image); oriented,rotation,paper=auto_orient_page(src)
    deskew_deg=0.
    if deskew:
        oriented,deskew_deg=auto_deskew(oriented); paper=detect_paper_bbox(oriented)
    grid_box,grid_conf=detect_grid_bbox(oriented,paper)
    boxes=make_lead_boxes(oriented,grid_box,label_anchors)
    calibration=estimate_grid_calibration(oriented,grid_box)
    hints=_normalize_recording_hints(recording_hints,speed_mm_s,gain_mm_mV)
    pulse_gains,pulse_conf=detect_calibration_pulse_gain(oriented,boxes,calibration.get("px_per_mm"))
    calibration=apply_recording_standard(calibration,hints,pulse_gains,pulse_conf)
    leads={}
    targets=boxes if include_rhythm else [b for b in boxes if b["kind"]=="diagnostic"]
    for item in targets:
        x0,y0,x1,y1=item["box"]; crop=oriented.crop((x0,y0,x1,y1))
        work=crop.copy(); work.thumbnail((2000,800),Image.Resampling.LANCZOS)
        try:
            mask,gray=photo_ink_mask(work)
            cl=trace_centerline(mask,gray)
            qc=lead_qc(cl,item["box"],grid_box)
            y=cl["y"].tolist(); conf=cl["confidence"].tolist()
            leads[item["lead"]]={"box":list(item["box"]),"work_size":list(work.size),
                "signal_px":y,"confidence":conf,"status":[str(v) for v in cl["status"]],
                "qc":qc,"error":None}
        except Exception as exc:
            leads[item["lead"]]={"box":list(item["box"]),"work_size":list(work.size),
                "signal_px":[],"confidence":[],"status":[],"qc":{"status":"FAIL","reasons":["extraction_error"]},
                "error":str(exc)}
    oq=overall_qc({k:v for k,v in leads.items() if k!="II-rhythm"},calibration,grid_conf)
    return {"version":VERSION,"rotation_ccw":rotation,"deskew_degrees":deskew_deg,
            "oriented_size":list(oriented.size),"paper_box":list(paper),"grid_box":list(grid_box),
            "layout":{"method":"grid+waveform_energy"+("+label_anchors" if label_anchors else ""),
                      "confidence":grid_conf,"label_anchors_used":sorted(label_anchors or {})},
            "calibration":calibration,"leads":leads,"overall_qc":oq}

def result_summary(result):
    return {"version":result["version"],"overall_qc":result["overall_qc"],
            "layout":result["layout"],"calibration":result["calibration"],
            "leads":{k:{"qc":v["qc"],"box":v["box"],"points":len(v["signal_px"]),"error":v["error"]}
                     for k,v in result["leads"].items()}}
