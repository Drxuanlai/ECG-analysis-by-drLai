"""Research-only 1-D ECG candidate reconstruction from a doctor-confirmed ROI.

Never infers ECG events or physical units. A candidate curve is accompanied by
per-column ink support, uncertainty and explicit gaps. Quality flags are not
clinical validation. This component deliberately ignores the old Rhythm clock.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image, ImageDraw

VERSION = 'ecg-core-waveform-reconstruction-v1.4.0'


def _odd(n: int) -> int:
    return max(3, int(n) | 1)


def _prep(image: Image.Image, max_width=1100, max_height=270):
    im = image.convert('RGB')
    orig = im.size
    if min(orig) < 20:
        raise ValueError('ROI too small')
    im.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    rgb = np.asarray(im, dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h,w=gray.shape
    bg=cv2.morphologyEx(gray,cv2.MORPH_CLOSE, np.ones((_odd(min(h,w)//11),)*2, np.uint8))
    bg=cv2.GaussianBlur(bg,(0,0),max(2.,min(h,w)/30))
    # Relative darkness resists a dark gray/uneven photo background.
    dark = np.clip((bg-gray)*255. / np.maximum(bg,45),0,140)
    red=rgb[:,:,0].astype(float)-(rgb[:,:,1].astype(float)+rgb[:,:,2].astype(float))/2
    redpen=np.clip((red-3)/20,0,1).astype(np.float32)
    normalized = np.clip(gray / np.maximum(bg, 35)*215,0,255).astype(np.uint8)
    return im, rgb, dark, redpen, normalized, orig


def _grid_penalty(dark,redpen):
    h,w=dark.shape
    # Long straight lines are down-weighted, never deleted: true ECG crossings
    # can coincide with grid lines.
    cutoff=max(5.0,float(np.percentile(dark,78)))
    binary=(dark>cutoff).astype(np.uint8)
    kh=max(15,w//11); kv=max(13,h//4)
    horizontal=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((1,kh),np.uint8)).astype(bool)
    vertical=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((kv,1),np.uint8)).astype(bool)
    return np.maximum(redpen,np.maximum(horizontal,vertical).astype(np.float32)*.65)


def reconstruct(image: Image.Image, *, min_evidence=0.40, max_jump_fraction=0.25):
    """Trace a *candidate* centerline and return non-interpolated point evidence.

    Accepted points are always anchored to a dark/neutral pixel. Unsupported
    columns are null. No temporal or voltage calibration is inferred.
    """
    im,rgb,dark,redpen,normalized,(ow,oh)=_prep(image)
    h,w=dark.shape
    grid=_grid_penalty(dark,redpen)
    # Ink strength: strong local dark contrasts, red grid down-weighted.
    strength=np.clip((dark-3.5)/26,0,1)*np.clip(1-.80*redpen,0,1)
    # Small vertical smoothing preserves thin QRS, and helps textured ink.
    strength=cv2.GaussianBlur(strength.astype(np.float32),(3,3),0.55)
    candidate=(strength>=max(.18,min_evidence*.55))&(grid<.92)
    # Prefer a stable local baseline only as a weak prior, never a forced path.
    row_score=np.mean(np.minimum(strength,.65),axis=1)
    ys=np.arange(h,dtype=np.float32)
    # Bias to central band avoids OCR text at top and calibration near margins.
    prior=row_score-.07*np.abs(ys-h*.5)/max(h,1)
    cy=int(np.argmax(cv2.GaussianBlur(prior[:,None],(1,_odd(max(5,h//14))),0).ravel()))
    maxjump=max(4,round(h*max_jump_fraction))
    jumps=np.unique(np.clip(np.rint(np.array([-maxjump,-maxjump*.65,-maxjump*.38,-maxjump*.18,-8,-4,-2,-1,0,1,2,4,8,maxjump*.18,maxjump*.38,maxjump*.65,maxjump])), -maxjump,maxjump).astype(int))
    # Cost of the path through unsupported columns is recorded separately;
    # this optimization ONLY proposes where an ink trace might lie.
    emission=1.5*(1-strength)+.55*grid + .05*np.abs(ys[:,None]-cy)/max(1,h)
    emission=emission.astype(np.float32)
    scores=emission[:,0].copy();parents=np.empty((w,h),np.int16 if h<32768 else np.int32)
    parents[0,:]=np.arange(h)
    ids=np.arange(h)
    for x in range(1,w):
        nxt=np.full(h,np.inf,np.float32); back=np.full(h,-1,np.int32)
        for step in jumps:
            # path y(now)=y(previous)+step
            if step>=0:
                dest=slice(step,h);src=slice(0,h-step)
            else:
                dest=slice(0,h+step);src=slice(-step,h)
            move = (.013*abs(step)+.00045*(step**2))
            proposal=scores[src]+move
            current=nxt[dest]
            best=proposal<current
            nxt[dest]=np.minimum(current,proposal)
            b=back[dest];b[best]=ids[src][best];back[dest]=b
        scores=nxt+emission[:,x]
        parents[x,:]=back
    path=np.empty(w,np.int32);path[-1]=int(np.argmin(scores))
    for x in range(w-1,0,-1):
        p=int(parents[x,path[x]]);path[x-1]=p if p>=0 else path[x]
    conf=np.zeros(w,np.float32)
    accepted=np.zeros(w,bool);ambiguous=np.zeros(w,bool)
    pixel_span=np.zeros(w,np.int32)
    for x,y in enumerate(path):
        # No 'snap to nearest ink' beyond two pixels, to prevent fabricated traces.
        lo=max(0,y-2);hi=min(h,y+3)
        nearby=np.flatnonzero(candidate[lo:hi,x])
        if nearby.size==0:continue
        ypick=int(lo+nearby[np.argmin(np.abs((lo+nearby)-y))])
        py=ypick
        s=float(strength[py,x]);g=float(grid[py,x])
        # Distinct strong ink runs elsewhere in the same column create ambiguity.
        col_ys=np.flatnonzero(candidate[:,x])
        independent=col_ys[np.abs(col_ys-py)>max(5,h//15)]
        span=int(independent.size)
        pixel_span[x]=span
        c=np.clip(.85*s+.18-.35*g,0,1)
        if span>max(2,h//22):c*=.62;ambiguous[x]=True
        if py<3 or py>=h-3:c*=.35;ambiguous[x]=True
        conf[x]=c
        if c>=min_evidence:
            accepted[x]=True;path[x]=py
    # A column's confidence is not evidence that QRS/P/T exists.
    # Even if a gap is short, never interpolate across it.
    y_work=[float(path[x]) if accepted[x] else None for x in range(w)]
    y_src=[float(path[x]*oh/h) if accepted[x] else None for x in range(w)]
    gap_lengths=[];run=0
    for val in accepted:
        if val:
            if run:gap_lengths.append(run);run=0
        else:run+=1
    if run:gap_lengths.append(run)
    coverage=float(accepted.mean());ambiguity=float((ambiguous&accepted).mean())
    median_conf=float(np.median(conf[accepted])) if accepted.any() else 0.
    longest=max(gap_lengths,default=0)
    reasons=[]
    if coverage<.6:reasons.append('low_supported_coverage')
    if longest>max(8,w//40):reasons.append('long_unresolved_gap')
    if median_conf<.58:reasons.append('low_ink_confidence')
    if ambiguity>.2:reasons.append('multiple_ink_runs')
    if not accepted.any():reasons.append('no_supported_waveform')
    reasons+=['not_ground_truth_verified','no_calibrated_ms_or_mV']
    return {'version':VERSION,'status':'REVIEW','reason_codes':reasons,
            'image_size':[ow,oh],'work_size':[w,h],
            'x_src':[float((x+.5)*ow/w) for x in range(w)],
            'y_src':y_src,'y_work':y_work,
            'confidence':[round(float(v),3) for v in conf],
            'supported':accepted.tolist(),'ambiguous':ambiguous.tolist(),
            'metrics':{'coverage':round(coverage,4),'median_confidence_supported':round(median_conf,4),
                       'ambiguous_supported_fraction':round(ambiguity,4),
                       'longest_unresolved_gap_columns':longest,'total_columns':w,
                       'supported_columns':int(accepted.sum())},
            '_render':{'image':im,'enhanced':normalized,'candidate':candidate,'strength':strength}}


def overlay(original:Image.Image,result):
    out=original.convert('RGB').copy()
    draw=ImageDraw.Draw(out)
    ww,hh=out.size
    xs=result['x_src'];ys=result['y_src'];last=None
    for i,(x,y) in enumerate(zip(xs,ys)):
        if y is None:
            last=None;continue
        xi=int(round(x));yi=int(round(y))
        color=(18,220,215) if not result['ambiguous'][i] else (255,175,30)
        if last is not None and i-last[0]==1:
            draw.line((last[1],last[2],xi,yi),fill=color,width=max(1,ww//1400))
        else:
            r=max(1,ww//1400);draw.ellipse((xi-r,yi-r,xi+r,yi+r),fill=color)
        last=(i,xi,yi)
    return out


def panel(original:Image.Image,result):
    im=result['_render']['image'];w,h=im.size
    enh=Image.fromarray(result['_render']['enhanced'])
    candid=Image.fromarray(result['_render']['candidate'].astype('uint8')*255)
    trace=overlay(original,result).resize((w,h),Image.Resampling.LANCZOS)
    out=Image.new('RGB',(w*2,h*2),'white')
    for i,t in enumerate([im,enh,candid,trace]):out.paste(t.convert('RGB'),((i%2)*w,(i//2)*h))
    return out


def exportable(result):
    return {k:v for k,v in result.items() if k!='_render'}
