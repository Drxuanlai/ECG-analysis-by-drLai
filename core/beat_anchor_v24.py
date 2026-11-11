"""Research-only beat windows and independent geometry QA for ECG images.

No clinical rhythm/QRS interpretation is asserted. AI proposals, auto candidates,
and manually supplied anchors have distinct provenance.
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from PIL import Image, ImageDraw

ALLOWED = {'P_ONSET','P_PEAK','P_OFFSET','QRS_ONSET','Q_PEAK','R_PEAK',
           'S_NADIR','QRS_OFFSET','J_POINT','ST_POINT','T_PEAK','T_END','BASELINE'}


def windows_for_anchors(anchors: list[dict], width: int, half_width: int=95) -> list[dict]:
    """Non-overlapping QRS-centered windows; not an RR regularity assumption."""
    if width < 20 or half_width < 10: raise ValueError('Invalid width/window')
    unique = sorted((dict(a) for a in anchors if isinstance(a,dict)
                     and isinstance(a.get('x_pixel'),(float,int))
                     and not isinstance(a.get('x_pixel'),bool)
                     and math.isfinite(a['x_pixel']) and 0<=a['x_pixel']<width),
                    key=lambda a:a['x_pixel'])
    results=[]
    for a in unique:
        x=int(round(a['x_pixel']))
        if results and x-results[-1]['anchor_x']<12:
            continue
        start=max(0,x-half_width); end=min(width,x+half_width+1)
        if results and start<=results[-1]['x1']:
            mid=(x+results[-1]['anchor_x'])//2
            results[-1]['x1']=min(results[-1]['x1'],mid)
            start=max(start,mid+1)
        if start>=end: continue
        results.append({'beat_id':f'B{len(results)+1}','anchor_x':x,'x0':start,'x1':end,
                        'anchor_source':str(a.get('source','UNKNOWN')),'anchor_qc':'REVIEW_REQUIRED'})
    return results


def image_r_candidates(roi: Image.Image, max_candidates: int=24, min_distance: int=70) -> list[dict]:
    """Conservative ink-column candidate finder (not a validated QRS detector).

    Detects narrow, vertically coherent DARK strokes. Grids/labels can trigger it;
    all results remain review required. Avoids a regular-RR prior.
    """
    gray=np.asarray(roi.convert('L'),dtype=np.float32)
    h,w=gray.shape
    if h<24 or w<40:return []
    # Local contrast reduces uneven lighting; vertical run detects tall dark strokes.
    from scipy.ndimage import median_filter, maximum_filter1d
    bg=median_filter(gray,size=(11,31),mode='nearest')
    ink=(bg-gray)>24
    ink &= gray < 165
    # Ignore peripheral text/top/bottom; never silently infer R of negative QRS.
    upper=max(2,int(h*.10));lower=min(h-2,int(h*.92))
    mask=ink[upper:lower]
    # vertical accumulation plus narrowness; grid lines are downweighted by density
    vs=mask.sum(axis=0).astype(float)
    typical=median_filter(vs,size=25,mode='nearest')
    score=np.maximum(vs-typical,0.0)
    # waveform narrow QRS versus repeated grid marks
    local_peak=maximum_filter1d(score,size=9,mode='nearest')
    floor=max(4.,float(np.percentile(score,85))*0.9)
    spots=np.flatnonzero((score>=local_peak-1e-6)&(score>=floor))
    spots=sorted(spots,key=lambda x:float(score[x]),reverse=True)
    retained=[]
    for x in spots:
        if any(abs(int(x)-v['x_pixel'])<min_distance for v in retained):continue
        rows=np.where(mask[:,x])[0]
        if len(rows)<5:continue
        retained.append({'x_pixel':int(x),'y_hint_pixel':int(upper+np.median(rows)),
                         'score':round(float(score[x]),2),'source':'IMAGE_CANDIDATE_UNVERIFIED',
                         'status':'REVIEW_REQUIRED'})
        if len(retained)>=max_candidates:break
    return sorted(retained,key=lambda a:a['x_pixel'])


def map_local_landmarks(payload: Any, window:dict, roi:Image.Image) -> list[dict]:
    """Bounds-check AI local pixels and convert to unchanged ROI coordinates."""
    raw=payload.get('landmarks',[]) if isinstance(payload,dict) else []
    if not isinstance(raw,list):return []
    w=window['x1']-window['x0'];h=roi.height
    out=[]
    for entry in raw[:30]:
        if not isinstance(entry,dict):continue
        kind=str(entry.get('landmark_type','')).upper()
        x=entry.get('x_pixel');y=entry.get('y_pixel')
        if kind not in ALLOWED or isinstance(x,bool) or isinstance(y,bool):continue
        if not isinstance(x,(int,float)) or not isinstance(y,(int,float)):continue
        if not math.isfinite(x) or not math.isfinite(y) or not(0<=x<w and 0<=y<h):continue
        out.append({'beat_id':window['beat_id'],'landmark_type':kind,
                    'x_pixel':int(round(window['x0']+x)),'y_pixel':int(round(y)),
                    'x_local_pixel':float(x),'y_local_pixel':float(y),
                    'source':'AI_BEAT_WINDOW_UNVERIFIED','status':'REVIEW_REQUIRED',
                    'evidence':str(entry.get('evidence',''))[:240]})
    return out


def evidence_checks(roi:Image.Image, candidates:list[dict],
                    windows:list[dict],ink_radius:int=4)->list[dict]:
    """QA flags only: image proximity and temporal window, not morphologic truth."""
    gray=np.asarray(roi.convert('L'),dtype=float)
    height,width=gray.shape
    result=[]
    index={w['beat_id']:w for w in windows}
    for p in candidates:
        record=dict(p);flags=[]
        x=int(p['x_pixel']);y=int(p['y_pixel'])
        win=index.get(p['beat_id'])
        if not win or not(win['x0']<=x<win['x1']):flags.append('OUTSIDE_BEAT_WINDOW')
        if not (0<=x<width and 0<=y<height):flags.append('OUT_OF_ROI')
        else:
            y0=max(0,y-ink_radius);y1=min(height,y+ink_radius+1)
            x0=max(0,x-ink_radius);x1=min(width,x+ink_radius+1)
            patch=gray[y0:y1,x0:x1]
            # contrast is only a rough ink proximity check; grid crossings pass too
            if float(np.min(patch))>=185:flags.append('NO_DARK_INK_NEAR_COORDINATE')
            if x<5 or x>=width-5 or y<5 or y>=height-5:flags.append('IMAGE_EDGE')
        record['qa_flags']=flags
        record['qa_state']='REVIEW_REQUIRED' if not flags else 'LOCALIZATION_REVIEW'
        result.append(record)
    return result


def compare_to_human(predictions:list[dict],human:list[dict],
                     tol_x:int=8,tol_y:int=8)->list[dict]:
    """One-to-one 2D QC by landmark type, without modifying predictions."""
    valid=[p for p in human if p.get('source')=='HUMAN_CONFIRMED']
    pairs=[];used=set()
    for p in predictions:
        options=[]
        for i,h in enumerate(valid):
            if i in used or h.get('landmark_type')!=p.get('landmark_type'):continue
            dx=round(p['x_pixel']-h['x_pixel']);dy=round(p['y_pixel']-h['y_pixel'])
            if abs(dx)<=tol_x:options.append((abs(dx)+abs(dy),i,dx,dy))
        if not options:
            pairs.append({'beat_id':p['beat_id'],'landmark_type':p['landmark_type'],
                          'x_pixel':p['x_pixel'],'y_pixel':p['y_pixel'],
                          'status':'NO_X_REFERENCE_MATCH'});continue
        _,idx,dx,dy=min(options)
        used.add(idx)
        pairs.append({'beat_id':p['beat_id'],'landmark_type':p['landmark_type'],
                      'x_pixel':p['x_pixel'],'y_pixel':p['y_pixel'],
                      'dx':dx,'dy':dy,'reference_id':valid[idx].get('id'),
                      'status':'XY_WITHIN_TOLERANCE' if abs(dy)<=tol_y else 'X_ONLY_Y_MISMATCH'})
    return pairs


def beat_overlay(roi:Image.Image,window:dict,predictions:list[dict],human:list[dict])->Image.Image:
    """No dense labels; current beat only, human and AI geometries separate."""
    cut=roi.crop((window['x0'],0,window['x1'],roi.height)).convert('RGB')
    d=ImageDraw.Draw(cut)
    for p in predictions:
        if p.get('beat_id')!=window['beat_id']:continue
        x=p['x_pixel']-window['x0'];y=p['y_pixel']
        d.ellipse((x-4,y-4,x+4,y+4),outline=(173,54,205),width=2)
    for p in human:
        x=p.get('x_pixel',-100)-window['x0'];y=p.get('y_pixel',-100)
        if not (0<=x<cut.width and 0<=y<cut.height):continue
        d.line((x-5,y,x+5,y),fill=(15,175,90),width=2)
        d.line((x,y-5,x,y+5),fill=(15,175,90),width=2)
    d.line((window['anchor_x']-window['x0'],0,window['anchor_x']-window['x0'],8),fill=(245,154,40),width=2)
    return cut
