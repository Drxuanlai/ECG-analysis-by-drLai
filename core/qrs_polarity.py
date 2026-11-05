"""v1.5.9: polarity-aware QRS component candidates (research-only).

Do not use human labels for detection. A positive or negative deflection is a
candidate, not a verified beat. Never derive HR, duration or diagnosis here.
"""
from __future__ import annotations

from PIL import Image, ImageDraw
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from core.qrs_components import _ink_mask, _baseline, _horizontal_run, detect_qrs_components

VERSION = 'ecg-core-v1.5.9-polarity-aware-qrs'


def _runs(column: np.ndarray) -> list[tuple[int,int]]:
    a = np.asarray(column, dtype=np.int8)
    z = np.diff(np.r_[0, a, 0])
    return list(zip(np.where(z == 1)[0].tolist(), np.where(z == -1)[0].tolist()))


def _dark_vertical_support(mask: np.ndarray, x: int, baseline: float, direction: str) -> tuple[float, int, int]:
    """Vertical stroke evidence in original mask, not inferred from RR.

    Ink at x may be split by slight slant or grid: aggregate nearby columns,
    but avoid counting isolated dots as a tall continuous stroke.
    """
    h,w = mask.shape
    max_length = 0
    best_y = -1
    for xx in range(max(0,x-4),min(w,x+5)):
        for start,end in _runs(mask[:,xx]):
            length = end-start
            if direction == 'positive':
                # A real upward stroke should approach the local isoelectric
                # line, though its distal half can be slanted.
                eligible = start < baseline-13 and end >= baseline-max(34, .22*h)
                tip = start
            else:
                eligible = end > baseline+13 and start <= baseline+max(34,.22*h)
                tip = end-1
            if eligible and length > max_length:
                max_length = length
                best_y = int(tip)
    return float(max_length), int(best_y), x


def _positive_events(mask: np.ndarray, baseline: float, *, min_prominence_px: float = 19.) -> list[dict]:
    h,w=mask.shape
    stroke = np.zeros(w, dtype=float)
    tops = np.full(w, np.nan)
    for x in range(7,w-7):
        run, tip, _ = _dark_vertical_support(mask,x,baseline,'positive')
        if run >= 10 and tip >= 0:
            # Outlier top ink alone must not outrank long QRS slopes.
            stroke[x] = min(run,150) * .75 + max(0,baseline-tip)*.25
            tops[x] = tip
    # Smooth short spatial gaps while leaving a QRS event window separate.
    filtered = median_filter(stroke,size=5)
    peaks, props = find_peaks(filtered, prominence=max(11,min_prominence_px*.55),distance=max(22,int(h*.10)))
    initial=[]
    for x,pr in zip(peaks,props['prominences']):
        if filtered[x] < min_prominence_px: continue
        lo,hi=max(7,int(x)-8),min(w-7,int(x)+9)
        xx=int(lo+np.argmax(stroke[lo:hi]))
        y=float(tops[xx]) if np.isfinite(tops[xx]) else None
        if y is None:continue
        # Recover distal tip when a long upright stroke is split into two
        # dark runs by the photographed paper grid or a thin diagonal.
        tip_options=[]
        for tx in range(max(0,xx-8), min(w,xx+9)):
            for start,end in _runs(mask[:,tx]):
                if end-start>=9 and max(3,int(.015*h)) <= start < baseline-12:
                    tip_options.append((start,tx))
        if tip_options:
            tip_y,tip_x=min(tip_options)
            if tip_y < y:
                y=float(tip_y)
                # Preserve candidate event X near the strong vertical stroke.

        # Distinct physical sharp stroke required. Extrema based on grid-only
        # isolated dots cannot satisfy this.
        if stroke[xx]<min_prominence_px:continue
        near_edge=bool(xx<max(16,.028*w) or xx>w-max(18,.045*w))
        clipped=bool(y<=max(4,.012*h))
        initial.append({'x_pixel':float(xx),'y_pixel':y,
                        'type':'R_APEX_CANDIDATE','polarity':'POSITIVE',
                        'stroke_score':round(float(stroke[xx]),2),
                        'prominence_proxy':round(float(pr),2),
                        'qc':'REVIEW','reason_codes':(
                            ['NEAR_ROI_EDGE'] if near_edge else [])+
                            (['TOP_CLIPPED_OR_TEXT_SUSPECT'] if clipped else [])})
    # One event per close x cluster, not an RR regularity constraint.
    initial.sort(key=lambda p:-p['stroke_score'])
    keep=[]
    for c in initial:
        if not any(abs(c['x_pixel']-p['x_pixel'])<max(18,.09*h) for p in keep):
            keep.append(c)
    return sorted(keep,key=lambda p:p['x_pixel'])


def detect_polarity_components(image: Image.Image, *, min_depth_px:float=17.) -> dict:
    """Keep old negative candidates AND find positive QRS strokes separately.

    No candidate is automatically VERIFIED. Edge contamination, flat-shelf
    calibration shapes and low voltage require review. Negative-only extraction
    must not suppress strong positive R complexes.
    """
    mask = _ink_mask(image)
    h,w=mask.shape
    b=_baseline(mask)
    if b is None:
        return {'version':VERSION,'status':'REVIEW','reason':'BASELINE_UNRESOLVED',
                'roi_size':[w,h],'positive_components':[], 'negative_components':[],
                'positive_unresolved':[], 'positive_count':0, 'negative_count':0,
                'positive_unresolved_count':0,'excluded_candidates':[], 'candidate_count':0,
                'polarity_proposal':'UNRESOLVED','notes':['Insufficient ink evidence; do not measure QRS.']}
    old=detect_qrs_components(image,min_depth_px=min_depth_px)
    pos=_positive_events(mask,float(b))
    suspect_pos=[]
    priority_pos=[]
    for p in pos:
        # This is a cautious *right-end exclusion zone* for QA review, NOT a
        # claim of calibration detection and NOT a hard signal deletion.
        # A very late event near the printer's gain pulse is retained in JSON.
        ix=int(p['x_pixel']); iy=int(p['y_pixel'])
        right_shelf=max([_horizontal_run(mask, yy, ix+3, min(w,ix+80))
                          for yy in range(max(0,int(b)-80), min(h,int(b)+12),5)] or [0])
        tip_shelf=_horizontal_run(mask,iy,ix-5,min(w,ix+45))
        calibration_pattern = (ix >= .77*w and
            (tip_shelf >= 10 or (ix >= .90*w and right_shelf >= 13)
             or (ix >= .77*w and right_shelf >= 15 and iy > b-55)
             or ix >= .94*w))
        if calibration_pattern:
            p['reason_codes']=list(p['reason_codes'])+['RIGHT_END_CALIBRATION_OR_TRUE_QRS_UNRESOLVED']
            p['qc']='UNRESOLVED'
            suspect_pos.append(p)
        elif p['x_pixel'] <= .035*w:
            p['reason_codes']=list(p['reason_codes'])+['LEFT_LABEL_OR_CLIPPED_QRS_UNRESOLVED']
            p['qc']='UNRESOLVED'
            suspect_pos.append(p)
        else:
            priority_pos.append(p)
    # Previously excluded negative deflections remain as reviewable evidence.
    # The old geometric calibration rule is not sufficiently specific to
    # dismiss a QRS on its own (observed in lead II).
    excluded=list(old.get('excluded_candidates',[]))
    neg=[]
    for c in old.get('components',[]):
        p=dict(c)
        y=float(p['S_nadir']['y_pixel'])
        if y>=h-3:
            p['reason_codes']=list(p.get('reason_codes',[]))+['BOTTOM_CLIPPED_UNVERIFIED']
        p['polarity']='NEGATIVE'
        neg.append(p)
    for e in excluded:
        e['status']='CALIBRATION_OR_QRS_UNRESOLVED'
        e['reason_codes']=list(e.get('reason_codes',[]))+['GEOMETRY_ALONE_CANNOT_EXCLUDE_QRS']
    # Positive evidence can dominate even when the old negative branch sees
    # a shelf or a deep S. Do not infer morphology without matching BOTH.
    positive_strength=sum(float(p['stroke_score']) for p in pos)
    negative_strength=sum(max(0,float(c.get('depth_below_baseline_px',0))) for c in neg)
    if priority_pos and neg:
        polarity='MIXED_OR_BIPHASIC_REVIEW'
    elif priority_pos:
        polarity='POSITIVE_EVIDENCE_REVIEW'
    elif neg:
        polarity='NEGATIVE_EVIDENCE_REVIEW'
    else:
        polarity='UNRESOLVED'
    return {'version':VERSION,'status':'REVIEW','roi_size':[w,h],
            'baseline_y_estimate':round(float(b),2),
            'polarity_proposal':polarity,
            'positive_components':priority_pos,'positive_unresolved':suspect_pos,'negative_components':neg,
            'excluded_candidates':excluded,
            'positive_count':len(priority_pos),'positive_unresolved_count':len(suspect_pos),'negative_count':len(neg),
            'candidate_count':len(priority_pos)+len(neg),
            'evidence_strength':{'positive_proxy':round(positive_strength,2),'negative_proxy':round(negative_strength,2)},
            'notes':['Candidate extrema only; R/S/QS classification unverified.',
                     'No RR regularity assumptions or clinical measurements.',
                     'Calibration exclusions are flagged, not silently deleted.',
                     'Image ink can be affected by grid, labels, and crop limits.']}


def polarity_overlay(image:Image.Image, report:dict) -> Image.Image:
    out=image.convert('RGB').copy();d=ImageDraw.Draw(out)
    for p in report.get('positive_components',[]):
        x,y=round(p['x_pixel']),round(p['y_pixel'])
        d.rectangle((x-4,y-4,x+4,y+4),outline='#8c39ca',width=2)
    for p in report.get('positive_unresolved',[]):
        x,y=round(p['x_pixel']),round(p['y_pixel'])
        d.rectangle((x-5,y-5,x+5,y+5),outline='#e69b12',width=2)
    for c in report.get('negative_components',[]):
        p=c['S_nadir'];x,y=round(p['x_pixel']),round(p['y_pixel'])
        d.ellipse((x-4,y-4,x+4,y+4),outline='#006ec4',width=2)
        if 'BOTTOM_CLIPPED_UNVERIFIED' in c.get('reason_codes',[]):
            d.line((x-6,y-6,x+6,y+6),fill='#ee9b21',width=2)
    for e in report.get('excluded_candidates',[]):
        x,y=round(e['x_pixel']),round(e['y_pixel'])
        d.line((x-5,y-5,x+5,y+5),fill='#e33a30',width=2)
        d.line((x-5,y+5,x+5,y-5),fill='#e33a30',width=2)
    return out


def compare_polarity_to_references(report:dict, references:list[dict], *, tol_x:float=10.,tol_y:float=8.) -> dict:
    """One-to-one independent QA matching; user marks never train detector."""
    predictions=[]
    for c in report.get('positive_components',[]):
        predictions.append(('R_PEAK', c['x_pixel'],c['y_pixel']))
    for c in report.get('negative_components',[]):
        p=c['S_nadir'];predictions.append(('S_NADIR',p['x_pixel'],p['y_pixel']))
    used=set();matches=[]
    for ref in references:
        typ=ref.get('fiducial_type','R_PEAK')
        choices=[]
        for i,(kind,x,y) in enumerate(predictions):
            if i in used or kind!=typ:continue
            dx=float(x)-float(ref['x_pixel']);dy=float(y)-float(ref['y_pixel'])
            if abs(dx)<=tol_x:choices.append((abs(dx)+abs(dy)/3,i,dx,dy))
        if choices:
            _,i,dx,dy=min(choices)
            used.add(i)
            outcome='XY_WITHIN_TOLERANCE' if abs(dy)<=tol_y else 'X_ONLY_Y_MISMATCH'
            matches.append({'reference_id':ref.get('reference_id'), 'reference_type':typ,
                            'error_x_px':round(dx,2),'error_y_px':round(dy,2),'status':outcome})
        else:
            matches.append({'reference_id':ref.get('reference_id'), 'reference_type':typ,
                            'status':'NO_X_CANDIDATE'})
    return {'status':'REVIEW_ONLY','matches':matches,
            'references_count':len(references),
            'xy_match_count':sum(m['status']=='XY_WITHIN_TOLERANCE' for m in matches),
            'note':'Only comparisons against saved points; not sensitivity unless exhaustive annotations verified.'}
