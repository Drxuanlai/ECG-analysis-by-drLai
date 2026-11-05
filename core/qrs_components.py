"""Research-only image evidence for QRS components (v1.5.7).

No human reference coordinates enter detection; no diagnostic claims or intervals.
Image-space y increases downward. Deep negative deflections are sought without
assuming regular RR or that every QRS has a positive R wave.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image, ImageDraw

VERSION = 'ecg-core-v1.5.7-qrs-components'


def _ink_mask(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    gray = rgb.mean(axis=2)
    # Dark, nearly neutral ink; red/pink ECG grid is usually lighter and more chromatic.
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    gray_cutoff = min(140.0, float(np.percentile(gray, 12)) + 34.0)
    mask = (gray < gray_cutoff) & (chroma < 43)
    return mask


def _baseline(mask: np.ndarray) -> float | None:
    h, w = mask.shape
    ys = []
    for x in range(max(4, int(w*.025)), min(w-4, int(w*.94))):
        pixels = np.flatnonzero(mask[:, x])
        # Exclude characters/calibration and steep QRS when estimating baseline.
        if 1 <= len(pixels) <= 9:
            ys.append(float(np.median(pixels)))
    if len(ys) < max(18, w // 15):
        return None
    return float(np.median(ys))


def _trace_y(mask: np.ndarray, x: int, ymin: int, ymax: int, side: str) -> float | None:
    h, w = mask.shape
    x = int(x)
    x0, x1 = max(0, x-1), min(w, x+2)
    if x0 >= x1:
        return None
    vals = np.flatnonzero(mask[max(0,ymin):min(h,ymax),x0:x1].any(axis=1))
    if not len(vals):
        return None
    absolute = vals + max(0, ymin)
    return float(absolute.max() if side == 'bottom' else absolute.min())


def detect_qrs_components(image: Image.Image, hints: list[dict] | None = None,
                          *, min_depth_px: float = 17.0) -> dict:
    """Localize putative negative QRS nadirs and any small preceding r apex.

    Candidate evidence is the neutral-dark ink shape, NOT saved physician marks.
    Only *proposals* are returned. QS is never asserted when r is not resolved.
    """
    mask = _ink_mask(image)
    h, w = mask.shape
    base = _baseline(mask)
    empty = {'version':VERSION,'status':'REVIEW','roi_size':[w,h],
             'baseline_y_estimate':base,'components':[], 'candidate_count':0,
             'notes':['Proposals only. Human marks never enter detection.',
                      'No rhythm assumptions, no QRS duration or diagnosis.']}
    if base is None or w < 70 or h < 45:
        empty['reason']='BASELINE_UNRESOLVED'
        return empty
    b = float(base)
    low = max(0, min(h-1,int(round(b + 8))))
    high = min(h, int(round(b + max(35, h*.48))))
    depths = np.zeros(w, dtype=np.float32)
    counts = np.zeros(w, dtype=np.float32)
    for x in range(2, w-2):
        segment=mask[low:high,x-1:x+2]
        where=np.flatnonzero(segment.any(axis=1))
        if where.size:
            depths[x] = max(0., float(low+where.max()-b))
            counts[x] = float(np.sum(segment.any(axis=1)))
    padded=np.pad(depths,(2,2),mode='edge')
    smooth=np.median(np.stack([padded[i:i+w] for i in range(5)],axis=0),axis=0)
    # High S is supported by both depth and ink along the steep descent;
    # a single printed dot/line should not count as a confirmed component.
    # Local maxima with sufficient contrast to nearby columns; do not infer RR.
    peaks=[]
    radius=max(13,int(h*.10))
    for x in range(radius,w-radius):
        v=float(smooth[x])
        if v<min_depth_px or v<max(float(smooth[x-1]),float(smooth[x+1])):continue
        left=float(np.min(smooth[x-radius:x]))
        right=float(np.min(smooth[x+1:x+radius+1]))
        if v - max(left,right) >= max(9.,min_depth_px*.52):
            peaks.append(x)
    peaks=sorted(peaks,key=lambda x:-float(smooth[x]))
    # Suppress duplicate minima within the same broad trough; NOT an RR rule.
    selected=[]
    for x in peaks:
        if x < 9 or x >= w-10:
            continue
        if any(abs(int(x)-int(prev)) < max(18,int(h*.16)) for prev in selected):
            continue
        selected.append(int(x))
    selected.sort()
    components=[]
    for idx, seed_x in enumerate(selected, start=1):
        left=max(0,seed_x-24);right=min(w-1,seed_x+24)
        candidates=[]
        for xx in range(left,right+1):
            yy=_trace_y(mask,xx,low,high,'bottom')
            if yy is None:continue
            # Require some ink around proposed nadir; avoid isolated bottom dot.
            local_support=int(mask[max(0,int(yy)-3):min(h,int(yy)+4),max(0,xx-2):min(w,xx+3)].sum())
            vertical_evidence=float(counts[xx])
            if vertical_evidence < 7:  # Reject lone dark dots / sparse grid intersections.
                continue
            utility = 0.65*float(yy-b) + 1.1*vertical_evidence - 0.16*abs(xx-seed_x)
            candidates.append((utility,float(yy),local_support,xx))
        if not candidates:continue
        _,bottom,support,sx=max(candidates,key=lambda t:t[0])
        depth=bottom-b
        if any(abs(sx - int(c['S_nadir']['x_pixel'])) < max(24,int(h*.15)) for c in components):
            continue
        r_search_start=max(1,sx-max(24,int(h*.15)))
        r_search_end=max(r_search_start,sx-4)
        r_options=[]
        for rx in range(r_search_start,r_search_end):
            ry=_trace_y(mask,rx,max(0,int(b-28)),max(1,int(b-2)),'top')
            if ry is not None:
                amp=b-ry
                r_options.append((amp,rx,ry))
        r_apex=None
        if r_options:
            amp,rx,ry=max(r_options)
            if amp>=3.0:
                r_apex={'x_pixel':float(rx),'y_pixel':float(ry),
                        'amplitude_proxy_px':round(float(amp),2),
                        'type':'r_APEX_CANDIDATE','status':'REVIEW'}
        flags=[]
        if depth < min_depth_px:flags.append('DEPTH_BELOW_THRESHOLD')
        if sx < 16 or sx > w-17:flags.append('NEAR_ROI_EDGE')
        if int(counts[seed_x]) < 3:flags.append('THIN_INK_EVIDENCE')
        if r_apex is None:flags.append('r_NOT_RESOLVED_QS_NOT_PROVEN')
        components.append({'component_id':f'C{idx:03d}',
            'qrs_event_x_candidate_px':float(sx),
            'S_nadir':{'x_pixel':float(sx),'y_pixel':float(bottom),
                       'type':'S_NADIR_CANDIDATE','status':'REVIEW'},
            'r_apex':r_apex,
            'depth_below_baseline_px':round(float(depth),2),
            'local_ink_support_pixels':int(support),
            'search_box_xyxy':[max(0,sx-28),max(0,int(b-30)),min(w,sx+16),min(h,int(bottom+8))],
            'morphology_proposal':'rS_POSSIBLE_REVIEW' if r_apex else 'NEGATIVE_QRS_REVIEW',
            'qc':'REVIEW', 'reason_codes':flags,
        })
    empty.update({'components':components,'candidate_count':len(components),
                  'reason':'IMAGE_MORPHOLOGY_UNVERIFIED'})
    return empty


def compare_components_to_references(report:dict,references:list[dict],
                                     *, tolerance_x_px:float=10.,tolerance_y_px:float=8.) -> dict:
    """Evaluation ONLY: saved physician labels are not fed into detection.

    Pair only like-for-like component types. Do not infer QS from an S mark.
    """
    used=set(); matches=[]
    for ref in references:
        kind=ref.get('fiducial_type','R_PEAK')
        field = 'S_nadir' if kind=='S_NADIR' else ('r_apex' if kind=='R_PEAK' else None)
        if field is None: continue
        options=[]
        for k,c in enumerate(report.get('components',[])):
            if (k,field) in used or not c.get(field): continue
            p=c[field]
            dx=float(p['x_pixel'])-float(ref['x_pixel'])
            dy=float(p['y_pixel'])-float(ref['y_pixel'])
            if abs(dx)<=tolerance_x_px:
                options.append((abs(dx)+abs(dy)/3,k,dx,dy,c))
        if not options:
            matches.append({'reference_id':ref.get('reference_id'), 'reference_type':kind,
                            'status':'NO_X_CANDIDATE'}); continue
        _,k,dx,dy,c=min(options)
        used.add((k,field))
        matches.append({'reference_id':ref.get('reference_id'), 'reference_type':kind,
                        'candidate_id':c['component_id'],
                        'error_x_px':round(dx,2),'error_y_px':round(dy,2),
                        'status':'XY_WITHIN_TOLERANCE' if abs(dy)<=tolerance_y_px
                                 else 'X_ONLY_Y_MISMATCH'})
    return {'status':'REVIEW_ONLY', 'matches':matches,
            'reference_count':len(references), 'compared_count':len(matches),
            'notes':['Reference labels used for evaluation only.',
                     'R_PEAK compared with r apex; S_NADIR compared with S nadir.',
                     'QS_NADIR/QRS_FIDUCIAL not silently mapped to R or S.']}


def component_overlay(image:Image.Image, report:dict, references:list[dict] | None=None) -> Image.Image:
    out=image.convert('RGB').copy();draw=ImageDraw.Draw(out)
    b=report.get('baseline_y_estimate')
    if b is not None:draw.line((0,round(b),out.width-1,round(b)),fill='#de9840',width=1)
    for c in report.get('components',[]):
        p=c['S_nadir'];x=int(round(p['x_pixel']));y=int(round(p['y_pixel']))
        draw.ellipse((x-4,y-4,x+4,y+4),outline='#1875d2',width=2)
        if c['r_apex']:
            r=c['r_apex'];a=int(round(r['x_pixel']));z=int(round(r['y_pixel']))
            draw.rectangle((a-3,z-3,a+3,z+3),outline='#cf44af',width=2)
    for ref in references or []:
        x=int(round(ref['x_pixel']));y=int(round(ref['y_pixel']))
        draw.line((x-6,y,x+6,y),fill='#19a35f',width=2)
        draw.line((x,y-6,x,y+6),fill='#19a35f',width=2)
    return out
