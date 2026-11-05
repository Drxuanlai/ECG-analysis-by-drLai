"""Research-only image evidence for QRS components (v1.5.7).

No human reference coordinates enter detection; no diagnostic claims or intervals.
Image-space y increases downward. Deep negative deflections are sought without
assuming regular RR or that every QRS has a positive R wave.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image, ImageDraw

VERSION = 'ecg-core-v1.5.8-calibration-small-r-qrs-components'


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



def _local_baseline(mask: np.ndarray, sx: int, global_base: float) -> float:
    """Use the pre-QRS iso-electric trace, not a single lead-wide y level.

    Avoid the steep QRS: inspect 12..38 px *before* the putative S.
    Return global fallback when insufficient ink supports a local baseline.
    """
    h, w = mask.shape
    values = []
    x0, x1 = max(1, sx-39), min(w-1, sx-11)
    for x in range(x0, x1):
        ys = np.flatnonzero(mask[:, x])
        if not 1 <= len(ys) <= 10:
            continue
        local = ys[(ys >= global_base-23) & (ys <= global_base+23)]
        if len(local):
            values.append(float(np.median(local)))
    if len(values) < 7:
        return float(global_base)
    return float(np.median(values))


def _horizontal_run(mask: np.ndarray, y: float, x0: int, x1: int,
                    *, radius_y: int = 2) -> int:
    """Longest supported horizontal run near a y coordinate."""
    h, w = mask.shape
    y = int(round(y))
    lo, hi = max(0, y-radius_y), min(h, y+radius_y+1)
    x0, x1 = max(0, x0), min(w, x1)
    if lo >= hi or x0 >= x1:
        return 0
    occupancy = np.any(mask[lo:hi, x0:x1], axis=0)
    longest = current = 0
    for val in occupancy:
        current = current+1 if val else 0
        longest = max(longest, current)
    return longest


def _post_s_recovery(mask: np.ndarray, sx: int, base: float,
                     *, within_px: int = 30) -> bool:
    """Find a post-nadir ink trace near baseline; no RR regularity assumed."""
    h, w = mask.shape
    for x in range(sx+7, min(w-2, sx+within_px+1)):
        ys = np.flatnonzero(mask[:, x])
        if len(ys) and np.any(np.abs(ys-base) <= 7):
            # Exclude a single isolated dot at a grid crossing.
            following = mask[max(0,int(base)-8):min(h,int(base)+9),x:min(w,x+5)]
            if following.sum() >= 4:
                return True
    return False


def _calibration_like(mask: np.ndarray, sx: int, sy: float,
                      base: float) -> tuple[bool, list[str]]:
    """Reject a stair-step calibration candidate only on multiple cues.

    A true large negative QRS rapidly returns to its local baseline.
    A calibration pulse edge usually ends in a long, flat shelf and may
    have another vertical edge nearby. No unconditional right-edge mask.
    """
    h, w = mask.shape
    flags = []
    flat_after = _horizontal_run(mask, sy, sx+2, sx+29) >= 11
    flat_before = _horizontal_run(mask, sy, sx-28, sx-2) >= 11
    return_to_base = _post_s_recovery(mask, sx, base)
    if flat_after or flat_before:
        flags.append('HORIZONTAL_CALIBRATION_SHELF')
    if not return_to_base:
        flags.append('NO_POST_QRS_BASELINE_RECOVERY')
    # A rectangular calibration edge has a long vertical segment linked to
    # a flat shelf. The two cues must coexist, not merely be near the edge.
    x0,x1=max(0,sx-3),min(w,sx+4)
    col_span=max((int(np.ptp(np.flatnonzero(mask[:,x])))+1
                  for x in range(x0,x1) if mask[:,x].sum()>=2),default=0)
    rectangular_stroke=col_span >= 30
    if rectangular_stroke and (flat_after or flat_before):
        flags.append('VERTICAL_EDGE_WITH_FLAT_SHELF')
    return bool((flat_after or flat_before) and
                (rectangular_stroke or not return_to_base)), flags


def _small_r_before_s(mask: np.ndarray, sx: int, local_base: float,
                      *, global_base: float) -> dict | None:
    """Look for a small pre-S positive excursion relative to LOCAL baseline.

    Require a locally supported dark trace before descent, within 3..19 px
    of S, and a rise of >=3 px above that baseline. Never infer QS from None.
    """
    h, w = mask.shape
    options = []
    for rx in range(max(1,sx-20), max(1,sx-3)):
        ry = _trace_y(mask, rx, max(0,int(local_base)-18),
                      min(h,int(local_base)-1), 'top')
        if ry is None:
            continue
        amp = local_base-ry
        if amp < 3 or amp > 18:
            continue
        near = mask[max(0,int(ry)-3):min(h,int(ry)+4),max(0,rx-2):min(w,rx+3)]
        if near.sum() < 3:
            continue
        options.append((float(amp),int(rx),float(ry)))
    if not options:
        return None
    # Choose the top of the excursion; closer to S is preferred only in ties.
    amp,rx,ry=max(options,key=lambda z:(z[0],z[1]))
    return {'x_pixel':float(rx),'y_pixel':float(ry),
            'amplitude_proxy_px':round(amp,2),
            'local_baseline_y':round(local_base,2),
            'type':'r_APEX_CANDIDATE','status':'REVIEW'}


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
             'baseline_y_estimate':base,'components':[], 'candidate_count':0, 'excluded_candidates':[],
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
    excluded=[]
    for idx, seed_x in enumerate(selected, start=1):
        left=max(0,seed_x-24);right=min(w-1,seed_x+24)
        candidates=[]
        for xx in range(left,right+1):
            yy=_trace_y(mask,xx,low,high,'bottom')
            if yy is None:continue
            local_support=int(mask[max(0,int(yy)-3):min(h,int(yy)+4),max(0,xx-2):min(w,xx+3)].sum())
            vertical_evidence=float(counts[xx])
            if vertical_evidence < 7:
                continue
            utility = 0.65*float(yy-b) + 1.1*vertical_evidence - 0.16*abs(xx-seed_x)
            candidates.append((utility,float(yy),local_support,xx))
        if not candidates:continue
        _,bottom,support,sx=max(candidates,key=lambda t:t[0])
        # Use the local pre-QRS baseline for morphology. Global baseline is
        # retained as a fallback, not an absolute criterion for a small r.
        local_base=_local_baseline(mask,sx,b)
        depth=bottom-local_base
        if any(abs(sx - int(c['S_nadir']['x_pixel'])) < max(24,int(h*.15)) for c in components):
            continue
        is_calibration,calibration_flags=_calibration_like(mask,sx,bottom,local_base)
        if is_calibration:
            excluded.append({'candidate_id':f'X{idx:03d}',
                'x_pixel':float(sx),'y_pixel':float(bottom),
                'type':'CALIBRATION_PULSE_CANDIDATE',
                'reason_codes':calibration_flags,'status':'EXCLUDED_FROM_QRS_CANDIDATES'})
            continue
        r_apex=_small_r_before_s(mask,sx,local_base,global_base=b)
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
            'local_baseline_y_estimate':round(local_base,2),
            'depth_below_baseline_px':round(float(depth),2),
            'local_ink_support_pixels':int(support),
            'search_box_xyxy':[max(0,sx-28),max(0,int(local_base-30)),min(w,sx+16),min(h,int(bottom+8))],
            'morphology_proposal':'rS_POSSIBLE_REVIEW' if r_apex else 'NEGATIVE_QRS_REVIEW',
            'qc':'REVIEW', 'reason_codes':flags,
        })
    empty.update({'components':components,'candidate_count':len(components),
                  'excluded_candidates':excluded,
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
    for e in report.get('excluded_candidates',[]):
        x=int(round(e['x_pixel']));y=int(round(e['y_pixel']))
        draw.line((x-6,y-6,x+6,y+6), fill='#d93025', width=2)
        draw.line((x-6,y+6,x+6,y-6), fill='#d93025', width=2)
    for ref in references or []:
        x=int(round(ref['x_pixel']));y=int(round(ref['y_pixel']))
        draw.line((x-6,y,x+6,y),fill='#19a35f',width=2)
        draw.line((x,y-6,x,y+6),fill='#19a35f',width=2)
    return out
