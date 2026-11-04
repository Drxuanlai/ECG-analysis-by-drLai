"""ECG Core 1.5.2: research-only R-peak proposals and independent human references.

Never interpret a steep ink shaft as a verified R peak. Supports inverted QRS
by using absolute local contrast; no RR periodicity or clinical interval output.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

VERSION = 'ecg-core-v1.5.2-r-peak-review'


def _smooth(a: np.ndarray, size: int) -> np.ndarray:
    size = max(3, int(size) | 1)
    return cv2.GaussianBlur(a.astype(np.float32).reshape(1, -1), (size, 1), 0).ravel()


def independent_rpeak_proposals(image: Image.Image, *, limit: int = 100) -> dict[str, Any]:
    """Image-derived recovery hints, not verified QRS or rhythm measurements.

    Detect narrow, high-contrast vertical features with a local-background
    subtraction, retaining both positive and negative deflections. Long full-
    height strokes, paper grid and near-edge columns carry rejection flags.
    The algorithm intentionally refuses to assert clinical R-peak labels.
    """
    im = np.asarray(image.convert('L'), dtype=np.float32)
    h, w = im.shape
    if h < 25 or w < 80:
        raise ValueError('ROI is too small')
    # Horizontally remove slow background; keep steep black or white deflections.
    gray = np.clip(im, 0, 255).astype(np.uint8)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(8, w / 70), sigmaY=2)
    residual = np.clip(background.astype(np.float32) - im, 0, 255)
    # Prefer longitudinal evidence without requiring a positive R polarity.
    local = np.percentile(residual, 82, axis=1, keepdims=True)
    threshold = np.maximum(8., local * .8)
    dark = (residual >= threshold).astype(np.uint8)
    vert = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((max(3, h // 18), 1), np.uint8))
    support = np.mean(vert > 0, axis=0).astype(np.float32)
    # Penalize long straight grid/calibration columns rather than deleting them.
    contrast = np.percentile(residual, 94, axis=0).astype(np.float32)
    novelty = _smooth(contrast / 50. + support, 5)
    baseline = _smooth(novelty, max(15, w // 29))
    score = np.maximum(0, novelty - .85 * baseline)
    positive = score[score > 0]
    cutoff = max(.19, float(np.quantile(positive, .72)) if positive.size else 100.)
    separation = max(6, w // 110)  # Only suppress duplicate pixel strokes; never RR regularization.
    margin = max(8, int(w * .018))
    peaks = []
    for x in range(margin, w - margin):
        window = score[max(0, x-separation):min(w, x+separation+1)]
        if score[x] >= cutoff and score[x] >= window.max() and x == max(0, x-separation) + int(window.argmax()):
            peaks.append(x)
    peaks = sorted(peaks, key=lambda x: score[x], reverse=True)
    keep = []
    for x in peaks:
        if all(abs(x-y) >= separation for y in keep):
            keep.append(x)
    keep = sorted(keep)[:limit]
    output=[]
    for x in keep:
        # Localization y is a dark-pixel coordinate, not an ECG baseline/peak annotation.
        lo, hi = max(0,x-2), min(w,x+3)
        col = residual[:,lo:hi]
        yy, _ = np.unravel_index(int(np.argmax(col)), col.shape)
        flags=[]
        if support[x] > .68: flags.append('extended_vertical_stroke_possible')
        if x < 2*margin or x > w-2*margin: flags.append('near_roi_edge')
        if np.mean(dark[:,x]) > .7: flags.append('grid_or_calibration_possible')
        output.append({'hint_id':f'H{len(output)+1:03d}', 'x_pixel':int(x),
                       'y_pixel':int(yy), 'strength_heuristic':round(float(score[x]),4),
                       'flags':flags, 'status':'UNVERIFIED_IMAGE_HINT'})
    return {'version':VERSION, 'status':'REVIEW', 'hints':output,
            'hint_count':len(output), 'roi_size':[w,h],
            'notes':['No RR regularization or positive-QRS requirement.',
                     'Hints may represent grid, T wave, artifact, or QRS.',
                     'These hints MUST NOT be counted as detected heartbeats.']}


def add_reference_point(points: list[dict], x: float, y: float, *, width: int,
                        height: int, min_separation_px: float = 3.) -> list[dict]:
    """Add directly reviewer-placed reference; duplicates and out-of-frame rejected."""
    xx,yy=float(x),float(y)
    if not (math.isfinite(xx) and math.isfinite(yy) and 0 <= xx < width and 0 <= yy < height):
        raise ValueError('Point outside ROI')
    if any(abs(p['x_pixel']-xx)<min_separation_px for p in points):
        raise ValueError('Reference too close to existing point')
    out=deepcopy(points)
    out.append({'reference_id':f'R{len(out)+1:03d}', 'x_pixel':round(xx,2),
                'y_pixel':round(yy,2), 'source':'HUMAN_DIRECT_MARK'})
    out.sort(key=lambda p:p['x_pixel'])
    for i,p in enumerate(out,1): p['reference_id']=f'R{i:03d}'
    return out


def match_references(events: list[dict], references: list[dict], *,
                     tolerance_px: float=10., review_complete: bool=False) -> dict:
    """One-to-one x matching; ARTIFACT labels never silently overridden.

    Countable performance requires independent exhaustive references AND
    complete review. An event which is near a reference but marked ARTIFACT
    is a disputed label rather than an automatic TP.
    """
    if tolerance_px <= 0: raise ValueError('tolerance_px must be positive')
    refs = sorted(references, key=lambda p:p['x_pixel'])
    evs = [e for e in events if e.get('review_label') != 'ARTIFACT']
    choices=[]
    for i,e in enumerate(evs):
        for j,r in enumerate(refs):
            dist=abs(float(e['x_pixel'])-float(r['x_pixel']))
            if dist <= tolerance_px: choices.append((dist,i,j))
    choices.sort()
    assigned_e=set();assigned_r=set();matches=[]
    for dist,i,j in choices:
        if i in assigned_e or j in assigned_r: continue
        assigned_e.add(i);assigned_r.add(j)
        matches.append({'event_id':evs[i]['event_id'], 'reference_id':refs[j]['reference_id'],
                        'x_predicted':evs[i]['x_pixel'], 'x_reference':refs[j]['x_pixel'],
                        'error_px':round(float(evs[i]['x_pixel']-refs[j]['x_pixel']),2)})
    unmatched_references=[r['reference_id'] for j,r in enumerate(refs) if j not in assigned_r]
    unmatched_events=[e['event_id'] for i,e in enumerate(evs) if i not in assigned_e]
    label_conflicts=[]
    for e in events:
        if e.get('review_label')=='ARTIFACT' and any(abs(e['x_pixel']-r['x_pixel'])<=tolerance_px for r in refs):
            label_conflicts.append(e['event_id'])
    verified = bool(review_complete and all(e.get('review_label') in ('QRS','ARTIFACT','UNCERTAIN') for e in events) and len(refs)>0 and not label_conflicts)
    counts={'matched_events':len(matches),'unmatched_references':len(unmatched_references),
            'unmatched_events':len(unmatched_events)}
    # Matched means spatial agreement only, not that the candidate is truly QRS.
    return {'status':'REFERENCE_COMPARISON_ONLY','matches':matches,
            'unmatched_reference_ids':unmatched_references,
            'unmatched_event_ids':unmatched_events,'artifact_label_conflicts':label_conflicts,
            'counts':counts, 'reference_complete':bool(review_complete),
            'exhaustive_scoring_allowed':verified,
            'performance_metrics':({'spatial_match_sensitivity':len(matches)/len(refs),
                'spatial_match_precision':len(matches)/max(len(evs),1)} if verified else None),
            'note':'Spatial matching is not clinical QRS verification. No ms conversion without verified pixel scale.'}


def mark_overlay(image:Image.Image, events:list[dict], references:list[dict], hints:list[dict]) -> Image.Image:
    out=image.convert('RGB').copy();d=ImageDraw.Draw(out);w,h=out.size
    for e in events:
        x=int(round(e['x_pixel']));lab=e.get('review_label','UNREVIEWED')
        color='#c83f40' if lab=='ARTIFACT' else '#2d8bce' if lab=='QRS' else '#e4a027'
        d.line((x,0,x,h-1),fill=color,width=1)
    for p in hints:
        x,y=int(p['x_pixel']),int(p['y_pixel']);d.ellipse((x-3,y-3,x+3,y+3),outline='#8e60ad',width=1)
    for r in references:
        x,y=int(round(r['x_pixel'])),int(round(r['y_pixel']))
        d.line((x-7,y,x+7,y),fill='#00bc71',width=2)
        d.line((x,y-7,x,y+7),fill='#00bc71',width=2)
    return out


def event_rpeak_localization(image: Image.Image, events: list[dict], *,
                             search_radius_px: int = 22) -> list[dict]:
    """Tentative R-like extrema *near existing events*, never verified peaks.

    Each original event is kept unchanged; a new position can differ from its
    x coordinate. The y coordinate is the location of high local dark contrast
    and may correspond to QRS flank, grid, text or artifact. No beat deletion.
    """
    a=np.asarray(image.convert('L'),dtype=np.float32);h,w=a.shape
    if h < 25 or w < 80: raise ValueError('ROI is too small')
    background=cv2.GaussianBlur(a,(0,0),sigmaX=max(6,w/90),sigmaY=2)
    contrast=np.maximum(background-a,0)
    out=[]
    for e in events:
        x=int(round(e['x_pixel']))
        if not 0 <= x < w:
            out.append({'event_id':e['event_id'],'x_event':e['x_pixel'],
                        'x_rpeak_hint':None,'y_rpeak_hint':None,'status':'NO_LOCAL_EVIDENCE'})
            continue
        lo=max(0,x-search_radius_px);hi=min(w,x+search_radius_px+1)
        sub=contrast[:,lo:hi]
        # Restrict very top/bottom text/neighbor-lead contamination; uncertain.
        ylo=max(2,int(h*.1));yhi=min(h-2,int(h*.9))
        roi=sub[ylo:yhi]
        yy,xx=np.unravel_index(np.argmax(roi),roi.shape)
        strength=float(roi[yy,xx]);xx+=lo;yy+=ylo
        flags=[]
        if strength < 10:flags.append('weak_local_ink')
        if xx <= lo+2 or xx >= hi-3: flags.append('search_window_edge')
        if np.mean(contrast[:,xx]>10)>.65: flags.append('grid_or_calibration_possible')
        out.append({'event_id':e['event_id'],'x_event':float(e['x_pixel']),
                    'x_rpeak_hint':int(xx), 'y_rpeak_hint':int(yy),
                    'delta_x_px':round(float(xx-e['x_pixel']),2),
                    'strength_heuristic':round(strength,2),
                    'status':'UNVERIFIED_LOCALIZATION_HINT', 'flags':flags})
    return out
