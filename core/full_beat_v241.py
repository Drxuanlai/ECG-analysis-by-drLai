"""Research-only QRS event curation and full P-QRS-ST-T windows (v2.4.1).

No label from the clinician reference set is used to create predictions. A
candidate can only become an analyzed beat through explicit user selection;
unselected candidates remain unresolved, never silently called artifacts.
"""
from __future__ import annotations
import math
from typing import Any
from PIL import Image, ImageDraw


def clean_candidates(raw: list[dict], width: int, merge_px: int = 9) -> list[dict]:
    """Deduplicate near-identical hypotheses without rejecting close true beats.

    Other close candidates (>merge_px) are retained for expert adjudication.
    """
    if width < 1:
        return []
    valid = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        x = item.get('x_pixel')
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            continue
        if not 0 <= x < width:
            continue
        rec = dict(item)
        rec['x_pixel'] = int(round(x))
        rec['score'] = _finite_score(item.get('score'))
        valid.append(rec)
    valid.sort(key=lambda r: (-r['score'], r['x_pixel']))
    kept = []
    for rec in valid:
        if all(abs(rec['x_pixel'] - q['x_pixel']) > merge_px for q in kept):
            kept.append(rec)
    kept.sort(key=lambda r: r['x_pixel'])
    for idx, item in enumerate(kept):
        item['candidate_id'] = f'C{idx+1:03d}'
        item['event_status'] = 'UNRESOLVED_NOT_A_BEAT'
    return kept


def _finite_score(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else 0.0
    except (TypeError, ValueError):
        return 0.0


def propose_qrs_candidates(candidates: list[dict], width: int,
                           min_separation: int = 55) -> list[dict]:
    """Rank candidate spikes for convenience, NOT a QRS diagnosis.

    Keep all hypotheses, including T-wave lookalikes. A local high-score
    narrow stroke can be a provisional QRS, never automatically confirmed.
    No RR-regularity constraint, and no mandatory minimum RR is enforced.
    """
    items = clean_candidates(candidates, width)
    if not items:
        return []
    positives = [c['score'] for c in items if c['score'] > 0]
    if positives:
        threshold = max(3.5, sorted(positives)[len(positives)//2] * 0.6)
    else:
        threshold = float('inf')
    order = sorted(range(len(items)), key=lambda i: items[i]['score'], reverse=True)
    provisionals: set[int] = set()
    for i in order:
        if items[i]['score'] < threshold:
            continue
        if any(abs(items[i]['x_pixel']-items[j]['x_pixel']) < min_separation for j in provisionals):
            continue
        provisionals.add(i)
    for i, item in enumerate(items):
        item['suggested_qrs'] = i in provisionals
        item['suggestion_status'] = ('QRS_LIKE_UNVERIFIED' if i in provisionals
                                     else 'NON_QRS_OR_CLOSE_QRS_UNRESOLVED')
    return items


def full_beat_windows(anchors: list[dict], width: int,
                      left_fraction: float = .36, right_fraction: float = .78,
                      fallback_rr: int = 175) -> list[dict]:
    """Build overlapping P-QRS-ST-T windows using only selected QRS anchors.

    These windows are for visual review; they do NOT assert waveform
    completeness and they do not rule out overlapping P/T in tachyarrhythmia.
    """
    if width < 20:
        raise ValueError('Image too narrow')
    if not (0.15 <= left_fraction <= 0.7 and 0.5 <= right_fraction <= 1.2):
        raise ValueError('Invalid analysis window ratios')
    cleaned = clean_candidates(anchors, width, merge_px=8)
    # No fixed beat spacing: a short-coupled real ventricular ectopic remains.
    xs = [p['x_pixel'] for p in cleaned]
    rr = [b-a for a,b in zip(xs, xs[1:]) if b>a]
    median_rr = int(sorted(rr)[len(rr)//2]) if rr else int(fallback_rr)
    median_rr = max(35, median_rr)
    windows=[]
    for i,p in enumerate(cleaned):
        x=p['x_pixel']
        prev_rr = x-xs[i-1] if i else median_rr
        next_rr = xs[i+1]-x if i+1<len(xs) else median_rr
        # Windows can overlap, but an overly large preceding gap should not
        # make the window swallow several unrelated complexes.
        before = max(20, min(round(prev_rr*left_fraction), round(median_rr*.8)))
        after = max(30, min(round(next_rr*right_fraction), round(median_rr*1.3)))
        x0=max(0,x-before)
        x1=min(width,x+after+1)
        if x0>=x or x>=x1:
            continue
        windows.append({
            'beat_id':f'B{len(windows)+1}',
            'anchor_x':x,'x0':x0,'x1':x1,
            'prev_rr_px':prev_rr,'next_rr_px':next_rr,
            'anchor_source':str(p.get('source','USER_SELECTED_QRS_CANDIDATE')),
            'anchor_qc':'USER_SELECTED_QRS_FOR_QA_NOT_GROUND_TRUTH',
            'coverage_warning': ('LEFT_TRUNCATED' if x0==0 else '') +
                (' RIGHT_TRUNCATED' if x1==width else ''),
            'beat_completeness':'UNVERIFIED',
        })
    return windows


def full_beat_prompt(lead: str, window: dict, image_size: tuple[int,int]) -> str:
    """Prompt with explicit anchor and no clinical reference landmarks."""
    anchor_local = window['anchor_x'] - window['x0']
    w,h = image_size
    return f'''You are visually localizing ECG landmarks for RESEARCH QA only.
Lead: {lead}. Cropped image dimensions: {w} x {h} px; pixel origin (0,0) TOP LEFT.
A human-selected UNVERIFIED QRS-event candidate is near LOCAL X={anchor_local} px.
That X is a temporal anchor, NOT a confirmed R peak or a morphology label.
Analyze ONLY the ONE suspected QRS near this anchor and its surrounding P-ST-T.
The T wave FOLLOWING that QRS belongs to this beat if visible, and MUST NOT
be counted as a second QRS event. Do not label a broad T-peak as R_PEAK.
Look to the left for P and to the right for J/ST/T, but events may overlap or be
truncated, particularly in short-coupled ectopy or tachyarrhythmia.
For QRS_ONSET/QRS_OFFSET, give the actual onset/end of depolarization if visible.
Return P_ONSET,P_PEAK,P_OFFSET,QRS_ONSET,Q_PEAK,R_PEAK,S_NADIR,QRS_OFFSET,
J_POINT,T_PEAK,T_END,BASELINE only if evidence is clear. R can be absent in QS.
Never invent a missing T_END or force a complete PQRST pattern.
Do not use clinician landmark measurements, RR-regularity or diagnosis guesses.
Return JSON exactly with fields {{"landmarks":[],"qrs_observed":null,
"t_observed":null,"notes":""}}. Populate landmarks ONLY with visual evidence,
using landmark_type, x_pixel, y_pixel and evidence. Never invent pixel values; omit uncertain landmarks.

'''


def validate_one_beat_landmarks(records: list[dict], window: dict) -> list[dict]:
    """Flag (not delete) timing-inconsistent AI landmarks.

    A true arrhythmia can violate ordinary morphology; therefore flags always
    remain REVIEW, never automatic proof of a false QRS.
    """
    anchor = window['anchor_x']
    width=max(1, window['x1']-window['x0'])
    limit=max(15, min(35, round(width*.18)))
    out=[]
    ptypes={'P_ONSET','P_PEAK','P_OFFSET'}
    ttypes={'T_PEAK','T_END','ST_POINT'}
    for r in records:
        r=dict(r); flags=list(r.get('qa_flags', []))
        x=r.get('x_pixel')
        kind=r.get('landmark_type')
        if isinstance(x,(int,float)):
            if kind in ('R_PEAK','Q_PEAK','S_NADIR') and abs(x-anchor)>limit:
                flags.append('QRS_EXTREMUM_FAR_FROM_SELECTED_ANCHOR')
            if kind in ptypes and x>anchor+limit:
                flags.append('P_WAVE_AFTER_QRS_ANCHOR_REVIEW')
            if kind in ttypes and x<anchor-limit:
                flags.append('T_OR_ST_BEFORE_QRS_ANCHOR_REVIEW')
        r['qa_flags']=list(dict.fromkeys(flags))
        r['qa_state']='LOCALIZATION_REVIEW' if flags else 'REVIEW_REQUIRED'
        out.append(r)
    return out


def candidate_overlay(roi: Image.Image,candidates:list[dict],selected_ids:list[str]) -> Image.Image:
    img=roi.convert('RGB').copy()
    d=ImageDraw.Draw(img)
    selected=set(selected_ids)
    for c in candidates:
        x=c['x_pixel']
        chosen=c['candidate_id'] in selected
        color=(20,150,85) if chosen else (199,123,30)
        d.line((x,0,x,min(12,roi.height-1)),fill=color,width=3 if chosen else 2)
        d.text((min(roi.width-45,x+3),3),c['candidate_id'],fill=color)
    return img
