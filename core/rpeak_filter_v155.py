"""v1.5.5: conservative morphology-aware re-ranking of v1.5.4 image hints.

Research/annotation only. Never consumes human labels or rhythm regularity to rank.
No hint is removed; priorities are a review queue, not a verified QRS detector.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import cv2
from PIL import Image
from core.rpeak_filter import filter_rpeak_hints, _prominence_image

VERSION = "ecg-core-v1.5.5-rpeak-extremum-ranking"


def _estimated_baseline(hints: list[dict], height: int) -> tuple[float, bool]:
    """Estimate center of mostly quiet ink using LOWER strength hints (no human labels).

    If there is inadequate evidence do not pretend baseline was identified.
    """
    eligible = [p for p in hints if p.get('y_pixel') is not None and 0 <= float(p['y_pixel']) < height]
    if len(eligible) < 5:
        return (height / 2., False)
    eligible.sort(key=lambda p: float(p.get('strength_heuristic') or 0.))
    center = [float(p['y_pixel']) for p in eligible[:max(4, int(len(eligible) * .65))]]
    med = float(np.median(center))
    mad = float(np.median(np.abs(np.asarray(center) - med)))
    if mad > height * .28:
        return (height / 2., False)
    return (med, True)


def rank_rpeak_hints(image: Image.Image, raw_hints: list[dict], *, shortlist_fraction: float = .35) -> dict[str, Any]:
    """Preserve the original candidate set; re-rank using hint tip + local morphology.

    Scoring uses image and machine proposals ONLY. Vertical height is not penalized:
    tall positive or deep negative R/QS-like events remain eligible for review.
    """
    base = filter_rpeak_hints(image, raw_hints, shortlist_fraction=shortlist_fraction)
    if not raw_hints:
        base['version'] = VERSION
        return base
    w, h = image.size
    prominence = _prominence_image(image)
    gray = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2GRAY).astype(np.float32)
    baseline, reliable = _estimated_baseline(raw_hints, h)
    for p, original in zip(base['hints'], raw_hints):
        x = int(p['x_pixel'])
        yv = original.get('y_pixel')
        if yv is None:
            y = int(round(baseline))
        else:
            y = int(np.clip(round(float(yv)), 0, h-1))
        ys = slice(max(0,y-3), min(h,y+4))
        xs = slice(max(0,x-2), min(w,x+3))
        tip_support = float(np.max(prominence[ys,xs])) if prominence[ys,xs].size else 0.
        # Along x, a genuine sharp peak is darker than its immediate outer shoulders.
        # Values are robust to colored grid and JPEG blur but remain only heuristics.
        ylo, yhi = max(0,y-4),min(h,y+5)
        center_dark = float(np.mean(gray[ylo:yhi, max(0,x-2):min(w,x+3)]))
        wing_averages=[]
        for dx in (-13,-9,9,13):
            xx=int(np.clip(x+dx,0,w-1))
            wing_averages.append(float(np.mean(gray[ylo:yhi,max(0,xx-1):min(w,xx+2)])))
        side_contrast=max(0.,float(np.median(wing_averages))-center_dark)
        displacement = abs(y-baseline)/max(12.,h*.28)
        extreme_score=float(np.clip(displacement,0,1)) if reliable else .25
        strength=float(original.get('strength_heuristic') or 0.)
        strength_score=float(np.clip(strength/1.0,0,1))
        visual_strength=float(np.clip(tip_support/40.,0,1))
        x_contrast=float(np.clip(side_contrast/25.,0,1))
        # Main difference from v1.5.4: long vertical runs don't dominate ranking;
        # extreme crest + strong locally supported tip are rewarded, regardless of polarity.
        score=(.40*extreme_score+.27*strength_score+.19*visual_strength+
               .14*x_contrast)
        if x < int(w*.015) or x >= w-int(w*.015):
            score -= .07   # small edge risk only; does not delete candidates
        p['score_v155'] = round(float(score),4)
        p['baseline_estimate_y'] = round(baseline,1) if reliable else None
        p['features_v155'] = {'tip_y':y, 'tip_prominence':round(tip_support,2),
              'x_shoulders_contrast':round(side_contrast,2),
              'displacement_from_baseline':round(abs(y-baseline),1),
              'extremum_score':round(extreme_score,3),
              'strength_score':round(strength_score,3)}
        p['prior_v154_status'] = p['status']
        p['status'] = 'UNRESOLVED'
    # Deduplicate near-identical hints, never enforce an RR or heart-rate template.
    ranked = sorted(range(len(base['hints'])),
                    key=lambda i: (-base['hints'][i]['score_v155'], base['hints'][i]['x_pixel']))
    min_sep = max(7, min(16, int(w * .006)))
    limit=max(1,int(np.ceil(len(ranked)*shortlist_fraction)))
    selected=[]
    for i in ranked:
        if len(selected)>=limit:break
        if all(abs(base['hints'][i]['x_pixel']-base['hints'][j]['x_pixel'])>=min_sep for j in selected):
            selected.append(i)
    for i in selected:base['hints'][i]['status']='PRIORITY_REVIEW'
    base['version']=VERSION
    base['priority_count']=len(selected)
    base['unresolved_count']=len(base['hints'])-len(selected)
    base['ranking_evidence']={'baseline_estimated':reliable,
        'baseline_y':round(baseline,1) if reliable else None,
        'uses_rr_regularization':False, 'uses_human_annotations':False,
        'shortlist_is_not_detection':True}
    base['notes'] += [
        'v1.5.5 re-ranks tip extremity and local image evidence without removing unresolved hints.',
        'Positive/negative QRS, P/T waves, calibration or grid artifacts remain possible.',
        'Shortlist fraction is only reviewer capacity and does not represent clinical confidence.']
    return base
