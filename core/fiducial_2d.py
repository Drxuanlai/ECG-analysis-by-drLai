"""v1.5.6: Auditable two-dimensional ECG fiducial QA.

Research only. A timestamp-like X agreement does not verify a waveform point.
No model has permission here to change human labels or call a candidate a verified QRS.
"""
from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image, ImageDraw

VERSION = 'ecg-core-v1.5.6-fiducial-2d-review'
REFERENCE_KINDS = ('R_PEAK', 'S_NADIR', 'QS_NADIR', 'QRS_FIDUCIAL')


def normalize_references(references: list[dict]) -> list[dict]:
    """Add classification without changing the original location or legacy marks."""
    result = []
    for i, original in enumerate(references):
        p = dict(original)
        kind = str(p.get('fiducial_type') or 'R_PEAK').upper()
        if kind not in REFERENCE_KINDS:
            raise ValueError('Unsupported reference kind: ' + kind)
        p['fiducial_type'] = kind
        p.setdefault('reference_id', f'R{i+1:03d}')
        result.append(p)
    return result


def _finite(v: Any) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (ValueError, TypeError, OverflowError):
        return False


def evaluate_fiducial_2d(
    ranked_report: dict, references: list[dict], *,
    tolerance_x_px: float = 5., tolerance_y_px: float = 8.,
    reference_complete: bool = False,
) -> dict:
    """One-to-one 2D matching with distinct X-only and XY classifications.

    All ranked candidates, priority and unresolved, are considered in the
    *raw 2D* review; priority matches are reported separately. To avoid a
    greedy X-only match consuming a better XY candidate, preference is given
    to 2D-valid pairs first, followed by X-only diagnostic pairs.
    """
    if not (0 < tolerance_x_px <= 100 and 0 < tolerance_y_px <= 100):
        raise ValueError('Invalid pixel tolerances')
    refs = normalize_references(references)
    hints = list(ranked_report.get('hints') or [])
    proposals = []
    for i, p in enumerate(hints):
        x = p.get('x_pixel')
        y = p.get('y_hint_pixel', p.get('y_pixel'))
        if not (_finite(x) and _finite(y)):
            continue
        for j, r in enumerate(refs):
            if not (_finite(r.get('x_pixel')) and _finite(r.get('y_pixel'))):
                continue
            dx = float(x) - float(r['x_pixel'])
            dy = float(y) - float(r['y_pixel'])
            if abs(dx) <= tolerance_x_px:
                spatial = abs(dy) <= tolerance_y_px
                # 2D matches first, then nearest scaled error for uniqueness.
                key = (0 if spatial else 1,
                       (dx/tolerance_x_px)**2 + (dy/tolerance_y_px)**2, abs(dx), i, j)
                proposals.append((key, i, j, dx, dy, spatial))
    occupied_hints, occupied_refs = set(), set()
    matched=[]
    for _,i,j,dx,dy,is_spatial in sorted(proposals, key=lambda z:z[0]):
        if i in occupied_hints or j in occupied_refs:
            continue
        occupied_hints.add(i)
        occupied_refs.add(j)
        h,r = hints[i],refs[j]
        kind = r['fiducial_type']
        row = {
            'reference_id':r['reference_id'],
            'reference_type':kind,
            'hint_id':h.get('hint_id'),
            'hint_status':h.get('status','UNRESOLVED'),
            'x_reference_px':float(r['x_pixel']),
            'y_reference_px':float(r['y_pixel']),
            'x_predicted_px':float(h['x_pixel']),
            'y_predicted_px':float(h.get('y_hint_pixel',h.get('y_pixel'))),
            'error_x_px':round(dx,3),
            'error_y_px':round(dy,3),
            'euclidean_error_px':round(float(np.hypot(dx,dy)),3),
            'position_qc':'XY_WITHIN_TOLERANCE' if is_spatial else 'X_ONLY_Y_MISMATCH',
            'morphology_qc':'UNVERIFIED',
        }
        # Two-dimensional proximity does not prove that a candidate is the
        # correct wave component (R/S/QS etc). No clinical truth is invented.
        matched.append(row)
    spatial=[m for m in matched if m['position_qc']=='XY_WITHIN_TOLERANCE']
    prioritized=[m for m in spatial if m['hint_status']=='PRIORITY_REVIEW']
    time_only=[m for m in matched if m['position_qc']=='X_ONLY_Y_MISMATCH']
    matched_ref_ids={m['reference_id'] for m in matched}
    xy_ref_ids={m['reference_id'] for m in spatial}
    out={
        'version': VERSION,
        'status': 'REVIEW_ONLY',
        'reference_complete':bool(reference_complete),
        'reference_count':len(refs),
        'raw_hint_count':len(hints),
        'priority_hint_count':sum(h.get('status')=='PRIORITY_REVIEW' for h in hints),
        'xy_within_tolerance_count':len(spatial),
        'priority_xy_within_tolerance_count':len(prioritized),
        'x_only_y_mismatch_count':len(time_only),
        'unmatched_reference_ids':[r['reference_id'] for r in refs if r['reference_id'] not in matched_ref_ids],
        'no_xy_match_reference_ids':[r['reference_id'] for r in refs if r['reference_id'] not in xy_ref_ids],
        'tolerances_px':{'x':tolerance_x_px,'y':tolerance_y_px},
        'matches':matched,
        'metrics':None,
        'notes':[
            'X-only match is not a valid 2D localization.',
            'XY match checks geometric proximity only, NOT correct P/Q/R/S/T identification.',
            'QS or S-nadir is not an R-peak. Reference types are kept distinct.',
            'Unresolved machine hints are preserved; never re-label human ARTIFACT.',
            'No RR template, patient diagnosis or ground truth labels enter ranking.',
            'Pixel distances are not milliseconds without validated calibration.',
        ],
    }
    if reference_complete and refs:
        # Metrics are explicitly *geometric review metrics*, not clinical
        # sensitivity/precision of QRS detection.
        out['metrics']={
            'raw_xy_reference_coverage':len(spatial)/len(refs),
            'priority_xy_reference_coverage':len(prioritized)/len(refs),
            'priority_xy_match_fraction':len(prioritized)/max(1,out['priority_hint_count']),
        }
        out['status']='EXHAUSTIVE_GEOMETRIC_QA_ONLY'
    return out


def fiducial_2d_overlay(image:Image.Image, ranked_report:dict,
                        references:list[dict], matches:dict) -> Image.Image:
    im=image.convert('RGB').copy()
    d=ImageDraw.Draw(im)
    for p in ranked_report.get('hints',[]):
        if not (_finite(p.get('x_pixel')) and _finite(p.get('y_hint_pixel',p.get('y_pixel')))):
            continue
        x=round(float(p['x_pixel'])); y=round(float(p.get('y_hint_pixel',p.get('y_pixel'))))
        c='#168dc7' if p.get('status')=='PRIORITY_REVIEW' else '#a688c6'
        d.ellipse((x-3,y-3,x+3,y+3),outline=c,width=2)
    refs=normalize_references(references)
    for r in refs:
        x=round(float(r['x_pixel']));y=round(float(r['y_pixel']))
        c='#19b26a' if r['fiducial_type']=='R_PEAK' else '#e59636'
        d.line((x-5,y,x+5,y), fill=c, width=2)
        d.line((x,y-5,x,y+5), fill=c, width=2)
    for m in matches.get('matches',[]):
        if m['position_qc']=='X_ONLY_Y_MISMATCH':
            x1,y1=m['x_reference_px'],m['y_reference_px']
            x2,y2=m['x_predicted_px'],m['y_predicted_px']
            d.line((round(x1),round(y1),round(x2),round(y2)),fill='#e84b47',width=2)
    return im
