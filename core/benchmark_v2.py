"""ECG v2 benchmark contract, safe import, and one-to-one evaluation.

Research QA only. Never use a prediction as a reference label. Pixel units
are not time or voltage units without a verified calibration.
"""
from __future__ import annotations
import hashlib
import io
import json
import math
import zipfile
from typing import Any

VALID_TYPES = {'R_PEAK','S_NADIR','QS_NADIR','QRS_FIDUCIAL','QRS_EVENT'}

def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False

def import_validation_zip(blob: bytes) -> dict:
    """Import prior QA without promoting incomplete annotations to ground truth."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        names = set(z.namelist())
        if 'roi_original_v15.png' not in names:
            raise ValueError('Validation ZIP lacks roi_original_v15.png')
        candidates = ['rpeak_reference_v153.json','rpeak_reference_v152.json']
        name = next((n for n in candidates if n in names), None)
        if not name:
            raise ValueError('Validation ZIP lacks reference JSON')
        raw = json.loads(z.read(name))
        from PIL import Image
        im = Image.open(io.BytesIO(z.read('roi_original_v15.png'))).convert('RGB')
        w,h = im.size
        points = []
        for i,p in enumerate(raw.get('reference_points') or []):
            x,y = p.get('x_pixel'),p.get('y_pixel')
            if not (_finite(x) and _finite(y) and 0 <= float(x)<w and 0 <= float(y)<h):
                continue
            typ = p.get('fiducial_type') or 'R_PEAK'
            if typ not in VALID_TYPES:
                typ='QRS_FIDUCIAL'
            points.append({'id':str(p.get('reference_id') or f'R{i+1:03d}'),
                           'x':float(x),'y':float(y),'type':typ,'source':'HUMAN',
                           'verified':False})
        windows = raw.get('annotation_windows') or {}
        reviewed = windows.get('windows') or []
        all_reviewed = bool(reviewed) and all(bool(a.get('reviewed')) for a in reviewed)
        complete = bool(raw.get('reference_complete')) and all_reviewed
        # Never auto-upgrade from prior UI's checkbox alone.
        roi_box = raw.get('roi_box')
        identity = f"{raw.get('image_sha256_12','unknown')}|{raw.get('lead','?')}|{roi_box}|{w}x{h}"
        case_id = 'IMPORTED_'+hashlib.sha256(identity.encode()).hexdigest()[:12]
        record = {'schema':'ecg-benchmark-v2.0','case_id':case_id,
                  'source_image_name':raw.get('image_name'),'source_image_sha256_12':raw.get('image_sha256_12'),
                  'lead':raw.get('lead'),'layout':raw.get('layout'),'roi_box':roi_box,
                  'roi_sha256':hashlib.sha256(z.read('roi_original_v15.png')).hexdigest(),
                  'roi_width':w,'roi_height':h,'reference_type':'fiducials',
                  'reference_points':points,'all_windows_reviewed':all_reviewed,
                  'reference_complete':complete,'review_status':'UNVERIFIED',
                  'dataset_split':'UNASSIGNED','development_exposed':True,'notes':'Imported human marks; complete only if all windows reviewed and confirmed.'}
        return {'record':record,'roi_image':im}

def validate_reference(rec:dict) -> list[str]:
    errors=[]
    w,h=rec.get('roi_width',0),rec.get('roi_height',0)
    if not isinstance(w,int) or not isinstance(h,int) or w<=0 or h<=0:errors.append('invalid ROI dimensions')
    seen=set()
    for p in rec.get('reference_points',[]):
        if p.get('id') in seen: errors.append('duplicate reference id')
        seen.add(p.get('id'))
        if p.get('type') not in VALID_TYPES:errors.append('invalid fiducial type')
        if not (_finite(p.get('x')) and _finite(p.get('y')) and 0<=float(p['x'])<w and 0<=float(p['y'])<h):
            errors.append('reference outside ROI')
    if rec.get('reference_complete') and not rec.get('all_windows_reviewed'):
        errors.append('complete flag without reviewed windows')
    if rec.get('dataset_split') not in ('TRAIN','VALIDATION','TEST','UNASSIGNED'):
        errors.append('unknown split')
    return errors

def match_one_to_one(reference:list[dict],predictions:list[dict],x_tol:float=10,y_tol:float|None=None):
    """Maximum-cardinality then lowest-distance bipartite matching.

    For small/medium ECG strips via scipy.optimize.linear_sum_assignment.
    Predictions must be independent of the reference list.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    n,m=len(reference),len(predictions)
    if not n or not m:
        return [],list(range(n)),list(range(m))
    # Unmatched cost > any valid match, invalid edge > 2*unmatched.
    invalid=1_000_000.0
    unmatched=1000.0
    C=np.full((n+m,n+m),invalid,dtype=float)
    for i,r in enumerate(reference):
        for j,p in enumerate(predictions):
            dx=abs(float(r['x'])-float(p['x']))
            dy=abs(float(r['y'])-float(p['y'])) if y_tol is not None and r.get('y') is not None and p.get('y') is not None else float('inf') if y_tol is not None else 0.0
            if dx<=x_tol and (y_tol is None or dy<=y_tol):
                C[i,j]=dx/max(x_tol,1)+(dy/max(y_tol,1) if y_tol is not None else 0)
    for i in range(n):C[i,m+i]=unmatched
    for j in range(m):C[n+j,j]=unmatched
    C[n:,m:]=0
    row,col=linear_sum_assignment(C)
    matches=[{'reference_index':int(i),'prediction_index':int(j),
              'dx':round(float(predictions[j]['x'])-float(reference[i]['x']),3),
              'dy':round(float(predictions[j]['y'])-float(reference[i]['y']),3) if predictions[j].get('y') is not None and reference[i].get('y') is not None else None}
             for i,j in zip(row,col) if i<n and j<m and C[i,j]<unmatched]
    matched_r={q['reference_index'] for q in matches};matched_p={q['prediction_index'] for q in matches}
    return matches,[i for i in range(n) if i not in matched_r],[i for i in range(m) if i not in matched_p]

def evaluate(record:dict,predictions:list[dict],x_tol:float=10,y_tol:float|None=None) -> dict:
    """Exhaustive detection metrics require completed verified reference and held-out test.

    Fiducial x/y localization can be compared only when types align. QRS event
    detection is scored separately from r/S/QS component localization.
    """
    errors=validate_reference(record)
    ref=record.get('reference_points',[])
    # QRS event timing can use any verified QRS landmark, but no morphology inference.
    match,fnidx,fpidx=match_one_to_one(ref,predictions,x_tol,y_tol)
    base={'status':'REVIEW','reference_count':len(ref),'prediction_count':len(predictions),
          'matched_count':len(match),'matches':match,
          'unmatched_reference_indices':fnidx,'unmatched_prediction_indices':fpidx,
          'x_tolerance_px':x_tol,'y_tolerance_px':y_tol,'errors':errors}
    eligible=(not errors and bool(record.get('reference_complete')) and
              bool(record.get('all_windows_reviewed')) and
              record.get('review_status')=='VERIFIED' and record.get('dataset_split')=='TEST' and
              not record.get('development_exposed',True))
    if not eligible:
        base['reason']='No sensitivity/precision: requires complete, reviewed, human-verified, held-out TEST reference not used to develop the detector.'
        base['sensitivity']=None;base['precision']=None;base['f1']=None
        return base
    tp=len(match);fn=len(fnidx);fp=len(fpidx)
    recall=tp/(tp+fn) if tp+fn else None
    precision=tp/(tp+fp) if tp+fp else None
    f1=2*recall*precision/(recall+precision) if recall is not None and precision is not None and (recall+precision)>0 else None
    return {**base,'status':'SCORED','tp':tp,'fp':fp,'fn':fn,
            'sensitivity':recall,'precision':precision,'f1':f1}
