"""Conservative ECG row-boundary proposals and mask QC, v1.3.4.

All coordinates are hypotheses; this never asserts waveform recognition.
Retains raw candidate evidence when grid/lead ownership is unresolved.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image, ImageDraw

VERSION = 'ecg-row-boundary-mask-qc-v1.3.4'


def ink_evidence(image: Image.Image):
    a = np.asarray(image.convert('RGB'), dtype=np.uint8)
    gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    # Local dark strokes relative to paper shading, not absolute gray threshold.
    bg = cv2.GaussianBlur(gray, (0, 0), max(5., min(a.shape[:2]) / 15.))
    highpass = np.maximum(bg - gray, 0)
    smooth = cv2.GaussianBlur(highpass, (0, 0), 1.0)
    r = a[:, :, 0].astype(np.float32)
    g = a[:, :, 1].astype(np.float32)
    b = a[:, :, 2].astype(np.float32)
    colored = (r - .5 * (g + b) > 7) & (r > g + 4)
    threshold = max(3.5, float(np.percentile(smooth, 88)) * .60)
    raw = (smooth > threshold)
    # Exclude obvious pink grid from *proposals* only; not from final signal.
    evidence = raw & (~colored | (smooth > threshold * 2.3))
    return evidence, raw


def propose_row_refinement(image: Image.Image, proposal: dict, *, max_shift_fraction=.028):
    """Suggest moving interior row boundaries away from non-grid stroke activity.

    Optimize a valley in non-grid ink score near each existing boundary.
    Evidence weakness / conflict prevents automatic approval. Columns remain fixed.
    """
    ys = list(map(int, proposal['row_boundaries']))
    xs = list(map(int, proposal['column_boundaries']))
    w, h = image.size
    ink, raw = ink_evidence(image)
    # Ignore most printed text near paper top and calibration at each cell edge.
    xkeep = np.zeros(w, bool)
    for a,b in zip(xs[:-1], xs[1:]):
        margin = max(5, int((b-a)*.12))
        if b-a > 2*margin: xkeep[a+margin:b-margin] = True
    # Horizontal projection weighted by independent columns with dark strokes;
    # grid rows affect almost every column and are penalized below.
    restricted = ink[:,xkeep]
    if restricted.shape[1] == 0: raise ValueError('No usable columns for row review')
    row_fraction = restricted.mean(axis=1).astype(np.float32)
    row_score = cv2.GaussianBlur(row_fraction.reshape(-1,1), (1,7), 0).ravel()
    out = [ys[0]]
    events=[]
    max_shift=max(4,round(h*max_shift_fraction))
    min_row=max(10,round(h*.07))
    for i in range(1,len(ys)-1):
        y=ys[i]
        lo=max(ys[i-1]+min_row, y-max_shift)
        hi=min(ys[i+1]-min_row, y+max_shift)
        if hi<=lo:
            out.append(y);events.append({'boundary':i,'old_y':y,'proposed_y':y,'status':'REVIEW','reason':'no_safe_search_window'})
            continue
        candidates=np.arange(lo,hi+1)
        # Candidate crossing with little nearby stroke activity.
        radius=max(2,round(h*.005))
        costs=np.array([np.mean(row_score[max(0,z-radius):min(h,z+radius+1)]) for z in candidates])
        # A mild penalty prevents arbitrary large movement in featureless regions.
        penalized=costs + .0025*np.abs(candidates-y)/max(1,max_shift)
        best=int(candidates[int(np.argmin(penalized))])
        old_cost=float(np.mean(row_score[max(0,y-radius):min(h,y+radius+1)]))
        best_cost=float(np.min(costs))
        improvement=max(0.,(old_cost-best_cost)/max(old_cost, .001))
        # If little evidence or no improvement, preserve the manually/geometry-selected boundary.
        move=best if improvement >= .18 and old_cost>.005 else y
        status='REVIEW'  # Never PASS on projection heuristics alone.
        out.append(move)
        events.append({'boundary':i,'old_y':y,'proposed_y':move,'suggested_delta_px':move-y,
                       'status':status,'reason':'projection_valley_review' if move!=y else 'unchanged_insufficient_evidence',
                       'relative_activity_reduction':round(improvement,3)})
    out.append(ys[-1])
    return {'version':VERSION,'row_boundaries_original':ys,'row_boundaries_suggested':out,
            'boundary_events':events,'status':'REVIEW',
            'note':'Conservative suggestions only; never auto-approve morphology or measurements.'}


def mask_qc(image: Image.Image, segmentation: dict, *, lead: str, edge_fraction=.095):
    masks=segmentation['masks']; h,w=np.asarray(image).shape[:2]
    candidate=np.asarray(masks['ecg_ink_candidate'])>0
    raw=np.asarray(masks['raw_dark_candidate'])>0
    uncertain=np.asarray(masks['uncertain_intersection'])>0
    grid=np.asarray(masks['confident_colored_grid'])>0
    # Inspect candidate geometry: long horizontal/vertical straight lines are
    # suspicious but could be baseline or a tall QRS. This is a REVIEW indicator.
    horizontal=cv2.morphologyEx(candidate.astype(np.uint8),cv2.MORPH_OPEN,
        np.ones((1,max(13,round(w*.12))),np.uint8))>0
    vertical=cv2.morphologyEx(candidate.astype(np.uint8),cv2.MORPH_OPEN,
        np.ones((max(11,round(h*.30)),1),np.uint8))>0
    straight=horizontal|vertical
    frac=float(candidate.mean())
    straight_fraction=float(straight.sum()/max(1,candidate.sum()))
    edge_px=max(2,round(h*edge_fraction))
    top_count=int(candidate[:edge_px].sum())
    bottom_count=int(candidate[-edge_px:].sum())
    inner_count=int(candidate[edge_px:-edge_px].sum()) if h>2*edge_px else 0
    # These are observational red flags, not proof that cardiac ink is clipped.
    edge_risk=float((top_count+bottom_count)/max(1,int(candidate.sum())))
    support=float(np.mean(np.any(candidate,axis=0)))
    uncertain_fraction=float(uncertain.mean())
    reasons=[]
    if frac<.003:reasons.append('insufficient_ink_candidate')
    if frac>.115:reasons.append('dense_ink_possible_grid')
    if straight_fraction>.35:reasons.append('straight_line_contamination_suspected')
    if uncertain_fraction>.12:reasons.append('large_uncertain_intersection')
    if edge_risk>.24:reasons.append('waveform_or_artifact_near_roi_edge')
    if support<.35:reasons.append('fragmented_or_sparse_column_support')
    if not reasons:reasons.append('signal_identity_unverified')
    return {'lead':lead,'status':'REVIEW','reasons':reasons,
        'metrics':{'candidate_fraction':round(frac,5),'raw_dark_fraction':round(float(raw.mean()),5),
          'confident_grid_fraction':round(float(grid.mean()),5),
          'uncertain_fraction':round(uncertain_fraction,5),
          'straight_candidate_fraction':round(straight_fraction,5),
          'edge_candidate_fraction':round(edge_risk,5),'column_support_fraction':round(support,5),
          'candidate_pixels':int(candidate.sum()),'edge_candidate_pixels':top_count+bottom_count,
          'inner_candidate_pixels':inner_count},
        'interpretation':'Engineering screening only; cannot certify waveform ownership or QRS integrity.'}


def qc_overlay(image: Image.Image, segmentation: dict, qc: dict):
    rgb=np.asarray(image.convert('RGB')).copy();h,w=rgb.shape[:2]
    cand=np.asarray(segmentation['masks']['ecg_ink_candidate'])>0
    uncertain=np.asarray(segmentation['masks']['uncertain_intersection'])>0
    rgb[cand]=(.55*rgb[cand]+.45*np.array([14,212,125])).astype(np.uint8)
    rgb[uncertain]=(.60*rgb[uncertain]+.40*np.array([245,154,40])).astype(np.uint8)
    out=Image.fromarray(rgb); d=ImageDraw.Draw(out)
    edge=max(2,round(h*.095));d.line((0,edge,w,edge),fill='#e13030',width=2)
    d.line((0,h-edge,w,h-edge),fill='#e13030',width=2)
    return out
