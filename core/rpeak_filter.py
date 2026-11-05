"""v1.5.4: morphology-aware R-like image hint triage, research only.

No rhythm regularity, human reference, diagnosis or verified beat count enters filtering.
All input hints are retained; 'PRIORITY_REVIEW' means a reviewer should inspect it,
not that it is a QRS. Ambiguous candidates are never silently deleted.
"""
from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageDraw

VERSION = 'ecg-core-v1.5.4-morphology-hint-triage'


def _prominence_image(image: Image.Image) -> np.ndarray:
    rgb = np.asarray(image.convert('RGB'), dtype=np.float32)
    # Use neutral/dark ink while avoiding forced RGB grid cancellation on gray paper.
    gray = cv2.cvtColor(rgb.astype('uint8'), cv2.COLOR_RGB2GRAY).astype(np.float32)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=9, sigmaY=3)
    return np.maximum(0., background - gray)


def filter_rpeak_hints(image: Image.Image, raw_hints: list[dict], *,
                       shortlist_fraction: float = .35) -> dict[str, Any]:
    """Rank all hint positions by LOCAL visual morphology, not RR or labels.

    shortlist_fraction is a REVIEW capacity control rather than a clinical
    decision threshold. A hint without strong evidence stays UNRESOLVED.
    """
    if not 0. < shortlist_fraction <= 1.: raise ValueError('Invalid shortlist fraction')
    w,h = image.size
    if w < 80 or h < 25: raise ValueError('Image ROI is too small')
    contrast = _prominence_image(image)
    entries=[]
    for hint in raw_hints:
        x=int(round(float(hint['x_pixel'])))
        if not 0 <= x < w: raise ValueError('Hint outside image')
        l=max(0,x-3);r=min(w,x+4)
        patch=contrast[:,l:r]
        # Central vertical ink compared with nearby baseline in the same row.
        row_peak=np.max(patch,axis=1)
        threshold=max(7.,float(np.percentile(row_peak,75))*.60)
        active=row_peak>=threshold
        # Number of consecutive supported y pixels; isolated grid crossings are weak.
        longest=0;run=0
        for hit in active:
            run=run+1 if hit else 0
            longest=max(longest,run)
        # Reject full-height lines only as a risk flag, not as verified artifact.
        extent=float(np.mean(active))
        central=float(np.quantile(row_peak, .92))
        # Local non-grid contrast prominence against flanking x locations.
        wings=[]
        for off in (-18,-13,13,18):
            xx=min(w-1,max(0,x+off))
            wings.append(float(np.quantile(contrast[:,xx],.92)))
        wing_med=float(np.median(wings))
        local_gain=max(0.,central-wing_med)
        # QRS shafts often extend vertically but do not fill the entire ROI.
        elongation=min(longest/max(5.,h*.19),1.)
        cover_penalty=max(0.,(extent-.50)/.25)
        flags=list(hint.get('flags') or [])
        edges = x < int(w*.025) or x >= w-int(w*.025)
        score=(.36*min(local_gain/25.,1.)+
               .34*elongation+
               .20*min(central/28.,1.)+
               .10*min(float(hint.get('strength_heuristic',0))/1.5,1.)-
               .22*min(cover_penalty,1.)-
               (.12 if edges else 0.))
        entries.append({'hint_id':hint.get('hint_id'), 'x_pixel':x,
            'y_hint_pixel':hint.get('y_pixel'), 'score_visual':round(float(score),4),
            'features':{'local_ink_gain':round(local_gain,2),
                        'max_vertical_run_px':int(longest),
                        'vertical_occupancy':round(extent,3),
                        'strong_ink_contrast':round(central,2)},
            'flags':flags+(['image_edge'] if edges else []),
            'status':'UNRESOLVED'})
    # Only suppress near-duplicate hint locations (not successive heartbeats).
    ranked=sorted(range(len(entries)),key=lambda i:(-entries[i]['score_visual'],entries[i]['x_pixel']))
    min_sep=max(7,min(16,int(w*.006)))
    selected=[]
    limit=max(1,int(np.ceil(len(entries)*shortlist_fraction))) if entries else 0
    for idx in ranked:
        entry=entries[idx]
        if len(selected)>=limit: break
        if all(abs(entry['x_pixel']-entries[j]['x_pixel'])>=min_sep for j in selected):
            selected.append(idx)
    for i in selected:entries[i]['status']='PRIORITY_REVIEW'
    return {'version':VERSION,'status':'REVIEW','roi_size':[w,h],
            'raw_hint_count':len(entries),'priority_count':len(selected),
            'unresolved_count':len(entries)-len(selected), 'hints':entries,
            'notes':['PRIORITY_REVIEW is NOT a verified QRS or R peak.',
                'All hints retained; reviewers must inspect unresolved regions.',
                'No RR regularity and no human labels used in filtering.',
                'Review fraction controls inspection workload, not clinical precision.']}


def filter_overlay(image:Image.Image, report:dict, references:list[dict]|None=None)->Image.Image:
    output=image.convert('RGB').copy();d=ImageDraw.Draw(output);h=output.height
    for p in report['hints']:
        x=int(p['x_pixel'])
        if p['status']=='PRIORITY_REVIEW':
            d.line((x, max(0,h-14),x,h-1),fill='#00bcd4',width=2)
            d.ellipse((x-3,5,x+3,11),fill='#00bcd4')
        else:
            d.line((x,h-7,x,h-1),fill='#9b67ae',width=1)
    for r in references or []:
        x,y=int(round(r['x_pixel'])),int(round(r['y_pixel']))
        d.line((x-6,y,x+6,y),fill='#05b967',width=2)
        d.line((x,y-6,x,y+6),fill='#05b967',width=2)
    return output


def compare_with_references(report:dict, references:list[dict], *,
                            tolerance_px:float=5., reference_complete:bool=False)->dict:
    """One-to-one evaluation only. Partial labels never yield sensitivity/precision."""
    if tolerance_px<=0:raise ValueError('Tolerance must be positive')
    available=[p for p in report['hints'] if p['status']=='PRIORITY_REVIEW']
    pairs=[]
    for i,p in enumerate(available):
        for j,r in enumerate(references):
            e=abs(float(p['x_pixel'])-float(r['x_pixel']))
            if e<=tolerance_px:pairs.append((e,i,j))
    used_i=set();used_j=set();matched=[]
    for error,i,j in sorted(pairs):
        if i in used_i or j in used_j:continue
        used_i.add(i);used_j.add(j)
        matched.append({'hint_id':available[i]['hint_id'],
                        'reference_id':references[j]['reference_id'],
                        'error_x_px':round(error,3)})
    result={'status':'REVIEW_ONLY', 'priority_hint_count':len(available),
        'reference_count':len(references),'spatially_matched_count':len(matched),
        'matches':matched,'unmatched_reference_ids':[r['reference_id']
            for j,r in enumerate(references) if j not in used_j],
        'reference_complete':bool(reference_complete),
        'metrics':None}
    if reference_complete and references:
        result['metrics']={'spatial_recall':len(matched)/len(references),
                           'spatial_precision':len(matched)/max(1,len(available))}
        result['status']='EXHAUSTIVE_SPATIAL_QA_ONLY'
    return result
