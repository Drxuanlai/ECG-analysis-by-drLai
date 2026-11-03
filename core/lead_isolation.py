"""Research QA: conservative per-lead pixel ownership hypotheses.

This module is *not* ECG digitization, beat detection, or proof that a pixel is
cardiac ink. It prevents pixels in overlapping extraction ROIs from being
silently counted in two leads. Ambiguous pixels remain unassigned.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import cv2
from PIL import Image, ImageDraw
from core.waveform_roi import _dark_stroke_map

VERSION = 'ecg-core-lead-isolation-v1.3.2'


def _clipped_box(box, width, height):
    x0,y0,x1,y1=map(int,box)
    return max(0,x0),max(0,y0),min(width,x1),min(height,y1)


def isolate_leads(im:Image.Image, rois:list[dict], *, ambiguous_margin_ratio:float=.04)->dict[str,Any]:
    """Return disjoint masks in full-image coordinates for 12 leads.

    Strong core anchors are from the layout cells. Ink connected to multiple
    lead cores, or pixels in competing extraction ROIs near a boundary, are
    marked ambiguous. No pixel may be owned by more than one lead.
    """
    if not (0 <= ambiguous_margin_ratio <= .2):
        raise ValueError('ambiguous_margin_ratio must be in [0, 0.2]')
    width,height=im.size
    entries=[q for q in rois if q.get('lead')!='II-rhythm']
    if len(entries)!=12 or len({q['lead'] for q in entries})!=12:
        raise ValueError('Expected exactly 12 unique ECG lead ROIs')
    ink=_dark_stroke_map(im).astype(bool)
    ownership=np.full((height,width),-1,np.int16)  # -1 no owner; -2 ambiguous
    supported=np.zeros((height,width),bool)
    diagnostics={}
    # Iterate independently by time-column; never compare different times.
    for col in sorted({int(q['col']) for q in entries}):
        members=sorted([(i,q) for i,q in enumerate(entries) if int(q['col'])==col],
                       key=lambda z:int(z[1]['row']))
        if len(members)!=3:raise ValueError('Each time column must contain exactly three leads')
        xlo=max(0,min(int(v['waveform_box'][0]) for _,v in members))
        xhi=min(width,max(int(v['waveform_box'][2]) for _,v in members))
        if xhi<=xlo:continue
        region_ink=ink[:,xlo:xhi].astype(np.uint8)
        # Only genuine pixels from the dark-stroke proposal are eligible.
        n,labels,stats,_=cv2.connectedComponentsWithStats(region_ink,8)
        region_owner=ownership[:,xlo:xhi]
        region_supported=supported[:,xlo:xhi]
        eligibility=[];cores=[]; centers=[]
        for idx,q in members:
            x0,y0,x1,y1=_clipped_box(q['waveform_box'],width,height)
            ex=np.zeros((height,xhi-xlo),bool)
            ex[y0:y1,max(0,x0-xlo):min(xhi-xlo,x1-xlo)]=True
            eligibility.append(ex)
            lx0,ly0,lx1,ly1=_clipped_box(q['layout_box'],width,height)
            span=max(2,ly1-ly0)
            # Narrow core excludes the geometric row boundary but leaves room
            # for genuine large excursions in the outer extraction ROI.
            pad=max(2,int(.17*span))
            co=np.zeros_like(ex)
            co[min(height,ly0+pad):max(0,ly1-pad),max(0,lx0-xlo):min(xhi-xlo,lx1-xlo)]=True
            cores.append(co)
            centers.append((ly0+ly1)/2.)
        eligible_count=sum(z.astype(np.uint8) for z in eligibility)
        candidate=(region_ink>0)&(eligible_count>0)
        region_supported[:]=candidate
        margin=max(2,int(np.median([q['layout_box'][3]-q['layout_box'][1] for _,q in members])*ambiguous_margin_ratio))
        boundaries=[(int(members[j][1]['layout_box'][3])+int(members[j+1][1]['layout_box'][1]))/2 for j in (0,1)]
        near_boundary=np.zeros_like(candidate)
        for yy in boundaries:
            near_boundary[max(0,int(yy)-margin):min(height,int(yy)+margin+1),:]=True
        # Component-level core evidence. Connected grid, text and adjacent traces
        # may bridge two rows; such components must not be confidently assigned.
        core_label_hits=[np.unique(labels[z&(region_ink>0)]) for z in cores]
        hit_by_component=[set() for _ in range(n)]
        for local_idx,vals in enumerate(core_label_hits):
            for cc in vals:
                if int(cc)>0: hit_by_component[int(cc)].add(local_idx)
        multi=np.array([len(v)>1 for v in hit_by_component],bool)
        multi[0]=False
        collision=candidate&multi[labels]
        unique_candidate=candidate&~collision
        # Fully competing ROI pixels near a row boundary are ambiguous even if
        # a component has a nominal core (critical for QRS crossing).
        uncertain=collision|(candidate&(eligible_count>1)&near_boundary)
        region_owner[uncertain]=-2
        for local_idx,(global_idx,q) in enumerate(members):
            owned=unique_candidate&eligibility[local_idx]&~uncertain
            # Components touching no protected core may be assigned only if the
            # pixel has exactly one eligible ROI; otherwise withhold ownership.
            touches=np.array([len(v)==1 and local_idx in v for v in hit_by_component],bool)
            touches[0]=False
            allowed=touches[labels]|((eligible_count==1)&(np.array([len(v)==0 for v in hit_by_component],bool)[labels]))
            owned &= allowed
            owned &= region_owner==-1
            region_owner[owned]=global_idx
        leftovers=candidate&(region_owner==-1)
        region_owner[leftovers]=-2
        for local_idx,(idx,q) in enumerate(members):
            potential=candidate&eligibility[local_idx]
            actual=int(np.sum(region_owner==idx))
            amb=int(np.sum((region_owner==-2)&potential))
            reasons=[]
            if actual==0:reasons.append('no_uniquely_attributed_ink')
            if amb:reasons.append('ambiguous_adjacent_lead_or_connected_ink')
            if not potential.any():reasons.append('no_ink_candidates_in_roi')
            diagnostics[q['lead']]={
                'status':'REVIEW', 'owned_ink_px':actual,
                'ambiguous_candidate_px':amb,
                'candidate_ink_px':int(potential.sum()),
                'ambiguous_fraction':round(amb/max(1,int(potential.sum())),4),
                'reasons':reasons,
                'layout_box':list(map(int,q['layout_box'])),
                'waveform_box':list(map(int,q['waveform_box'])),
                'column':col,
            }
    masks={q['lead']:(ownership==i).astype(np.uint8)*255 for i,q in enumerate(entries)}
    ambiguous=(ownership==-2).astype(np.uint8)*255
    assert all(not np.any((ownership==i)&(ownership==j)) for i in range(len(entries)) for j in range(i+1,len(entries)))
    return {'version':VERSION,'lead_masks':masks,'ambiguous_mask':ambiguous,
            'diagnostics':diagnostics,'candidate_mask':supported.astype(np.uint8)*255,
            'owner_index':ownership}


def isolation_overlay(im:Image.Image, result:dict, rois:list[dict], max_width:int=1800)->Image.Image:
    """Cyan=owned candidate; orange=withheld ambiguity; not a measured trace."""
    a=np.asarray(im.convert('RGB')).copy()
    for m in result['lead_masks'].values():
        on=m>0
        if np.any(on):a[on]=(a[on].astype(np.float32)*.35+np.array([10,205,226])*.65).astype(np.uint8)
    amb=result['ambiguous_mask']>0
    if np.any(amb):a[amb]=(a[amb].astype(np.float32)*.2+np.array([255,145,15])*.8).astype(np.uint8)
    out=Image.fromarray(a)
    d=ImageDraw.Draw(out)
    for q in rois:
        bx=q['layout_box'];c='#a4d3ff' if q['lead']!='II-rhythm' else '#b744c8'
        d.rectangle((bx[0],bx[1],bx[2]-1,bx[3]-1),outline=c,width=2)
    if out.width>max_width:out.thumbnail((max_width,max_width),Image.Resampling.LANCZOS)
    return out
