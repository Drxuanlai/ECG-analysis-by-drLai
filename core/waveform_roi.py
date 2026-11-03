"""v1.3.1: conservative extraction ROI refinement, separate from layout ROI.

Important: evidence-based suggestions are REVIEW only; overlap is explicit and
no signal measurements/diagnoses are authorized by the geometry engine.
"""
from __future__ import annotations
from typing import Any
import numpy as np
import cv2
from PIL import Image, ImageDraw

VERSION='ecg-core-lead-roi-v1.3.1'


def _dark_stroke_map(im: Image.Image) -> np.ndarray:
    """Local black/neutral trace evidence; suppress slowly varying gray paper.

    This is NOT a waveform segmentation mask. The map is ONLY used to check
    edge proximity and propose which ROI needs extra review.
    """
    rgb=np.asarray(im.convert('RGB'),dtype=np.uint8)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    f=gray.astype(np.float32)
    bg=cv2.GaussianBlur(f,(0,0),4.5)
    residual=np.maximum(0,bg-f)
    # Median + MAD robust to overall dark photographs.
    med=float(np.median(residual));mad=float(np.median(np.abs(residual-med)))
    threshold=max(5.0,med+3*max(1.0,mad))
    # Positive excess of red suppresses the pink grid; neutral dark ink kept.
    a=rgb.astype(np.float32)
    red=np.maximum(0,a[:,:,0]-0.5*(a[:,:,1]+a[:,:,2]))
    neutral=(red<10)|(f<75)
    m=((residual>=threshold)&neutral).astype(np.uint8)
    return m


def _vertical_runs(mask:np.ndarray, xa:int,xb:int, ylo:int,yhi:int, min_len:int)->list:
    """Find significant near-vertical dark runs in window, in source coordinates."""
    h,w=mask.shape
    xa=max(0,xa);xb=min(w,xb);ylo=max(0,ylo);yhi=min(h,yhi)
    if xb<=xa or yhi<=ylo:return []
    out=[]
    # Straight line runs are a weak hypothesis only; grid lines also exist.
    # Erode horizontal red grid by requiring occupancy in 3-pixel x window.
    for x in range(xa,xb,2):
        col=mask[ylo:yhi,max(0,x-1):min(w,x+2)]
        if col.shape[1]==0:continue
        v=np.any(col>0,axis=1)
        starts=np.flatnonzero(v & ~np.r_[False,v[:-1]])
        ends=np.flatnonzero(v & ~np.r_[v[1:],False])
        for s,e in zip(starts,ends):
            if e-s+1>=min_len:out.append((x,ylo+int(s),ylo+int(e),int(e-s+1)))
    return out


def refine_waveform_rois(im:Image.Image, rois:list[dict], *, vertical_pad_ratio:float=.09,
                         max_expand_ratio:float=.19, per_lead:dict|None=None)->list[dict]:
    """Derive independent waveform extraction boxes from unmodified layout boxes.

    Extraction ROI can overlap adjacent rows when deep QRS extends across a
    layout boundary. That overlap is flagged; it must NOT silently be counted
    as two independent lead traces. Manual extraction overrides don't change
    layout identity. No x expansion across time-column borders is performed.
    """
    w,h=im.size
    stroke=_dark_stroke_map(im)
    manual=per_lead or {}
    out=[]
    for v in rois:
        layout=list(map(int,v['box'])); x0,y0,x1,y1=layout
        lead=v['lead'];
        if lead=='II-rhythm':
            out.append({**v,'layout_box':layout,'waveform_box':layout,
                        'refinement_status':'REVIEW','refinement_reasons':['rhythm_roi_not_refined'],
                        'vertical_edge_evidence':{'upper':0,'lower':0},'waveform_box_source':'layout'})
            continue
        height=y1-y0
        base_pad=max(4,int(round(vertical_pad_ratio*height)))
        max_expand=max(base_pad,int(round(max_expand_ratio*height)))
        # Ignore labels at first 10% and calibration near final 12% of a column.
        xa=x0+int(.12*(x1-x0));xb=x1-int(.16*(x1-x0))
        # Detect conspicuous vertical traces approaching an edge. Require run
        # length relative to ROI, and multiple distinct x positions to avoid
        # single grid lines. This is a hazard flag, not guaranteed QRS identity.
        min_run=max(8,int(.085*height))
        candidates=_vertical_runs(stroke,xa,xb,max(0,y0-max_expand),min(h,y1+max_expand),min_run)
        upper=[z for z in candidates if z[1]<y0+int(.085*height) and z[2]>y0-int(.02*height)]
        lower=[z for z in candidates if z[2]>y1-int(.085*height) and z[1]<y1+int(.02*height)]
        # Distinct x clusters indicate real recurrent waveform strokes. Grid
        # residual can also trigger => REVIEW always, never auto PASS.
        def clusters(z):
            xx=sorted(set(t[0] for t in z));n=0;prev=-999
            for q in xx:
                if q-prev>max(6,int(.015*(x1-x0))):n+=1
                prev=q
            return n
        cu=clusters(upper);cl=clusters(lower)
        up=base_pad;down=base_pad
        if cu>=2:
            up=min(max_expand,max(base_pad, y0-min(q[1] for q in upper)+base_pad//2))
        if cl>=2:
            down=min(max_expand,max(base_pad,max(q[2] for q in lower)-y1+base_pad//2))
        proposed=[x0,max(0,y0-up),x1,min(h,y1+down)]
        reasons=['expanded_vertical_for_boundary_safety']
        if cu>=2:reasons.append('possible_upper_waveform_boundary_contact')
        if cl>=2:reasons.append('possible_lower_waveform_boundary_contact')
        if not candidates:reasons.append('no_reliable_vertical_stroke_evidence')
        if lead in manual:
            requested=list(map(int,manual[lead]))
            if not (0<=requested[0]<requested[2]<=w and 0<=requested[1]<requested[3]<=h):
                raise ValueError('Invalid manual waveform box for '+lead)
            if requested[0]<x0 or requested[2]>x1:
                raise ValueError('Waveform ROI may not cross time-column boundaries: '+lead)
            proposed=requested;reasons.append('manual_waveform_override')
        out.append({**v,'layout_box':layout,'waveform_box':proposed,
                    'waveform_box_source':'manual' if lead in manual else 'edge_safe_auto',
                    'refinement_status':'MANUAL' if lead in manual else 'REVIEW',
                    'refinement_reasons':reasons,
                    'vertical_edge_evidence':{'upper':cu,'lower':cl}})
    # Record extraction overlap, never silently treat as disjoint traces.
    for i,v in enumerate(out):
        if v['lead']=='II-rhythm':continue
        overlaps=[]
        a=v['waveform_box']
        for j,u in enumerate(out):
            if j==i or u['lead']=='II-rhythm' or u['col']!=v['col']:continue
            b=u['waveform_box']
            ox=max(0,min(a[2],b[2])-max(a[0],b[0]))
            oy=max(0,min(a[3],b[3])-max(a[1],b[1]))
            if ox>0 and oy>0:
                l=max(a[0],b[0]);r=min(a[2],b[2]);top=max(a[1],b[1]);bottom=min(a[3],b[3])
                # Small padding overlap is intentional. Flag for review only
                # when image evidence is present within that shared band.
                overlap_ink=float(np.mean(stroke[top:bottom,l:r])) if r>l and bottom>top else 0.0
                overlaps.append({'lead':u['lead'],'overlap_px':[ox,oy],
                                 'shared_ink_fraction':round(overlap_ink,4),
                                 'possible_trace_collision':bool(overlap_ink>=.025)})
        v['waveform_overlaps']=overlaps
        if any(z['possible_trace_collision'] for z in overlaps):v['refinement_reasons'].append('adjacent_lead_signal_collision_possible')
    return out


def render_dual_overlay(im:Image.Image,rois:list[dict], max_width:int=1800)->Image.Image:
    out=im.convert('RGB').copy();d=ImageDraw.Draw(out)
    ww=out.width;t=max(2,ww//1100)
    for v in rois:
        a=v['layout_box'];b=v['waveform_box'];lead=v['lead']
        # Thin layout solid line. Thicker extraction outline with corner ticks.
        d.rectangle([a[0],a[1],a[2]-1,a[3]-1],outline='#2f9ed9',width=t)
        if lead!='II-rhythm':
            d.rectangle([b[0],b[1],b[2]-1,b[3]-1],outline='#f59b2d',width=t+1)
        else:
            d.rectangle([b[0],b[1],b[2]-1,b[3]-1],outline='#b744c8',width=t+1)
        x=max(0,min(ww-1,a[0]+2));y=max(0,min(out.height-1,a[1]+2))
        d.rectangle([x,y,min(ww-1,x+8*len(lead)+9),min(out.height-1,y+17)],fill='#1a6b99')
        d.text((x+3,y+2),lead,fill='white')
    if out.width>max_width:out.thumbnail((max_width,max_width),Image.Resampling.LANCZOS)
    return out
