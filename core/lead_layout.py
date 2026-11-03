"""ECG Core v1.3: conservative image/layout ROI hypotheses for 3x4 (+rhythm).

Research QA only. Does not alter pixel signal, generate clinical measurements, or
assume that a photographed print is a rectangular uniform grid.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import io
import re
import shutil
import subprocess
import csv
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

VERSION = 'ecg-core-layout-v1.3'
LAYOUT = (('I','aVR','V1','V4'), ('II','aVL','V2','V5'), ('III','aVF','V3','V6'))
EXPECTED = {label:(r,c) for r,row in enumerate(LAYOUT) for c,label in enumerate(row)}


def _image(x:Image.Image) -> Image.Image:
    if not isinstance(x,Image.Image):raise TypeError('Expected PIL Image')
    return x.convert('RGB')


def _read_labels(im:Image.Image, enabled=True):
    """Optional local Tesseract. OCR yields evidence only; never silently overrides layout."""
    if not enabled or shutil.which('tesseract') is None:return []
    w,h=im.size
    # OCR on mostly original resolution. Extra-strong contrast for light text.
    rgb=np.asarray(im.convert('RGB'))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(gray)
    results=[]
    for psm in (11,):
        b=io.BytesIO();Image.fromarray(clahe).save(b,format='PNG')
        try:
            p=subprocess.run(['tesseract','stdin','stdout','--psm',str(psm),'-l','eng','tsv'],input=b.getvalue(),capture_output=True,timeout=9,check=True)
            rows=csv.DictReader(io.StringIO(p.stdout.decode('utf8',errors='replace')),delimiter='\t')
            for q in rows:
                raw=(q.get('text') or '').strip(); s=re.sub('[^A-Za-z0-9]','',raw).upper()
                if s not in ('I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6'):continue
                try:
                    conf=float(q['conf']);x=int(q['left']);y=int(q['top']);bw=int(q['width']);bh=int(q['height'])
                except (ValueError,KeyError,TypeError):continue
                if conf<36 or bw<2 or bh<3:continue
                lead=s if s.startswith('V') or s in ('I','II','III') else 'a'+s[1:].upper()
                if lead not in EXPECTED:continue
                r,c=EXPECTED[lead]
                # Reject labels in header / rhythm strip, and implausible cells.
                if y<.055*h or y>.81*h:continue
                if abs((x+bw/2)/w-(c+.07)/4)>.16:continue
                if abs((y+bh/2)/h-(.19+.205*r))>.19:continue
                results.append({'lead':lead,'x':x,'y':y,'w':bw,'h':bh,'confidence':round(conf,1),'row':r,'col':c})
        except (subprocess.SubprocessError,OSError):continue
    # deduplicate label based on OCR confidence and expected position
    out={}
    for v in results:
        old=out.get(v['lead'])
        if old is None or old['confidence']<v['confidence']:out[v['lead']]=v
    return list(out.values())


def _grid_region(im):
    """Detect broad grid field vertically; never treat this as physical scale."""
    a=np.asarray(im.convert('RGB')).astype(np.int16);h,w=a.shape[:2]
    red=a[:,:,0]-(a[:,:,1]+a[:,:,2])//2
    mask=(red>11)&(a[:,:,0]>a[:,:,1]+6)&(a[:,:,0]>95)
    row=np.mean(mask,axis=1)
    q=np.where(row>max(.008,float(np.quantile(row,.70))*.33))[0]
    if len(q)>h*.35:
        y0=int(np.quantile(q,.02));y1=int(np.quantile(q,.98))
        if y1-y0>.48*h:return (max(0,y0),min(h,y1+1)),True
    return (0,h),False


def _activity_profile(im, xa, xb):
    rgb=np.asarray(im.convert('RGB'))
    h,w=rgb.shape[:2];xa=max(0,xa);xb=min(w,xb)
    if xb-xa<20:return np.zeros(h,float)
    crop=rgb[:,xa:xb]
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY).astype(float)
    # horizontal structure tendency. Dark text/grid are only weak hints.
    blur=cv2.GaussianBlur(gray,(0,0),2.0)
    ink=np.maximum(0,blur-gray)
    prof=np.percentile(ink,85,axis=1)
    return cv2.GaussianBlur(prof[:,None].astype('float32'),(1,31),0).ravel()


def _safe_edges(edges,n,min_gap):
    e=[int(round(v)) for v in edges]
    if len(e)!=n+1 or e[0]<0 or e[-1]>100000 or any(e[i+1]-e[i]<min_gap for i in range(n)):
        raise ValueError('Ranh giới cột/hàng không hợp lệ hoặc chồng lên nhau.')
    return e


def _calibration_column_hypotheses(im):
    """Weak boundary candidates from tall dark strokes near expected cell ends.

    This is explicitly a heuristic, never an assured calibration detector. The
    output is used only if evidence is present for at least two cell boundaries.
    """
    a=np.asarray(im.convert('RGB'));h,w=a.shape[:2]
    gray=cv2.cvtColor(a,cv2.COLOR_RGB2GRAY)
    bg=cv2.GaussianBlur(gray,(0,0),max(4.0,h*.01))
    ink=np.maximum(0,bg.astype('float32')-gray.astype('float32'))
    # Suppress weak grid while preserving dark rectangular pulse strokes.
    cut=max(6.0,float(np.percentile(ink,84)))
    mask=(ink>cut).astype('uint8')
    vk=max(18,int(.035*h))
    vertical=cv2.morphologyEx(mask,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(2,vk)))
    profile=vertical[int(.09*h):int(.76*h)].sum(0).astype(float)
    candidates=[]
    for col in range(1,4):
        expected=col*w/4
        low=max(0,int((col/4-.082)*w)); high=min(w,int((col/4+.082)*w))
        if high-low<10:continue
        sub=profile[low:high];j=int(np.argmax(sub))+low;peak=float(profile[j])
        # Compare peak against local clutter and demand a genuinely dark tall stroke.
        local=float(np.median(sub));quality=peak/max(1.,local+1)
        if peak>=max(30.,.045*h) and quality>=1.5:
            candidates.append((col,j,round(peak,1),round(quality,2)))
    return candidates


def detect_layout(im:Image.Image,*,ocr=True):
    """Infer 3x4 layout with explicit provenance and conservative confidence.

    Key property: column boundaries depend on detected lead-label starts when credible;
    otherwise one known limitation is reported (not claimed as detected).
    """
    im=_image(im);w,h=im.size
    if w<500 or h<250:raise ValueError('Image too small to propose 3x4 ECG layout')
    labels=_read_labels(im,ocr)
    observed={c:[] for c in range(4)}
    for x in labels:observed[x['col']].append(x['x'])
    # Robust column anchors: all labels printed near beginning of respective column.
    anchors={c:float(np.median(v)) for c,v in observed.items() if v}
    pitch=w/4
    candidate=[]
    for c in range(4):
        if c in anchors:
            candidate.append(anchors[c])
        else:
            refs=[(anchors[k]+(c-k)*pitch) for k in anchors]
            candidate.append(float(np.median(refs)) if refs else c*pitch+.025*w)
    # In inconsistent OCR positions, avoid corrupting plausible shape; geometry fallback.
    credible=(len(anchors)>=2 and all(.55*pitch<=candidate[i+1]-candidate[i]<=1.48*pitch for i in range(3)))
    if credible:
        bounds=[0]+[int(round(candidate[c]-.025*pitch)) for c in range(1,4)]+[w]
        bounds=[max(0,min(w,b)) for b in bounds]
        if any(bounds[i+1]-bounds[i]<.14*w for i in range(4)):credible=False
    pulse_hypotheses=[]
    pulse_based=False
    if not credible:
        bounds=[int(round(w*c/4)) for c in range(5)]
        pulse_hypotheses=_calibration_column_hypotheses(im)
        pulse_map={c:x for c,x,score,quality in pulse_hypotheses}
        if len(pulse_map)>=2:
            projected=[]
            for c in range(1,4):
                if c in pulse_map:projected.append(float(pulse_map[c]))
                else:projected.append(float(np.median([x+(c-k)*w/4 for k,x in pulse_map.items()])))
            trial=[0]+[int(round(v)) for v in projected]+[w]
            if all(.13*w <= trial[i+1]-trial[i] <= .37*w for i in range(4)):
                bounds=trial; pulse_based=True
    # Vertical layout: use lead-label row anchors only when 2+ independent rows.
    ys={r:[v['y']+v['h']*.5 for v in labels if v['row']==r] for r in range(3)}
    row_y={r:float(np.median(q)) for r,q in ys.items() if q}
    vertical=(len(row_y)>=2 and all(row_y.get(r+1,1e6)>row_y.get(r,-1e6) for r in range(2)))
    # Label printed at/above waveform row: offset proportional to row spacing.
    if vertical:
        step=np.median(np.diff(sorted(row_y.values())))
        est=[row_y.get(r, float(np.median([row_y[k]+(r-k)*step for k in row_y]))) for r in range(3)]
        if not (.10*h<est[0]<.37*h and .40*h<est[2]<.76*h):vertical=False
    if not vertical:
        # Estimate print-record top using visible grid; robust to gray/no red grid fallback.
        (gy0,gy1), grid_seen=_grid_region(im)
        if grid_seen and gy0>.06*h and gy0<.2*h:
            est=[gy0+.12*h,gy0+.32*h,gy0+.52*h]
        else:est=[.22*h,.43*h,.64*h]
    # Region band edges are halfway between row lines; allow deep/large QRS by
    # allocating almost whole row, not a narrow centerline strip.
    centers=[int(round(y)) for y in est]
    step1=max(30,centers[1]-centers[0]);step2=max(30,centers[2]-centers[1])
    y0=max(0,int(centers[0]-.49*step1))
    y1=int((centers[0]+centers[1])/2)
    y2=int((centers[1]+centers[2])/2)
    y3=min(h,int(centers[2]+.47*step2))
    # Rhythm region is distinct from three diagnostic rows.
    ry0=max(y3+2,int(.76*h)); ry1=min(h,int(.975*h))
    if ry1-ry0<.075*h:ry0=max(y3+2,int(.78*h));ry1=h
    xmode='ocr_anchored' if credible else ('calibration_stroke_candidate' if pulse_based else 'geometry_fallback')
    ymode='ocr_anchored' if vertical else 'image_geometry_fallback'
    reasons=[]
    if not credible:reasons.append('column_labels_insufficient_or_inconsistent')
    if pulse_based:reasons.append('calibration_stroke_hypothesis_requires_review')
    if not vertical:reasons.append('row_labels_insufficient_or_inconsistent')
    if ry0>=ry1:reasons.append('rhythm_region_unresolved')
    return {'version':VERSION,'size':[w,h], 'columns':bounds,'rows':[y0,y1,y2,y3],
            'rhythm_box':[0,ry0,w,ry1] if ry0<ry1 else None,
            'labels':labels,'calibration_stroke_hypotheses':pulse_hypotheses,'confidence':('REVIEW' if reasons else 'CANDIDATE'),
            'x_method':xmode,'y_method':ymode,'reasons':reasons,
            'note':'Candidate ROI only. Confirm source lead labels and deep QRS visually.'}


def resolve_layout(im, layout, *, columns=None, rows=None, per_lead=None, rhythm=None):
    w,h=im.size
    xs=_safe_edges(columns or layout['columns'],4,max(12,int(.10*w)))
    ys=_safe_edges(rows or layout['rows'],3,max(15,int(.09*h)))
    if ys[-1]>h:raise ValueError('Row bounds outside image')
    manual=per_lead or {}
    out=[]
    for row in range(3):
        for col in range(4):
            label=LAYOUT[row][col]; box=manual.get(label,(xs[col],ys[row],xs[col+1],ys[row+1]))
            box=tuple(int(q) for q in box)
            x0,y0,x1,y1=box
            if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Bad '+label+' box')
            out.append({'lead':label,'row':row,'col':col,'box':list(box),
                        'source':'manual' if label in manual else 'shared_grid',
                        'status':'MANUAL' if label in manual else 'REVIEW'})
    rb=rhythm if rhythm is not None else layout.get('rhythm_box')
    if rb and 0<=rb[0]<rb[2]<=w and 0<=rb[1]<rb[3]<=h:
        out.append({'lead':'II-rhythm','row':3,'col':0,'box':list(map(int,rb)),
                    'source':'shared_grid','status':'REVIEW'})
    return out


def render_overlay(im, rois, show_labels=True, scale_max=1700):
    out=im.convert('RGB').copy();w,h=out.size;d=ImageDraw.Draw(out)
    colors=['#19a4df','#18a86b','#d88d19']
    for v in rois:
        x0,y0,x1,y1=v['box'];color='#b33bd4' if v['lead']=='II-rhythm' else colors[v['row']]
        thick=max(2,w//700);d.rectangle([x0,y0,x1-1,y1-1],outline=color,width=thick)
        if show_labels:
            l=v['lead'];pw=max(30,10*len(l));ph=max(17,w//120)
            d.rectangle([x0,y0,min(w-1,x0+pw),min(h-1,y0+ph)],fill=color)
            d.text((x0+3,y0+1),l,fill='white')
    if max(out.size)>scale_max:out.thumbnail((scale_max,scale_max),Image.Resampling.LANCZOS)
    return out
