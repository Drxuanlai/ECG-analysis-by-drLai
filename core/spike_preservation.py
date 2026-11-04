"""Evidence-based vertical stroke / QRS-spike candidate QA for ECG Core v1.4.1.

These are NOT certified R peaks or QRS detections. Regions are modified only
where observed near-black ink supports the proposed point. No physical units.
"""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image, ImageDraw
from core.waveform_reconstruction import reconstruct, _prep, exportable

VERSION='ecg-core-spike-preservation-v1.4.1'

def _runs(indices):
    if len(indices)==0:return []
    k=np.flatnonzero(np.diff(indices)>1)+1
    return np.split(indices,k)

def _candidate_strokes(image, base):
    im,rgb,dark,redpen,normalized,orig=_prep(image)
    h,w=dark.shape
    # Strong neutral/dark strokes, not colored grid. A local contrast rule is
    # necessary on dark phone photographs, not a global grayscale threshold.
    ink=(dark>=max(9.,float(np.percentile(dark,83)))) & (redpen<.55)
    ink=cv2.morphologyEx(ink.astype(np.uint8),cv2.MORPH_CLOSE,
                          np.ones((3,1),np.uint8)).astype(bool)
    # Long vertical segments across more than ~10% of this lead ROI.
    min_h=max(9,int(h*.115)); max_h=int(h*.72)
    rows=[]
    for x in range(5,w-5):
        segs=[a for a in _runs(np.flatnonzero(ink[:,x])) if min_h<=len(a)<=max_h]
        if not segs:continue
        seg=max(segs,key=len); top=int(seg[0]);bottom=int(seg[-1]);height=len(seg)
        if top<3 or bottom>h-4:continue
        contrast=float(np.median(dark[seg,x]));
        if contrast<10:continue
        rows.append((x,top,bottom,height,contrast))
    # Merge horizontally adjacent observations of the same stroke.
    groups=[]
    for item in rows:
        if groups and item[0]-groups[-1][-1][0]<=3 and abs(item[1]-groups[-1][-1][1])<=max(7,h//8):
            groups[-1].append(item)
        else:groups.append([item])
    out=[]
    baseline=np.asarray([np.nan if y is None else y for y in base['y_work']],float)
    for grp in groups:
        # Prefer a slim tall high-contrast stroke; broad upright text/calibration
        # must not be silently promoted to an ECG fiducial.
        mid=max(grp,key=lambda q:q[3]+.015*q[4]);x,top,bottom,height,_=mid
        if len(grp)>max(16,w//30):continue
        left=baseline[max(0,x-24):max(0,x-7)];right=baseline[min(w,x+8):min(w,x+25)]
        neighbours=np.concatenate([left[np.isfinite(left)],right[np.isfinite(right)]])
        if len(neighbours)<6:continue
        base_y=float(np.median(neighbours))
        # Require a connection from a local baseline to the stroke. It is not
        # enough for a line of paper grid to be vertical and dark.
        if not(top-3<=base_y<=bottom+3):continue
        upward=(base_y-top)>(bottom-base_y)
        apex=top if upward else bottom
        excursion=abs(base_y-apex)
        if excursion<max(9,h*.12):continue
        x0=min(t[0] for t in grp);x1=max(t[0] for t in grp)
        # Reject calibration/text when the stroke is too wide or isolated from
        # nearby usable waveform observations.
        if x1-x0>max(13,w//65):continue
        # Most dark pixels in vertical interval must form a connected shaft.
        evidence=float(np.mean(ink[top:bottom+1,x]))
        if evidence<.72:continue
        out.append({'x_work':int(x),'x_src':round((x+.5)*orig[0]/w,2),
                    'y_apex_work':int(apex),'y_apex_src':round(apex*orig[1]/h,2),
                    'baseline_work':round(base_y,2),'direction':'UP' if upward else 'DOWN',
                    'vertical_run_px':int(height),'excursion_px':round(excursion,1),
                    'evidence_fraction':round(evidence,3),'status':'CANDIDATE_REVIEW',
                    'reason':'vertical_ink_evidence_not_proof_of_QRS'})
    # Nonmaximum suppression of nearby stroke candidates.
    out.sort(key=lambda a:a['excursion_px'],reverse=True)
    selected=[]
    for v in out:
        if all(abs(v['x_work']-u['x_work'])>max(11,w//80) for u in selected):
            selected.append(v)
    return sorted(selected,key=lambda a:a['x_work']),ink

def _refine(image,base,spikes,ink):
    h,w=ink.shape
    y=list(base['y_work']); confidence=list(base['confidence']); supported=list(base['supported'])
    origin=[('BASE' if s else 'UNRESOLVED') for s in supported]
    # Set at most one apex point per detected stroke. No forced diagonal join,
    # no filling of the steep flanks: those remain evidence-limited.
    for event in spikes:
        x=event['x_work'];a=event['y_apex_work']
        close=np.flatnonzero(ink[max(0,a-2):min(h,a+3),max(0,x-2):min(w,x+3)])
        if len(close)==0:continue
        # Pin to observed ink at the actual candidate apex.
        if ink[a,x]:
            y[x]=float(a);supported[x]=True;origin[x]='SPIKE_APEX_CANDIDATE'
            confidence[x]=min(float(confidence[x]),.70) if base['supported'][x] else .50
    ow,oh=base['image_size']
    return {'version':VERSION,'status':'REVIEW','reason_codes':
            ['spike_candidates_not_qrs_validated','not_ground_truth_verified','no_calibrated_ms_or_mV'],
            'image_size':base['image_size'],'work_size':base['work_size'],
            'x_src':base['x_src'],
            'y_src':[None if v is None else float(v*oh/h) for v in y],
            'y_work':y,'supported':supported,'confidence':confidence,'point_origin':origin,
            'spike_candidates':spikes,
            'metrics':{'spike_candidates':len(spikes),'apex_points_with_ink_support':origin.count('SPIKE_APEX_CANDIDATE'),
                       'baseline_coverage':base['metrics']['coverage']},
            '_render':{'ink':ink}}

def reconstruct_spike_aware(image:Image.Image,**kwargs):
    base=reconstruct(image,**kwargs)
    spikes,ink=_candidate_strokes(image,base)
    return {'base':base,'refined':_refine(image,base,spikes,ink)}

def comparison_panel(original:Image.Image,comparison):
    work=original.convert('RGB')
    w,h=comparison['base']['work_size']
    im=work.resize((w,h),Image.Resampling.LANCZOS)
    out=Image.new('RGB',(w*2,h*2),'white')
    out.paste(im,(0,0))
    mask=comparison['refined']['_render']['ink']
    out.paste(Image.fromarray(mask.astype(np.uint8)*255).convert('RGB'),(w,0))
    for row,key in enumerate(['base','refined']):
        r=comparison[key];o=im.copy();d=ImageDraw.Draw(o)
        prev=None
        for i,y in enumerate(r['y_work']):
            if y is None:prev=None;continue
            pt=(i,int(round(y)))
            # No line across unresolved gaps or candidate discontinuities.
            if prev is not None and i-prev[0]==1 and abs(pt[1]-prev[1])<max(8,h//10):
                d.line((prev[0],prev[1],pt[0],pt[1]),fill=(0,210,215) if key=='base' else (255,130,20),width=2)
            else:d.ellipse((pt[0]-1,pt[1]-1,pt[0]+1,pt[1]+1),fill=(255,130,20))
            prev=pt
        if key=='refined':
            for e in r['spike_candidates']:
                x,y=e['x_work'],e['y_apex_work']
                d.ellipse((x-4,y-4,x+4,y+4),outline=(225,15,70),width=2)
        out.paste(o,((row%2)*w,h))
    # Lower-right shows the proposed apex while retaining base in other columns.
    o=im.copy();d=ImageDraw.Draw(o);r=comparison['refined'];prev=None
    for i,y in enumerate(r['y_work']):
        if y is None:prev=None;continue
        pt=(i,int(round(y)))
        if prev is not None and i-prev[0]==1 and abs(pt[1]-prev[1])<max(8,h//10):
            d.line((*prev, *pt),fill=(255,135,15),width=2)
        else:d.ellipse((pt[0]-1,pt[1]-1,pt[0]+1,pt[1]+1),fill=(255,135,15))
        prev=pt
    for e in r['spike_candidates']:
        x,y=e['x_work'],e['y_apex_work'];d.ellipse((x-4,y-4,x+4,y+4),outline=(220,20,70),width=2)
    out.paste(o,(w,h))
    return out

def export_comparison(comparison):
    return {'version':VERSION,'base':exportable(comparison['base']),
            'refined':{k:v for k,v in comparison['refined'].items() if k!='_render'},
            'disclaimer':'RESEARCH QA ONLY; NOT R PEAKS; NOT ECG MEASUREMENTS'}
