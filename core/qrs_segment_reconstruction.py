"""ECG Core v1.4.2: conservative segment-level ink tracing QA.

Outputs candidate traces, NOT verified QRS/R peaks or clinical measurements.
Rejects unsubstantiated segments and retains nulls for unobserved ink.
"""
from __future__ import annotations
import numpy as np
import cv2
from PIL import Image, ImageDraw
from core.waveform_reconstruction import _prep, reconstruct
from core.spike_preservation import _candidate_strokes

VERSION='ecg-core-qrs-segment-v1.4.2'


def _stroke_evidence(image):
    im,rgb,dark,redpen,_,_= _prep(image)
    # A dark, neutral candidate pixel. Red grid is strongly penalized, never
    # assumed absent, and line morphology is only a warning.
    threshold=max(6.,float(np.percentile(dark,72)))
    prob=np.clip((dark-threshold+7.)/22.,0.,1.) * np.clip(1.-.9*redpen,0.,1.)
    prob=cv2.GaussianBlur(prob.astype(np.float32),(3,3),.55)
    return prob, (prob>=.27)


def _anchored_path(prob, ink, x0, y0, x1, y1, max_step=13):
    """Constrained shortest path between two observed points; gaps remain null.

    Cost discourages unsupported pixels. Never infer a track from continuity
    alone; each output point must individually have actual ink evidence.
    """
    h,w=prob.shape
    if x1<=x0 or not(0<=x0<w and 0<=x1<w):return None
    ys=np.arange(h)
    a=max(0,min(h-1,int(round(y0))));b=max(0,min(h-1,int(round(y1))))
    dp=np.full(h,np.inf,np.float32);dp[a]=0.
    parents=[]
    for x in range(x0+1,x1+1):
        prior=np.full(h,np.inf,np.float32);par=np.full(h,-1,np.int16)
        for d in range(-max_step,max_step+1):
            if d>=0: dest=slice(d,h);src=slice(0,h-d)
            else:dest=slice(0,h+d);src=slice(-d,h)
            penalty=.055*abs(d)+.002*d*d
            cand=dp[src]+penalty
            update=cand<prior[dest]
            prior[dest]=np.minimum(prior[dest],cand)
            pp=par[dest];pp[update]=ys[src][update];par[dest]=pp
        # A blank column is allowed for traversing, but cannot be exported.
        evidence=prob[:,x]
        nonink=np.where(ink[:,x],0.,3.2)
        dp=prior+2.5*(1.-evidence)+nonink
        parents.append(par)
    if not np.isfinite(dp[b]):return None
    path=np.empty(x1-x0+1,np.int32);path[-1]=b
    for i in range(len(parents)-1,-1,-1):
        prev=int(parents[i][path[i+1]])
        if prev<0:return None
        path[i]=prev
    if path[0]!=a:return None
    return path


def reconstruct_segments(image:Image.Image, *,min_evidence=.42, max_span_fraction=.18):
    base=reconstruct(image,min_evidence=min_evidence)
    spikes,shaft_ink=_candidate_strokes(image,base)
    prob,ink=_stroke_evidence(image)
    h,w=prob.shape
    output=list(base['y_work']); origin=['BASE' if v is not None else 'UNRESOLVED' for v in output]
    candidates=[];used=np.zeros(w,bool)
    margin=max(11,int(w*.008));width=max(12,min(38,int(w*.030)))
    baseline=np.array([np.nan if y is None else y for y in base['y_work']],float)
    for event in spikes:
        xc=int(event['x_work']);apex=int(event['y_apex_work'])
        item={'x_work':xc,'apex_y_work':apex,'direction':event['direction'],
              'status':'REJECTED','reasons':[],'candidate_window':[max(0,xc-width),min(w-1,xc+width)],
              'supported_points':0,'changed_points':0}
        if xc<margin+width or xc>=w-margin-width:
            item['reasons'].append('near_roi_edge');candidates.append(item);continue
        left=np.flatnonzero(np.isfinite(baseline[max(0,xc-width):xc-6]))+max(0,xc-width)
        right=np.flatnonzero(np.isfinite(baseline[xc+7:min(w,xc+width+1)]))+xc+7
        if len(left)<3 or len(right)<3:
            item['reasons'].append('missing_flank_anchors');candidates.append(item);continue
        xa=int(np.median(left[:min(len(left),5)]));xb=int(np.median(right[-min(len(right),5):]))
        ya=int(round(np.median(baseline[max(xa-2,0):xa+3][np.isfinite(baseline[max(xa-2,0):xa+3])]))) if np.isfinite(baseline[xa]) else -1
        yb=int(round(np.median(baseline[max(xb-2,0):xb+3][np.isfinite(baseline[max(xb-2,0):xb+3])]))) if np.isfinite(baseline[xb]) else -1
        if ya<0 or yb<0 or xb-xa>max(42,int(w*max_span_fraction)) or np.any(used[xa:xb+1]):
            item['reasons'].append('invalid_or_overlapping_window');candidates.append(item);continue
        # A field filled with dense ink/grid cannot support a unique QRS trace.
        # This screen intentionally rejects candidates in heavily textured
        # photograph folds and grid-dominated regions, even if a path exists.
        local_density=float(np.mean(ink[:,max(0,xc-20):min(w,xc+21)]))
        item['local_ink_density']=round(local_density,3)
        if local_density>.31:
            item['reasons'].append('dense_grid_or_artifact_background');candidates.append(item);continue
        if not shaft_ink[apex,xc] or prob[apex,xc]<.23:
            item['reasons'].append('apex_lacks_local_ink');candidates.append(item);continue
        # The apex is an explicit intermediate anchor: both flanks are traced.
        p1=_anchored_path(prob,ink,xa,ya,xc,apex)
        p2=_anchored_path(prob,ink,xc,apex,xb,yb)
        if p1 is None or p2 is None:
            item['reasons'].append('no_anchored_segment_path');candidates.append(item);continue
        path=np.r_[p1[:-1],p2]
        xx=np.arange(xa,xb+1)
        pixel_evidence=ink[path,xx] & (prob[path,xx]>=.30)
        supported=float(pixel_evidence.mean())
        item['supported_fraction']=round(supported,3)
        item['candidate_window']=[xa,xb]
        item['supported_points']=int(pixel_evidence.sum())
        # Reject if the reconstructed segment has no meaningful excursion
        # or the path only follows background/grid with weak ink support.
        baseline_y=float(np.median([ya,yb]));exc=abs(float(apex)-baseline_y)
        if supported<.70 or exc<max(8,h*.10):
            item['reasons'].append('insufficient_full_segment_evidence');candidates.append(item);continue
        # The segment is still only a candidate. Preserve null columns and
        # do NOT extend across missing-pixel evidence.
        changed=0
        for x,y,ok in zip(xx,path,pixel_evidence):
            if not ok:continue
            if output[x] is None or abs(float(output[x])-float(y))>.5:changed+=1
            output[x]=float(y);origin[x]='QRS_SEGMENT_CANDIDATE'
        item['changed_points']=changed;item['status']='SEGMENT_CANDIDATE_REVIEW'
        used[xa:xb+1]=True;candidates.append(item)
    oh=base['image_size'][1]
    result={'version':VERSION,'status':'REVIEW',
      'reason_codes':['candidate_segments_not_qrs_verified','not_ground_truth_verified','no_calibrated_ms_or_mV'],
      'image_size':base['image_size'],'work_size':base['work_size'],
      'x_src':base['x_src'],
      'y_work':output,
      'y_src':[None if v is None else round(v*oh/h,3) for v in output],
      'point_origin':origin,'segments':candidates,
      'metrics':{'stroke_candidates':len(spikes),
                 'segment_candidates':sum(s['status']=='SEGMENT_CANDIDATE_REVIEW' for s in candidates),
                 'rejected_segments':sum(s['status']=='REJECTED' for s in candidates),
                 'columns_changed':sum(a=='QRS_SEGMENT_CANDIDATE' for a in origin),
                 'candidate_ink_coverage':round(float(np.mean([v is not None for v in output])),4)},
      '_render':{'probability':prob,'ink':ink}}
    return {'base':base,'segment':result}


def compare_panel(image:Image.Image, comp):
    w,h=comp['base']['work_size'];original=image.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)
    canvas=Image.new('RGB',(2*w,2*h),'white')
    canvas.paste(original,(0,0))
    m=Image.fromarray(np.uint8(comp['segment']['_render']['ink'])*255).convert('RGB')
    canvas.paste(m,(w,0))
    for j,which in enumerate(['base','segment']):
        view=original.copy();dr=ImageDraw.Draw(view);last=None
        for x,y in enumerate(comp[which]['y_work']):
            if y is None:last=None;continue
            pt=(x,round(y))
            # Don't draw artificial connecting strokes across missing data
            # or implausibly large discontinuities.
            if last is not None and x-last[0]==1 and abs(pt[1]-last[1])<=max(8,h//12):
                col=(0,190,210) if j==0 else (255,140,25)
                dr.line((last[0],last[1],*pt),fill=col,width=2)
            last=pt
        if j==1:
            for s in comp['segment']['segments']:
                xx,yy=s['x_work'],s['apex_y_work'];col=(30,195,80) if s['status'].startswith('SEGMENT_') else (210,40,60)
                dr.ellipse((xx-4,yy-4,xx+4,yy+4),outline=col,width=2)
        canvas.paste(view,((j%2)*w,h))
    return canvas


def exportable(comp):
    return {'base_version':comp['base']['version'],
            'segment':{k:v for k,v in comp['segment'].items() if k!='_render'},
            'disclaimer':'RESEARCH_ONLY. Segments are not certified QRS or ECG measurements.'}
