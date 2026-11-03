"""Multi-layout ECG proposals. QA-only; never infer clinical measurements.

Explicit layout selection is authoritative when OCR is insufficient. 3x4 and
6x2 patterns map every physical row/column to a standard lead name. Rhythm
strip is optional and is NEVER guessed from a 6x2 print.
"""
from __future__ import annotations
import io,re,csv,shutil,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageOps

VERSION='ecg-core-multilayout-v1.3.3'
LAYOUTS={
 '3x4':(('I','aVR','V1','V4'),('II','aVL','V2','V5'),('III','aVF','V3','V6')),
 '6x2':(('I','V1'),('II','V2'),('III','V3'),('aVR','V4'),('aVL','V5'),('aVF','V6')),
 '12x1':tuple((s,) for s in ('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6')),
}
LEADS={v for rows in LAYOUTS.values() for row in rows for v in row}

def _ocr_labels(im:Image.Image,enabled=True):
    if not enabled or not shutil.which('tesseract'):return []
    a=np.asarray(im.convert('L'));h,w=a.shape
    # OCR only for lead label clues; never create coordinates from guesses.
    import cv2
    enhanced=cv2.createCLAHE(clipLimit=2.,tileGridSize=(8,8)).apply(a)
    b=io.BytesIO();Image.fromarray(enhanced).save(b,format='PNG')
    try:
        p=subprocess.run(['tesseract','stdin','stdout','--psm','11','-l','eng','tsv'],input=b.getvalue(),capture_output=True,check=True,timeout=12)
    except (OSError,subprocess.SubprocessError):return []
    out=[]
    for q in csv.DictReader(io.StringIO(p.stdout.decode('utf8',errors='replace')),delimiter='\t'):
        s=re.sub(r'[^A-Za-z0-9]','',q.get('text','')).upper()
        if s not in ('I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6'):continue
        try:x,y,bw,bh=int(q['left']),int(q['top']),int(q['width']),int(q['height']);conf=float(q['conf'])
        except (ValueError,TypeError,KeyError):continue
        if conf<28 or bw<2 or bh<3 or not (.07*h<y<.94*h):continue
        lead=s if not s.startswith('AV') else 'a'+s[1:]
        out.append({'lead':lead,'x':x,'y':y,'w':bw,'h':bh,'confidence':round(conf,1)})
    return out

def suggest_layout(im:Image.Image,ocr=True):
    """Suggest only when spatially consistent anchors justify one geometry."""
    w,h=im.size;obs=_ocr_labels(im,ocr); scores={}
    for name,pattern in LAYOUTS.items():
        rows=len(pattern);cols=len(pattern[0]);votes={};matches=[]
        for q in obs:
            for r,row in enumerate(pattern):
                if q['lead'] not in row:continue
                c=row.index(q['lead']); xn=(q['x']+q['w']/2)/w;yn=(q['y']+q['h']/2)/h
                # Lead text usually begins near left of its column and above waveform.
                expected_x=(c+.06)/cols
                expected_y=(.17 + r*(.64/max(1,rows-1))) if rows<=6 else .10+r*.067
                dx=abs(xn-expected_x);dy=abs(yn-expected_y)
                xtol=.19/cols+.025;ytol=.68/rows+.022
                if dx<=xtol and dy<=ytol:
                    qual=max(0.,1-dx/xtol)*max(0.,1-dy/ytol)
                    matches.append({'lead':q['lead'],'row':r,'col':c,'x':q['x'],'y':q['y'],'confidence':q['confidence'],'fit':round(qual,3)})
                    votes[q['lead']]=max(votes.get(q['lead'],0),qual*(.65+.35*q['confidence']/100))
        scores[name]={'score':round(sum(votes.values()),3),'anchors':matches,'unique_leads':len(votes)}
    ranked=sorted(scores,key=lambda k:scores[k]['score'],reverse=True)
    best=ranked[0];runner=ranked[1]
    # Two independent spatially credible lead labels plus clear margin required.
    strong=scores[best]['unique_leads']>=3 and scores[best]['score']>=1.15 and scores[best]['score']-scores[runner]['score']>=.55
    return {'version':VERSION,'suggested':best if strong else 'UNKNOWN',
            'status':'REVIEW','scores':scores,'labels':obs,
            'reason':None if strong else 'insufficient_discriminating_lead_labels',
            'note':'Always visually confirm layout. Unknown must not trigger 3x4 extraction.'}

def propose_boxes(im:Image.Image,layout_type:str,*,has_rhythm:bool|None=None,
                  column_boundaries=None,row_boundaries=None,per_lead=None):
    if layout_type not in LAYOUTS:raise ValueError('Unsupported/unknown layout: explicit selection required')
    w,h=im.size;pattern=LAYOUTS[layout_type];nr=len(pattern);nc=len(pattern[0])
    if has_rhythm is None:has_rhythm=(layout_type=='3x4')
    if has_rhythm and layout_type!='3x4':raise ValueError('Rhythm strip is only supported as an explicit 3x4 option in v1.3.3')
    # Default bounds are tentative, not an automatic recognition of waveform extent.
    xa=[round(w*i/nc) for i in range(nc+1)]
    if layout_type=='3x4':
        ya=[round(h*v) for v in ((.12,.33,.54,.75) if has_rhythm else (.13,.39,.65,.91))]
    elif layout_type=='6x2':ya=[round(h*v) for v in (.155,.278,.400,.525,.650,.775,.900)]
    else:ya=[round(h*(.065+i*.074)) for i in range(13)]
    xs=list(map(int,column_boundaries or xa));ys=list(map(int,row_boundaries or ya))
    if len(xs)!=nc+1 or len(ys)!=nr+1 or xs[0]<0 or xs[-1]>w or ys[0]<0 or ys[-1]>h:
        raise ValueError('Boundary list mismatch')
    if any(xs[i+1]-xs[i]<max(15,w//(nc*7)) for i in range(nc)) or any(ys[i+1]-ys[i]<max(6,h//(nr*5)) for i in range(nr)):
        raise ValueError('ROI boundaries overlap or are too narrow')
    overrides=per_lead or {};rois=[]
    for r,row in enumerate(pattern):
        for c,lead in enumerate(row):
            box=list(map(int,overrides.get(lead,[xs[c],ys[r],xs[c+1],ys[r+1]])))
            a,b,d,e=box
            if not(0<=a<d<=w and 0<=b<e<=h):raise ValueError('Invalid box for '+lead)
            rois.append({'lead':lead,'row':r,'col':c,'layout_box':box,'waveform_box':box,
                         'box':box,'source':'manual' if lead in overrides else 'layout_hypothesis','status':'REVIEW'})
    if has_rhythm:
        ry0=max(ys[-1]+2,round(.79*h));ry1=min(h,round(.97*h))
        if ry1>ry0+8:
            box=[0,ry0,w,ry1]
            rois.append({'lead':'II-rhythm','row':nr,'col':0,'box':box,'layout_box':box,
                         'waveform_box':box,'status':'REVIEW','source':'rhythm_user_confirmed'})
    return {'version':VERSION,'layout_type':layout_type,'rhythm_present':bool(has_rhythm),
            'column_boundaries':xs,'row_boundaries':ys,'rois':rois,
            'status':'REVIEW','note':'All ROI coordinates are hypotheses; visual confirmation required.'}

def overlay(im,proposal,max_width=1900):
    out=im.convert('RGB').copy();d=ImageDraw.Draw(out);w,h=out.size
    palette=['#129cc8','#26a87a','#e29a19','#c665d2','#6d8bd9','#cf5e68']
    for q in proposal['rois']:
        x0,y0,x1,y1=q['box'];color='#b235bf' if q['lead']=='II-rhythm' else palette[q['row']%len(palette)]
        d.rectangle((x0,y0,x1-1,y1-1),outline=color,width=max(2,w//850))
        pw=max(34,9*len(q['lead']));d.rectangle((x0,y0,min(x1-1,x0+pw),min(y1-1,y0+19)),fill=color)
        d.text((x0+2,y0+2),q['lead'],fill='white')
    if out.width>max_width:out.thumbnail((max_width,max_width),Image.Resampling.LANCZOS)
    return out
