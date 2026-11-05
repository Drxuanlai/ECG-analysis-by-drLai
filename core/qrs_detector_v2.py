"""Independent conservative, polarity-agnostic QRS candidate baseline v2.

Uses ONLY the ROI image; never reads human labels, historic predictions, RR
clock or waveform v1. Detected events remain REVIEW, not clinical QRS.
"""
from __future__ import annotations
import cv2
import numpy as np

def detect_qrs_candidates(image, *, min_distance_px:int=24, max_candidates:int=120)->dict:
    from PIL import Image
    if isinstance(image,Image.Image): rgb=np.asarray(image.convert('RGB'))
    else:
        rgb=np.asarray(image)
        if rgb.ndim==2:rgb=np.stack([rgb]*3,axis=-1)
    h,w=rgb.shape[:2]
    if w<60 or h<32:return {'status':'UNRESOLVED','candidates':[],'reasons':['ROI_TOO_SMALL']}
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    # Contrast-normalized dark evidence with horizontal background estimate.
    gray=cv2.GaussianBlur(gray,(3,3),0)
    bg=cv2.medianBlur(gray,21)
    ink=cv2.subtract(bg,gray).astype(np.float32)
    # Differential energy captures steep positive OR negative ECG deflections.
    # Suppress purely horizontal rules using vertical derivative; downweight
    # columns with ink running across almost the full ROI height.
    dy=np.abs(np.diff(ink,axis=0,prepend=ink[:1]))
    col_density=(ink>11).mean(axis=0)
    row_lo=int(h*0.10);row_hi=max(row_lo+1,int(h*0.90))
    vertical=cv2.Sobel(ink,cv2.CV_32F,1,0,ksize=3)
    energy=np.quantile(np.abs(vertical[row_lo:row_hi]),0.88,axis=0)
    # Component candidate for narrow spiky QRS; signal energy alone does not
    # claim R peak or S nadir. For dense grids, prefer unresolved.
    width=max(5,(w//150)|1)
    kernel=np.ones(width,np.float32)/width
    smooth=np.convolve(energy,kernel,mode='same')
    base=np.median(smooth)
    mad=np.median(np.abs(smooth-base)) + 1e-3
    threshold=max(float(base+2.8*mad),float(np.quantile(smooth,0.70)))
    from scipy.signal import find_peaks
    peaks,props=find_peaks(smooth,height=threshold,distance=max(8,int(min_distance_px)),prominence=max(1,1.8*mad))
    order=sorted(peaks,key=lambda x:float(smooth[x]),reverse=True)[:max_candidates]
    order.sort()
    candidates=[]
    for x in order:
        if x<4 or x>w-5:continue
        # dark component's local extreme is unknown; report X only.
        dense=bool(col_density[x]>.50)
        candidates.append({'id':f'B{len(candidates)+1:03d}','x':float(x),'y':None,
                           'type':'QRS_EVENT','strength':round(float(smooth[x]),3),
                           'status':'UNRESOLVED' if dense else 'CANDIDATE_REVIEW',
                           'reasons':['COLUMN_DENSE_INK'] if dense else ['NOT_VERIFIED']})
    return {'schema':'ecg-qrs-detector-v2.0','status':'REVIEW_ONLY','candidates':candidates,
            'reasons':['INDEPENDENT_IMAGE_ONLY_BASELINE','UNVERIFIED_IMAGE_ARTIFACTS'],
            'parameters':{'min_distance_px':min_distance_px,'max_candidates':max_candidates},
            'diagnostic':{'width':w,'height':h,'threshold':round(threshold,3)}}
