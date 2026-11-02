"""Experimental, image-only ECG ink extraction; never a source of clinical measurements.

Produces independently inspectable candidates. Does not synthesize a 1-D waveform.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image

VERSION = 'adaptive-segmentation-1.0'


def _odd(n):
    n = max(3, int(n))
    return n if n % 2 else n+1


def _rgb(image):
    if isinstance(image, Image.Image): return np.asarray(image.convert('RGB'))
    a=np.asarray(image, dtype=np.uint8)
    if a.ndim==2: return np.repeat(a[:,:,None],3,axis=2)
    return a[:,:,:3]


def _cleanup(mask):
    # Remove only isolated speckles. No closing, which could connect grid to ECG.
    n, ids, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
    output=np.zeros(mask.shape,np.uint8)
    for i in range(1,n):
        x,y,w,h,area=stats[i]
        if area>=3 or w>=5 or h>=5: output[ids==i]=255
    return output


def extract_ink(image):
    """Return images, metrics and candidate masks; does not claim waveform identity."""
    rgb=_rgb(image)
    h,w=rgb.shape[:2]
    if min(h,w)<24: raise ValueError('Lead crop too small')
    lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    l=lab[:,:,0]
    # Large-scale illumination correction; robust to dim gray photographs.
    scale=min(151,_odd(max(25,min(h,w)//2)))
    bg=cv2.GaussianBlur(l,(scale,scale),0).astype(np.float32)
    norm=np.clip(l.astype(np.float32)-bg+205,0,255).astype(np.uint8)
    norm=cv2.createCLAHE(clipLimit=1.8,tileGridSize=(8,8)).apply(norm)
    # Relative darkness is used in addition to an absolute threshold: this avoids
    # the all-zero mask encountered on dark photos with global color cutoff.
    blur=cv2.GaussianBlur(norm,(0,0),sigmaX=2.0)
    local_diff=np.maximum(0,blur.astype(np.float32)-norm.astype(np.float32))
    blackhat=cv2.morphologyEx(norm,cv2.MORPH_BLACKHAT,np.ones((9,9),np.uint8))
    otsu, otsu_mask=cv2.threshold(norm,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    adaptive=cv2.adaptiveThreshold(norm,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV,_odd(max(15,min(51,min(h,w)//5))),5)
    # Red/pink grid is penalized but never blindly removed at every intersection.
    f=rgb.astype(np.float32)
    red=np.maximum(0,f[:,:,0]-(f[:,:,1]+f[:,:,2])/2)
    chroma=np.max(f,axis=2)-np.min(f,axis=2)
    colored_grid=(red>7)&(chroma>11)&(norm>75)
    lowgray=norm < min(205,max(95,int(np.percentile(norm,18))))
    candidates={
        'relative':((local_diff>=max(2.5,float(np.percentile(local_diff,76)))) & (norm<242)),
        'adaptive':(adaptive>0)&(norm<235),
        'blackhat':(blackhat>=max(4,float(np.percentile(blackhat,78))))&(norm<238),
        'otsu':(otsu_mask>0)&(norm<235),
    }
    # Keep truly dark signal even at grid intersections; penalize chromatic-only pixels.
    for k,m in candidates.items():
        m=(m & (~colored_grid | lowgray))
        candidates[k]=_cleanup(m)
    def metric(a):
        binary=a>0
        density=float(binary.mean())
        column=float(np.mean(np.any(binary,axis=0)))
        # Penalize dense background and blank masks; not a clinical accuracy score.
        bad= (density<.001 or density>.12 or column<.15)
        score= (2.0*min(column,.9)-2.2*abs(density-.035)
                -4.0*max(0,density-.12)-1.0*max(0,.008-density))
        return {'density':round(density,5),'column_coverage':round(column,4),
                'screening_score':round(float(score),4),'screening_review':bool(bad)}
    qm={k:metric(v) for k,v in candidates.items()}
    eligible=[k for k,v in qm.items() if not v['screening_review']]
    chosen=max(eligible or candidates, key=lambda k:qm[k]['screening_score'])
    mask=candidates[chosen]
    status='REVIEW' if not eligible or qm[chosen]['screening_review'] else 'CANDIDATE'
    # Explicitly not PASS: candidate-mask geometry alone does not validate true ECG.
    return {'version':VERSION,'enhanced':Image.fromarray(norm),
            'ink_mask':Image.fromarray(mask),'candidates':{k:Image.fromarray(v) for k,v in candidates.items()},
            'method':chosen,'status':status,'metrics':qm,'size':[w,h],
            'warning':'CANDIDATE is not validated ECG waveform; inspect original and mask.'}


def four_panel(image, result):
    rgb=_rgb(image); h,w=rgb.shape[:2]
    base=Image.fromarray(rgb)
    a=np.asarray(result['enhanced']); enh=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_GRAY2RGB))
    mask=np.asarray(result['ink_mask'])
    maskrgb=np.repeat(mask[:,:,None],3,axis=2)
    overlay=rgb.copy(); overlay[mask>0]=np.clip(.55*overlay[mask>0]+.45*np.array([0,220,120]),0,255).astype(np.uint8)
    out=Image.new('RGB',(w*2,h*2),'white')
    for im,xy in [(base,(0,0)),(enh,(w,0)),(Image.fromarray(maskrgb),(0,h)),(Image.fromarray(overlay),(w,h))]:out.paste(im,xy)
    return out
