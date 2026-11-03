"""Separate candidate ECG ink from confident colored grid and non-waveform suspects.

No single mask proves cardiac activity. In particular, grid crossings remain
uncertain, not forcibly erased. OCR/text and calibration are only proposals.
"""
from __future__ import annotations
import cv2,numpy as np
from PIL import Image
VERSION='signal-artifact-evidence-v1.3.3'

def separate(im:Image.Image, *, exclude_margins=True):
    rgb=np.asarray(im.convert('RGB')); h,w=rgb.shape[:2]
    if min(h,w)<24:raise ValueError('ROI too small')
    f=rgb.astype('float32');gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype('float32')
    blur=cv2.GaussianBlur(gray,(0,0),4.0);slow=cv2.GaussianBlur(gray,(0,0),max(9,min(h,w)/7))
    local=np.maximum(0,blur-gray);dark=np.maximum(0,slow-gray)
    residual=np.maximum(local,.65*dark)
    med=float(np.median(residual));mad=float(np.median(np.abs(residual-med)))
    cut=max(5.,med+3*max(1.,mad))
    candidate=(residual>cut)
    red=f[:,:,0]-(f[:,:,1]+f[:,:,2])*.5
    chrom=(red>max(6.,float(np.percentile(red,80))+1))&(f[:,:,0]>f[:,:,1]+5)&(gray>62)
    # Only colored lines that extend for many columns/rows are confident grid candidates.
    row_mean=chrom.mean(axis=1);col_mean=chrom.mean(axis=0)
    row_grid=(row_mean>max(.19,float(np.quantile(row_mean,.85))))
    col_grid=(col_mean>max(.19,float(np.quantile(col_mean,.85))))
    structural=chrom&(row_grid[:,None]|col_grid[None,:])
    # Chromatic non-ink is reasonable, but dark trace crossing colored lines is uncertain.
    confident_grid=structural & (local<max(11.,cut*1.5))
    grid_collision=structural&candidate&~confident_grid
    # Non-waveform candidates, including tall calibration at edge or printed labels.
    # We do NOT assert these are definitely text/calibration.
    dark_binary=(residual>cut).astype('uint8')
    n,labels,stats,_=cv2.connectedComponentsWithStats(dark_binary,8)
    suspect=np.zeros((h,w),bool)
    for idx in range(1,n):
        x,y,cw,ch,area=map(int,stats[idx])
        near_edge=(x<.10*w or x+cw>.90*w)
        tall=ch>.27*h and cw<.12*w
        dense=(area>=30 and cw>.03*w and ch>.07*h and (x<.17*w or y<.16*h))
        if (near_edge and tall) or dense:suspect[labels==idx]=True
    if exclude_margins:
        suspect[:max(1,int(.035*h)),:]=True
        suspect[max(0,int(.965*h)):,:]=True
    ink_candidate=candidate & ~confident_grid & ~suspect
    uncertain=grid_collision | (candidate&suspect)
    # What is excluded from quantitative use is not removed from the review evidence.
    masks={'ecg_ink_candidate':ink_candidate,'confident_colored_grid':confident_grid,
           'non_waveform_suspect':suspect&candidate,'uncertain_intersection':uncertain,
           'raw_dark_candidate':candidate}
    counts={k:int(v.sum()) for k,v in masks.items()}
    return {'version':VERSION,'status':'REVIEW','masks':{k:v.astype('uint8')*255 for k,v in masks.items()},
            'metrics':{'pixel_counts':counts,'ecg_candidate_fraction':round(float(ink_candidate.mean()),5),
                       'grid_fraction':round(float(confident_grid.mean()),5),
                       'uncertain_fraction':round(float(uncertain.mean()),5),
                       'raw_candidate_fraction':round(float(candidate.mean()),5)},
            'reasons':['not_waveform_verified','gray_grid_may_remain','calibration_text_are_hypotheses'],
            'note':'A candidate mask is not a traced ECG; ambiguous ink must not be discarded without review.'}

def sheet(im,result):
    rgb=np.asarray(im.convert('RGB'));h,w=rgb.shape[:2];out=Image.new('RGB',(w*3,h*2),'white')
    tiles=[im,Image.fromarray(result['masks']['raw_dark_candidate']),
           Image.fromarray(result['masks']['confident_colored_grid']),
           Image.fromarray(result['masks']['non_waveform_suspect']),
           Image.fromarray(result['masks']['uncertain_intersection'])]
    overlay=rgb.copy();m=result['masks']['ecg_ink_candidate']>0
    overlay[m]=(.40*overlay[m]+.60*np.array([20,220,140])).astype('uint8')
    tiles.append(Image.fromarray(overlay))
    for k,t in enumerate(tiles):out.paste(t.convert('RGB'),((k%3)*w,(k//3)*h))
    return out
