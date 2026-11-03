"""Grid-aware ECG image evidence for research QA only.

No ECG waveform is reconstructed, no clinical measurement made. Grid maps are
observational hypotheses; do not erase neutral ink indiscriminately.
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from core.adaptive_segmentation import extract_ink, _rgb

VERSION = 'grid-aware-1.2.0'


def _runs(positions, length):
    out=np.zeros(length, np.uint8)
    if not len(positions): return out
    groups=np.split(positions,np.where(np.diff(positions)>1)[0]+1)
    for g in groups:
        at=int(g[np.argmax(g*0+1)]) if len(g)==1 else int(np.median(g))
        out[max(0,at-1):min(length,at+2)]=1
    return out


def grid_evidence(rgb):
    """Detect red/pink grid by chromaticity; grey grid remains uncertain.

    Returns conservative grid-colored pixel evidence, plus row/column traces.
    This is not an assertion that every grid intersection can be removed.
    """
    f=rgb.astype(np.float32)
    r,g,b=f[:,:,0],f[:,:,1],f[:,:,2]
    red=r-.5*(g+b)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    chrom=np.maximum(red,0)
    chrom_gate=(chrom>max(4.,float(np.percentile(chrom,65))+.5))&(r>g+3)&(r>b+3)&(gray>55)
    h,w=gray.shape
    row=np.mean(chrom_gate,axis=1)
    col=np.mean(chrom_gate,axis=0)
    # Only mark periodic looking line candidates. Reject wholly colored backgrounds.
    def select(profile, dim):
        size=max(3,int(dim*.008));size+=1-size%2
        trend=cv2.GaussianBlur(profile.astype(np.float32).reshape(-1,1),(1,max(3,size|1)),0).ravel()
        residual=profile-trend
        high=max(.055,float(np.percentile(residual,78)))
        positions=np.flatnonzero((residual>high)&(profile>.055))
        return _runs(positions,dim)
    rows=select(row,h); cols=select(col,w)
    # Colored pixel map is independent of line hypothesis; no deletion at intersections.
    grid_colored=chrom_gate.astype(np.uint8)*255
    long_lines=((rows[:,None]>0)|(cols[None,:]>0))&chrom_gate
    return {'color_map':grid_colored,'line_map':long_lines.astype(np.uint8)*255,
            'rows':int(np.count_nonzero(rows)),'columns':int(np.count_nonzero(cols)),
            'colored_fraction':float(chrom_gate.mean())}


def segment_grid_aware(image):
    rgb=_rgb(image);h,w=rgb.shape[:2]
    if min(h,w)<24: raise ValueError('Lead crop too small')
    older=extract_ink(image)
    enhanced=np.asarray(older['enhanced'])
    # Dark neutral printing retains contrast on red and grey paper.
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY).astype(np.float32)
    f=rgb.astype(np.float32)
    neutral_spread=np.max(f,axis=2)-np.min(f,axis=2)
    red=f[:,:,0]-(f[:,:,1]+f[:,:,2])*.5
    # Multiple spatial backgrounds: small waveform details and slow image lighting.
    background=cv2.GaussianBlur(gray,(0,0),sigmaX=5.0)
    contrast=np.maximum(0,background-gray)
    bg_long=cv2.GaussianBlur(gray,(0,0),sigmaX=max(12,min(h,w)/6))
    darkness=np.maximum(0,bg_long-gray)
    high=max(5.,float(np.percentile(contrast,83)))
    local=(contrast>=high)&(darkness>=max(4.,float(np.percentile(darkness,60))))
    # Very dark ink is retained even when chromatic grid traverses the same pixel.
    hard_dark=(gray<np.percentile(gray,12))&(darkness>=7)
    chromatic_red=(red>6)&(neutral_spread>10)&(gray>75)
    ink=(local|hard_dark)&(~chromatic_red | (darkness>max(17.,float(np.percentile(darkness,90)))))
    ink=np.uint8(ink)*255
    # Connected components filter removes isolated specks, not narrow QRS strokes.
    n,ids,stats,_=cv2.connectedComponentsWithStats(ink,8)
    clean=np.zeros_like(ink)
    for k in range(1,n):
        x,y,cw,ch,area=[int(v) for v in stats[k]]
        if area>=4 or cw>=6 or ch>=6:clean[ids==k]=255
    ev=grid_evidence(rgb)
    selected=clean>0
    grid=ev['line_map']>0
    overlap=float((selected&grid).sum()/max(1,selected.sum()))
    # Identify line-shaped residuals as QA only. ECG baseline may be horizontal.
    horizontal=cv2.morphologyEx(clean,cv2.MORPH_OPEN,np.ones((1,max(20,w//6)),np.uint8))>0
    vertical=cv2.morphologyEx(clean,cv2.MORPH_OPEN,np.ones((max(18,h//3),1),np.uint8))>0
    straight_fraction=float(((horizontal|vertical)&selected).sum()/max(1,selected.sum()))
    density=float(selected.mean()); coverage=float(np.any(selected,axis=0).mean())
    # Score independent of 'truth': a QC review aid, never auto-PASS.
    review=[]
    if density<.002:review.append('almost_empty')
    if density>.13:review.append('dense_mask')
    if coverage<.35:review.append('low_column_coverage')
    if straight_fraction>.32:review.append('straight_residue_or_isoelectric')
    if ev['colored_fraction']>.25:review.append('heavy_chromatic_grid')
    # Disagreement of candidate methods is evidence of uncertainty, not a reason to choose one blindly.
    old=np.asarray(older['ink_mask'])>0
    union=(old|selected).sum(); iou=float((old&selected).sum()/union) if union else 1.
    if iou<.35:review.append('method_disagreement')
    return {'version':VERSION,'original':Image.fromarray(rgb),'enhanced':older['enhanced'],
        'ink_mask':Image.fromarray(clean),'grid_map':Image.fromarray(ev['color_map']),
        'grid_lines':Image.fromarray(ev['line_map']),
        'legacy_mask':older['ink_mask'],'legacy_method':older['method'],
        'status':'REVIEW','review_reasons':review or ['requires_visual_reference'],
        'metrics':{'density':round(density,5),'column_coverage':round(coverage,4),
           'straight_residue_fraction':round(straight_fraction,4),'grid_overlap_fraction':round(overlap,4),
           'mask_iou_with_v11':round(iou,4),'colored_grid_fraction':round(ev['colored_fraction'],4),
           'grid_detected_rows':ev['rows'],'grid_detected_columns':ev['columns']},
        'size':[w,h], 'note':'Grid-aware QA only. Mask does NOT prove ECG identity or numeric accuracy.'}


def diagnostic_sheet(result):
    frames=[]
    rgb=np.asarray(result['original']);h,w=rgb.shape[:2]
    for key in ('original','enhanced','grid_map','ink_mask','legacy_mask'):
        x=np.asarray(result[key]); x=np.repeat(x[:,:,None],3,axis=2) if x.ndim==2 else x
        frames.append(Image.fromarray(x.astype(np.uint8)).convert('RGB'))
    overlay=rgb.copy(); mask=np.asarray(result['ink_mask'])>0
    overlay[mask]=np.clip(.55*overlay[mask]+.45*np.array([0,210,115]),0,255).astype('uint8')
    frames.append(Image.fromarray(overlay))
    out=Image.new('RGB',(3*w,2*h),'white')
    for i,img in enumerate(frames):out.paste(img,((i%3)*w,(i//3)*h))
    return out
