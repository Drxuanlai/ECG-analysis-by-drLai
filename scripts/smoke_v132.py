import json
from pathlib import Path
from PIL import Image,ImageOps
from core.lead_layout import detect_layout,resolve_layout
from core.waveform_roi import refine_waveform_rois
from core.lead_isolation import isolate_leads,isolation_overlay

cases={'CASE_001':'/mnt/data/IMG_1386.png','CASE_002':'/mnt/data/ecg.png','CASE_003':'/mnt/data/IMG_1394.jpg'}
outdir=Path('/mnt/data/v132_smoke');outdir.mkdir(exist_ok=True)
report={}
for name,src in cases.items():
    im=ImageOps.exif_transpose(Image.open(src)).convert('RGB')
    if max(im.size)>2800:im.thumbnail((2800,2800))
    layout=detect_layout(im,ocr=False)
    rois=refine_waveform_rois(im,resolve_layout(im,layout))
    result=isolate_leads(im,rois)
    overlay=isolation_overlay(im,result,rois)
    overlay.save(outdir/(name+'_isolation.png'))
    report[name]={'leads':len(result['diagnostics']), 'owned':sum(d['owned_ink_px'] for d in result['diagnostics'].values()),'ambiguous':int((result['ambiguous_mask']>0).sum()),'statuses':sorted(set(d['status'] for d in result['diagnostics'].values()))}
    print(name,report[name],flush=True)
(outdir/'smoke_report.json').write_text(json.dumps(report,indent=2))
