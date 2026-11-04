import numpy as np
from PIL import Image,ImageDraw
from core.spike_preservation import reconstruct_spike_aware,comparison_panel,export_comparison

def img(peak=True):
    im=Image.new('RGB',(480,140),(232,232,231));d=ImageDraw.Draw(im)
    for x in range(0,480,12):d.line((x,0,x,139),fill=(206,194,193))
    for y in range(0,140,12):d.line((0,y,479,y),fill=(206,194,193))
    pts=[]
    for x in range(480):
        phase=x%120
        y=80-56*np.exp(-((phase-54)/2.2)**2) if peak else 80
        pts.append((x,round(y)))
    d.line(pts,fill=(32,32,32),width=2)
    return im

def test_schema():
    c=reconstruct_spike_aware(img())
    assert c['refined']['status']=='REVIEW'
    assert len(c['refined']['y_src'])==len(c['base']['y_src'])
    assert 'base' in export_comparison(c)

def test_candidate_is_not_clinical_r_peak():
    c=reconstruct_spike_aware(img())
    assert all(e['status']=='CANDIDATE_REVIEW' for e in c['refined']['spike_candidates'])

def test_never_interpolates_unsupported_base_columns():
    c=reconstruct_spike_aware(Image.new('RGB',(360,140),'white'))
    assert all(y is None for y in c['refined']['y_src'])

def test_visual():
    im=img();p=comparison_panel(im,reconstruct_spike_aware(im))
    assert p.width>im.width and p.height>im.height
