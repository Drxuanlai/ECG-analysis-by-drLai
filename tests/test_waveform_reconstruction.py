import numpy as np
from PIL import Image,ImageDraw
from core.waveform_reconstruction import reconstruct, exportable, overlay, panel


def synthetic(kind='regular'):
    w,h=420,140
    im=Image.new('RGB',(w,h),(215,215,212));d=ImageDraw.Draw(im)
    for x in range(0,w,9):d.line((x,0,x,h),fill=(195,179,177))
    for y in range(0,h,9):d.line((0,y,w,y),fill=(195,179,177))
    if kind!='blank':
        points=[]
        for x in range(w):
            phase=x%105
            y=75+4*np.sin(x/18)-43*np.exp(-((phase-46)/2.3)**2)+16*np.exp(-((phase-50)/2.5)**2)
            points.append((x,int(y)))
        if kind=='gaps':
            points=[p for p in points if not 155<=p[0]<188]
            for a,b in zip(points[:-1],points[1:]):
                if b[0]-a[0]==1:d.line((a,b),fill=(30,31,31),width=2)
        else:d.line(points,fill=(30,31,31),width=2)
    return im


def test_schema_and_no_calibration():
    r=reconstruct(synthetic())
    assert r['status']=='REVIEW' and len(r['x_src'])==len(r['y_src'])
    assert 0<=r['metrics']['coverage']<=1
    assert 'no_calibrated_ms_or_mV' in r['reason_codes']
    assert '_render' not in exportable(r)


def test_gaps_not_interpolated():
    r=reconstruct(synthetic('gaps'),min_evidence=.46)
    assert len(r['y_src'])>0
    assert any(y is None for y in r['y_src'])


def test_no_waveform_claim_on_blank():
    r=reconstruct(Image.new('RGB',(360,140),'white'))
    assert r['metrics']['coverage']==0
    assert 'no_supported_waveform' in r['reason_codes']


def test_render_sizes():
    im=synthetic();r=reconstruct(im)
    assert overlay(im,r).size==im.size
    p=panel(im,r)
    assert p.width==2*r['work_size'][0]
