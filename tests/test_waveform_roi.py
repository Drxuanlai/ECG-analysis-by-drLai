import numpy as np
import pytest
from PIL import Image
from core.lead_layout import detect_layout,resolve_layout
from core.waveform_roi import refine_waveform_rois,render_dual_overlay


def fixture():
    im=Image.fromarray(np.full((700,1400,3),225,np.uint8))
    layout=detect_layout(im,ocr=False)
    rois=resolve_layout(im,layout)
    return im,rois


def test_preserve_layout_boxes():
    im,base=fixture()
    out=refine_waveform_rois(im,base)
    assert len(out)==13
    assert all(a['box']==b['layout_box'] for a,b in zip(base,out))
    assert all(b['waveform_box'][1]<=b['layout_box'][1] and b['waveform_box'][3]>=b['layout_box'][3] for b in out)


def test_no_horizontal_crossing():
    im,base=fixture()
    out=refine_waveform_rois(im,base)
    for q in out:
        assert q['layout_box'][0]==q['waveform_box'][0]
        assert q['layout_box'][2]==q['waveform_box'][2]


def test_manual_override_is_local():
    im,base=fixture()
    v=next(x for x in base if x['lead']=='V2')
    x0,y0,x1,y1=v['box']; new=[x0,max(0,y0-20),x1,min(im.height,y1+30)]
    out=refine_waveform_rois(im,base,per_lead={'V2':new})
    assert next(x for x in out if x['lead']=='V2')['waveform_box']==new
    assert next(x for x in out if x['lead']=='I')['waveform_box_source']=='edge_safe_auto'


def test_invalid_manual_cross_column_rejected():
    im,base=fixture()
    v=next(x for x in base if x['lead']=='V2')
    x0,y0,x1,y1=v['box']
    with pytest.raises(ValueError):
        refine_waveform_rois(im,base,per_lead={'V2':[x0-5,y0,x1,y1]})


def test_overlap_is_reported():
    im,base=fixture()
    out=refine_waveform_rois(im,base)
    assert any(v.get('waveform_overlaps') for v in out if v['lead']!='II-rhythm')
    assert render_dual_overlay(im,out).size==im.size
