from PIL import Image,ImageDraw
import pytest
from core.lead_layout import detect_layout,resolve_layout,render_overlay,LAYOUT

def test_layout_fallback_12_leads():
    im=Image.new('RGB',(1600,800),'white')
    layout=detect_layout(im,ocr=False)
    assert layout['confidence']=='REVIEW'
    roi=resolve_layout(im,layout)
    assert len([x for x in roi if x['lead']!='II-rhythm'])==12
    assert roi[0]['lead']=='I' and roi[-2]['lead']=='V6'

def test_reject_overlapping_columns():
    im=Image.new('RGB',(1600,800),'white');lay=detect_layout(im,ocr=False)
    with pytest.raises(ValueError): resolve_layout(im,lay,columns=[0,100,110,120,1600])

def test_manual_roi_applies_only_selected_lead():
    im=Image.new('RGB',(1600,800),'white');lay=detect_layout(im,ocr=False)
    roi=resolve_layout(im,lay,per_lead={'V3':[900,400,1100,600]})
    v3=next(x for x in roi if x['lead']=='V3');assert v3['status']=='MANUAL'
    assert v3['box']==[900,400,1100,600]
    assert next(x for x in roi if x['lead']=='V2')['status']=='REVIEW'

def test_overlay_output():
    im=Image.new('RGB',(1600,800),'white');lay=detect_layout(im,ocr=False)
    assert render_overlay(im,resolve_layout(im,lay)).size==(1600,800)
