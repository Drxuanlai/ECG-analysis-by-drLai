from PIL import Image
import pytest
from core.multi_layout import suggest_layout,propose_boxes,LAYOUTS,overlay

def test_unknown_does_not_force_3x4():
    im=Image.new('RGB',(1600,800),'white')
    assert suggest_layout(im,ocr=False)['suggested']=='UNKNOWN'
    with pytest.raises(ValueError):propose_boxes(im,'UNKNOWN')

def test_6x2_without_rhythm():
    im=Image.new('RGB',(1944,820),'white')
    d=propose_boxes(im,'6x2')
    assert len(d['rois'])==12
    assert not d['rhythm_present']
    assert [q['lead'] for q in d['rois']]==['I','V1','II','V2','III','V3','aVR','V4','aVL','V5','aVF','V6']

def test_3x4_with_explicit_rhythm():
    im=Image.new('RGB',(1600,800),'white')
    d=propose_boxes(im,'3x4',has_rhythm=True)
    assert len(d['rois'])==13 and d['rois'][-1]['lead']=='II-rhythm'

def test_no_implicit_rhythm_6x2():
    im=Image.new('RGB',(1600,800),'white')
    with pytest.raises(ValueError):propose_boxes(im,'6x2',has_rhythm=True)

def test_manual_override():
    im=Image.new('RGB',(1600,800),'white')
    d=propose_boxes(im,'6x2',per_lead={'V2':[900,240,1400,335]})
    assert next(q for q in d['rois'] if q['lead']=='V2')['box']==[900,240,1400,335]
    assert next(q for q in d['rois'] if q['lead']=='V3')['source']=='layout_hypothesis'

def test_multi_overlay():
    im=Image.new('RGB',(1600,800),'white')
    assert overlay(im,propose_boxes(im,'6x2')).size==(1600,800)
