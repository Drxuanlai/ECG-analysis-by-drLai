import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import io
from PIL import Image
import pytest
from core.visual_reader_v21 import (layout_boxes,choose_leads,sanitized_observations,
    extract_json,reconciliation,image_for_model,image_from_bytes,report_export,LEADS)

def test_layout_3x4():
    d=layout_boxes((3200,1200),'3x4',rhythm=True)
    assert len(d)==13 and d['I'][0]<d['aVR'][0]<d['V1'][0]<d['V4'][0]
    assert 'II-rhythm' in d

def test_layout_6x2():
    d=layout_boxes((3200,1200),'6x2')
    assert len(d)==12 and d['I'][1]<d['II'][1]<d['III'][1]
    assert d['V1'][0]>d['I'][0] and 'II-rhythm' not in d

def test_wrong_layout_rejected():
    with pytest.raises(ValueError):layout_boxes((900,600),'4x3')

def test_custom_edges_rejected():
    with pytest.raises(ValueError):layout_boxes((900,600),'6x2',edges_x=[0,.9,.2])

def test_all_12_unknown_not_normal():
    d=sanitized_observations({'lead_observations':[{'lead':'II','qrs':'R cao'}]})
    assert len(d['lead_observations'])==12
    assert d['lead_observations']['V3']['visibility']=='UNKNOWN'
    assert d['lead_observations']['II']['qrs']=='R cao'

def test_hypotheses_limited():
    d=sanitized_observations({'hypotheses':[{'type':'AF_SUSPECT','leads':['II','BAD'],'evidence':'uneven RR'}]})
    assert d['hypotheses'][0]['leads']==['II']

def test_prioritize_mandatory_even_no_issue():
    picks=choose_leads({'hypotheses':[]},set(LEADS)|{'II-rhythm'})
    assert len(picks)==6 and picks[:4]==['II-rhythm','II','V1','V2']

def test_json_fence():
    assert extract_json('```json\n{"lead_observations":[]}\n```')['lead_observations']==[]

def test_cross_recheck_not_verified():
    d=reconciliation({'lead_observations':{'V1':{'observation':'rS'}}},
                     {'lead_observations':{'V1':{'observation':'rS'}}},['V1'])
    assert d[0]['verification_status']=='AI_SECOND_VIEW_ONLY_NOT_INDEPENDENT_MEASUREMENT'

def test_no_fake_quant_verification():
    result=report_export({}, {}, [], {}, '3x4', [], 'hash')
    assert result['quantitative_verification']['status']=='NOT_AVAILABLE'

def test_image_pipeline():
    i=Image.new('RGB',(800,200),'white'); raw=image_for_model(i)
    result=image_from_bytes(raw)
    assert result.size==(800,200)
