import io,json,zipfile
import numpy as np
import pytest
from PIL import Image
from core.benchmark_v2 import import_validation_zip,match_one_to_one,evaluate
from core.qrs_detector_v2 import detect_qrs_candidates

def sample_zip(complete=False):
    f=io.BytesIO();buf=io.BytesIO();Image.new('RGB',(100,50),'white').save(buf,format='PNG')
    ref={'image_name':'x.png','image_sha256_12':'abc','lead':'II','layout':'6x2','roi_box':[0,0,100,50],
      'reference_points':[{'reference_id':'R1','x_pixel':25,'y_pixel':15}], 'reference_complete':complete,
      'annotation_windows':{'windows':[{'reviewed':False}]}}
    with zipfile.ZipFile(f,'w') as z:
        z.writestr('roi_original_v15.png',buf.getvalue());z.writestr('rpeak_reference_v153.json',json.dumps(ref))
    return f.getvalue()

def test_import_does_not_upgrade_unreviewed():
    obj=import_validation_zip(sample_zip(True))['record']
    assert not obj['reference_complete'] and obj['review_status']=='UNVERIFIED'

def test_match_prevents_duplicate_tp():
    refs=[{'x':100,'y':10}]
    preds=[{'x':99,'y':10},{'x':101,'y':10}]
    m,fn,fp=match_one_to_one(refs,preds,5)
    assert len(m)==1 and len(fn)==0 and len(fp)==1

def test_incomplete_never_scores():
    rec=import_validation_zip(sample_zip(True))['record']
    r=evaluate(rec,[{'x':25}],10)
    assert r['status']=='REVIEW' and r['sensitivity'] is None

def test_verified_test_can_score():
    rec=import_validation_zip(sample_zip())['record'];rec.update(all_windows_reviewed=True,reference_complete=True,review_status='VERIFIED',dataset_split='TEST',development_exposed=False)
    rep=evaluate(rec,[{'x':25},{'x':80}]);assert rep['status']=='SCORED' and rep['tp']==1 and rep['fp']==1

def test_2d_not_equal_x_only():
    a=[{'x':20,'y':10}];b=[{'x':20,'y':45}]
    assert len(match_one_to_one(a,b,5,None)[0])==1
    assert len(match_one_to_one(a,b,5,5)[0])==0

def test_blank_not_claimed_pass():
    res=detect_qrs_candidates(Image.new('RGB',(300,100),'white'))
    assert res['status']=='REVIEW_ONLY' and not res['candidates']

def test_minimal_image_unresolved():
    assert detect_qrs_candidates(Image.new('RGB',(40,20),'white'))['status']=='UNRESOLVED'


def test_development_case_cannot_score_even_if_test_selected():
    rec=import_validation_zip(sample_zip())['record']
    rec.update(all_windows_reviewed=True,reference_complete=True,review_status='VERIFIED',dataset_split='TEST')
    assert evaluate(rec,[{'x':25}])['status']=='REVIEW'
