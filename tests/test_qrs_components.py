import io
import json
import zipfile
from PIL import Image, ImageDraw
from core.qrs_components import (detect_qrs_components, compare_components_to_references,
                                 component_overlay)


def _synthetic_rs():
    im=Image.new('RGB',(260,110),'white')
    d=ImageDraw.Draw(im)
    d.line([(0,52),(50,52),(55,47),(60,52),(65,52),(71,95),
            (76,52),(160,52),(165,47),(170,52),(175,52),
            (181,95),(186,52),(259,52)],fill='black',width=2)
    return im


def test_empty_image_review():
    r=detect_qrs_components(Image.new('RGB',(200,100),'white'))
    assert r['status']=='REVIEW'
    assert r['candidate_count']==0


def test_no_references_used_during_detection():
    im=_synthetic_rs()
    a=detect_qrs_components(im)
    b=detect_qrs_components(im,hints=[{'x_pixel':100,'y_pixel':7}])
    assert a==b


def test_component_schema():
    r=detect_qrs_components(_synthetic_rs())
    assert r['version'].endswith('qrs-components')
    for c in r['components']:
        assert c['qc']=='REVIEW'
        assert c['S_nadir']['type']=='S_NADIR_CANDIDATE'
        assert c['morphology_proposal'] in ('rS_POSSIBLE_REVIEW','NEGATIVE_QRS_REVIEW')


def test_reference_type_separation():
    report={'components':[{'component_id':'C001','S_nadir':{'x_pixel':30,'y_pixel':81},'r_apex':{'x_pixel':25,'y_pixel':40}}]}
    refs=[{'reference_id':'R1','x_pixel':30,'y_pixel':81,'fiducial_type':'S_NADIR'},
          {'reference_id':'R2','x_pixel':25,'y_pixel':40,'fiducial_type':'R_PEAK'},
          {'reference_id':'R3','x_pixel':30,'y_pixel':81,'fiducial_type':'QS_NADIR'}]
    r=compare_components_to_references(report,refs)
    assert len(r['matches'])==2
    assert all(x['status']=='XY_WITHIN_TOLERANCE' for x in r['matches'])


def test_overlay_size():
    im=_synthetic_rs()
    assert component_overlay(im,detect_qrs_components(im)).size==im.size


def test_actual_v1_roi_smoke():
    import os
    zpath='/mnt/data/ecg_core_v1_5_6_fiducial_2d_validation.zip'
    if not os.path.exists(zpath):return
    with zipfile.ZipFile(zpath) as z:
        roi=Image.open(io.BytesIO(z.read('roi_original_v15.png'))).convert('RGB')
        refs=json.loads(z.read('rpeak_reference_v152.json'))['reference_points']
    r=detect_qrs_components(roi)
    assert r['status']=='REVIEW' and r['candidate_count']>0
    assert len(compare_components_to_references(r,refs)['matches'])==3
