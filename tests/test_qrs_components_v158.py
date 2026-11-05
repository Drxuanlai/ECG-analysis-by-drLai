"""Safety regressions for v1.5.8 image-only QRS component proposals."""
import io
import os
import zipfile
from PIL import Image, ImageDraw
from core.qrs_components import detect_qrs_components, component_overlay


def _image_with_rs_and_calibration():
    im = Image.new('RGB', (420, 160), 'white')
    d = ImageDraw.Draw(im)
    # Local baseline shifted around first beat, with a small positive r.
    d.line([(0,73),(65,73),(70,69),(75,73),(80,73),
            (85,132),(94,73),(190,73),
            (195,68),(200,73),(206,73),(211,135),(221,73),(300,73)],
           fill='black',width=3)
    # Rectangular calibration pulse: plateau -> long vertical -> top shelf.
    d.line([(310,73),(330,73),(330,108),(346,108),(346,25),
            (385,25),(385,108),(405,108),(405,73),(419,73)],
           fill='black',width=3)
    return im


def test_calibration_pulse_not_promoted_to_qrs():
    r=detect_qrs_components(_image_with_rs_and_calibration(),min_depth_px=15)
    assert r['status']=='REVIEW'
    assert all(c['S_nadir']['x_pixel'] < 310 for c in r['components'])
    assert len(r['excluded_candidates']) >= 1


def test_overlay_and_schema_have_exclusion():
    im=_image_with_rs_and_calibration()
    r=detect_qrs_components(im,min_depth_px=15)
    assert 'excluded_candidates' in r
    assert component_overlay(im,r).size == im.size


def test_unrelated_blank_image_has_no_qrs():
    r=detect_qrs_components(Image.new('RGB',(420,160),'white'))
    assert not r['components']
    assert not r['excluded_candidates']


def test_original_roi_regression_when_fixture_available():
    path='/mnt/data/ecg_core_v1_5_7_qrs_components_validation.zip'
    if not os.path.exists(path):
        return
    with zipfile.ZipFile(path) as z:
        im=Image.open(io.BytesIO(z.read('roi_original_v15.png'))).convert('RGB')
    result=detect_qrs_components(im)
    assert len(result['components'])==3
    assert len(result['excluded_candidates'])==1
    assert sorted(round(c['S_nadir']['x_pixel']) for c in result['components'])==[151,335,510]
    assert all(c['r_apex'] is not None for c in result['components'])
    assert abs(result['components'][0]['r_apex']['x_pixel']-147)<3
