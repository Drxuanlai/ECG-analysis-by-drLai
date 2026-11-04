from PIL import Image, ImageDraw
from core.beat_candidates import detect_candidates, public_report, candidate_overlay


def _synthetic(xs, wide=False, invert=False):
    im = Image.new('RGB', (960, 150), 'white')
    d = ImageDraw.Draw(im)
    yy = 55
    d.line((0, yy, 959, yy), fill=(30, 30, 30), width=2)
    for x in xs:
        points = ([(x-13,yy),(x-5,yy-9),(x,yy+59 if invert else yy-48),(x+9,yy+12),(x+18,yy)]
                  if not wide else [(x-29,yy),(x-13,yy-10),(x,yy+52 if invert else yy-45),(x+16,yy+15),(x+29,yy)])
        d.line(points, fill=(25,25,25), width=3)
    return im


def test_blank_has_no_events():
    im = Image.new('RGB',(960,150),'white')
    out = detect_candidates(im)
    assert out['status']=='REVIEW'
    assert out['candidates']==[]


def test_irregular_spacing_is_never_periodically_filtered():
    im = _synthetic([100,237,521,644,851])
    out = detect_candidates(im,sensitivity=.7)
    assert out['metrics']['no_rr_regularization'] is True
    assert out['metrics']['clinical_qrs_verified']==0
    assert all(c['status']=='CANDIDATE_REVIEW' for c in out['candidates'])
    # No assertions of clinical recall: synthetic vertical ink is only a signal.


def test_negative_and_wide_are_supported_inputs():
    for wide,invert in [(False,True),(True,False),(True,True)]:
        out=detect_candidates(_synthetic([170,425,790],wide=wide,invert=invert))
        assert out['status']=='REVIEW'
        assert all(0<=c['x_pixel']<=960 for c in out['candidates'])


def test_export_omits_debug_arrays():
    im=_synthetic([200,440,700]);out=detect_candidates(im)
    assert '_debug' not in public_report(out)
    assert candidate_overlay(im,out).size==im.size


def test_tiny_roi_is_rejected():
    try:
        detect_candidates(Image.new('RGB',(25,15),'white'))
    except ValueError:
        return
    raise AssertionError('Tiny ROI should fail')
