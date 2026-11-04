import numpy as np
from PIL import Image,ImageDraw
from core.qrs_segment_reconstruction import _anchored_path,reconstruct_segments,exportable


def test_trace_anchored_on_real_pixels():
    prob=np.zeros((40,60),np.float32)
    coords=[(x,20-max(0,13-abs(x-30))) for x in range(17,44)]
    for x,y in coords:prob[y,x]=1.
    ink=prob>.5
    p1=_anchored_path(prob,ink,17,20,30,7)
    p2=_anchored_path(prob,ink,30,7,43,20)
    assert p1 is not None and p2 is not None
    assert p1[0]==20 and p1[-1]==7 and p2[0]==7 and p2[-1]==20


def test_empty_image_never_claims_valid_qrs():
    im=Image.new('RGB',(320,110),'white')
    res=reconstruct_segments(im)
    assert res['segment']['status']=='REVIEW'
    assert res['segment']['metrics']['segment_candidates']==0
    assert all(s['status']!='PASS' for s in res['segment']['segments'])


def test_export_is_json_safe_and_no_render():
    import json
    im=Image.new('RGB',(320,110),'white');r=reconstruct_segments(im)
    out=exportable(r)
    json.dumps(out,allow_nan=False)
    assert '_render' not in out['segment']


def test_input_size_and_nulls():
    im=Image.new('RGB',(500,120),'white');draw=ImageDraw.Draw(im)
    draw.line([(20,65),(90,65),(93,20),(97,65),(180,65)],fill='black',width=2)
    r=reconstruct_segments(im)
    assert len(r['segment']['y_src'])==r['base']['work_size'][0]
    assert any(v is None for v in r['segment']['y_src'])
