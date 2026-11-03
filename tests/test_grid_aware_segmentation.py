import numpy as np
from PIL import Image,ImageDraw
from core.grid_aware_segmentation import segment_grid_aware,grid_evidence


def _synthetic(bg,grid):
    im=Image.new('RGB',(480,180),bg);d=ImageDraw.Draw(im)
    for x in range(0,480,11):d.line((x,0,x,179),fill=grid,width=1)
    for y in range(0,180,11):d.line((0,y,479,y),fill=grid,width=1)
    d.line([(0,95),(135,95),(147,32),(155,145),(170,94),(479,94)],fill=(25,25,27),width=3)
    return im

def test_red_grid_retains_ink():
    ev=segment_grid_aware(_synthetic((248,247,245),(232,174,182)))
    assert np.asarray(ev['ink_mask']).sum()>0
    assert ev['status']=='REVIEW'
    assert 'grid_overlap_fraction' in ev['metrics']

def test_dark_grey_page_retains_ink():
    ev=segment_grid_aware(_synthetic((112,111,110),(135,131,129)))
    assert np.asarray(ev['ink_mask']).sum()>0

def test_blank_never_claims_pass():
    ev=segment_grid_aware(Image.new('RGB',(320,100),'white'))
    assert ev['status']!='PASS'

def test_grid_map_shape():
    x=np.ones((80,240,3),dtype=np.uint8)*245
    assert grid_evidence(x)['color_map'].shape==(80,240)
