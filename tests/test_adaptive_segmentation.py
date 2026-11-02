import numpy as np
from PIL import Image, ImageDraw
from core.adaptive_segmentation import extract_ink


def test_dim_gray_photo_must_not_yield_empty_mask():
    im=Image.new('RGB',(480,180),(110,108,107))
    dr=ImageDraw.Draw(im)
    for x in range(0,480,12):dr.line((x,0,x,179),fill=(126,113,110),width=1)
    for y in range(0,180,12):dr.line((0,y,479,y),fill=(126,113,110),width=1)
    dr.line([(0,95),(130,95),(145,93),(152,38),(159,140),(167,96),(480,96)],fill=(24,24,24),width=3)
    out=extract_ink(im)
    assert np.asarray(out['ink_mask']).sum()>0
    assert out['status'] in ('CANDIDATE','REVIEW')


def test_bright_pink_grid_and_ink_not_empty():
    im=Image.new('RGB',(480,180),'white');dr=ImageDraw.Draw(im)
    for x in range(0,480,10):dr.line((x,0,x,179),fill=(244,189,190))
    for y in range(0,180,10):dr.line((0,y,479,y),fill=(244,189,190))
    dr.line([(0,95),(140,95),(147,30),(154,144),(165,96),(479,95)],fill=(30,30,35),width=3)
    ev=extract_ink(im)
    assert np.asarray(ev['ink_mask']).sum()>0


def test_blank_image_never_pass():
    ev=extract_ink(Image.new('RGB',(240,80),'white'))
    assert ev['status']!='PASS'
