from PIL import Image,ImageDraw
from core.signal_artifact import separate,sheet
import numpy as np

def test_blank_no_ink():
    img=Image.new('RGB',(300,180),'white');ev=separate(img)
    assert ev['status']=='REVIEW'
    assert ev['metrics']['pixel_counts']['raw_dark_candidate']==0

def test_red_grid_and_neutral_trace():
    im=Image.new('RGB',(300,180),'white');d=ImageDraw.Draw(im)
    for x in range(10,300,15):d.line((x,0,x,179),fill=(230,157,165),width=1)
    for y in range(8,180,15):d.line((0,y,299,y),fill=(230,157,165),width=1)
    d.line([(10,90),(75,90),(88,83),(91,109),(95,41),(99,93),(160,89),(190,72),(220,90),(290,90)],fill=(20,20,20),width=3)
    result=separate(im)
    assert result['metrics']['pixel_counts']['raw_dark_candidate']>0
    assert sheet(im,result).size==(900,360)

def test_gray_grid_uncertainty_not_called_confident_red():
    im=Image.new('RGB',(300,180),'white');d=ImageDraw.Draw(im)
    for x in range(0,300,12):d.line((x,0,x,179),fill=(185,185,185))
    ev=separate(im)
    assert ev['metrics']['pixel_counts']['confident_colored_grid']==0
