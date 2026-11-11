from PIL import Image,ImageDraw
from core.beat_anchor_v24 import (windows_for_anchors,map_local_landmarks,
 evidence_checks,compare_to_human,image_r_candidates,beat_overlay)

def test_windows_irregular():
 w=windows_for_anchors([{'x_pixel':100},{'x_pixel':270},{'x_pixel':323}],500,80)
 assert len(w)==3 and w[0]['x1']<=w[1]['x0'] and w[1]['x1']<=w[2]['x0']

def test_invalid_anchors():
 assert not windows_for_anchors([{'x_pixel':-1},{'x_pixel':float('nan')}],150)

def test_local_offset_and_bounds():
 roi=Image.new('RGB',(300,90),'white');win={'beat_id':'B1','x0':80,'x1':180,'anchor_x':120}
 got=map_local_landmarks({'landmarks':[{'landmark_type':'R_PEAK','x_pixel':45,'y_pixel':10},
 {'landmark_type':'R_PEAK','x_pixel':120,'y_pixel':10}]},win,roi)
 assert len(got)==1 and got[0]['x_pixel']==125

def test_check_proposal_not_validated():
 roi=Image.new('RGB',(300,90),'white');win={'beat_id':'B1','x0':80,'x1':180,'anchor_x':120}
 p={'beat_id':'B1','landmark_type':'R_PEAK','x_pixel':125,'y_pixel':10}
 assert evidence_checks(roi,[p],[win])[0]['qa_state']=='LOCALIZATION_REVIEW'

def test_2d_one_to_one():
 h=[{'id':'M1','source':'HUMAN_CONFIRMED','landmark_type':'R_PEAK','x_pixel':100,'y_pixel':15}]
 p=[{'beat_id':'B1','landmark_type':'R_PEAK','x_pixel':103,'y_pixel':50},
 {'beat_id':'B1','landmark_type':'R_PEAK','x_pixel':102,'y_pixel':15}]
 result=compare_to_human(p,h)
 assert result[0]['status']=='X_ONLY_Y_MISMATCH' and result[1]['status']=='NO_X_REFERENCE_MATCH'

def test_synthetic_spike_candidates():
 img=Image.new('RGB',(370,110),'white');d=ImageDraw.Draw(img)
 for x in (60,155,280):d.line((x,22,x,85),fill='black',width=3)
 c=image_r_candidates(img,min_distance=35)
 assert len(c)>=3

def test_overlay_dimensions():
 im=Image.new('RGB',(350,80),'white')
 w={'beat_id':'B1','x0':40,'x1':190,'anchor_x':100}
 assert beat_overlay(im,w,[],[]).size==(150,80)
