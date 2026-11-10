import unittest
from PIL import Image
from core.grid_evidence_v23 import (GridCalibration,measure_points,overlay,sanitize_ai_landmarks,paper_values,validated_roi)

class TestGridEvidence(unittest.TestCase):
    def setUp(self):
        self.cal=GridCalibration(10,60,10,60,25.,10.)
    def test_calibration(self):
        self.assertEqual(self.cal.px_per_mm_x,10)
        self.assertAlmostEqual(self.cal.ms_per_px,4)
        self.assertAlmostEqual(self.cal.mv_per_px,.01)
    def test_boxes(self):
        self.assertAlmostEqual(paper_values(50,5)['small_box_ms'],20)
        self.assertAlmostEqual(paper_values(50,5)['small_box_mv'],.2)
    def test_reject_bad_grid(self):
        with self.assertRaises(ValueError):GridCalibration(10,10,0,50,25,10)
    def test_measurement(self):
        def p(i,kind,x,y,beat='B1'):
            return dict(id=str(i),beat_id=beat,landmark_type=kind,x_pixel=x,y_pixel=y,source='HUMAN_CONFIRMED')
        pts=[p(1,'QRS_ONSET',100,40),p(2,'QRS_OFFSET',135,40),p(3,'R_PEAK',115,50),
             p(4,'BASELINE',110,65),p(5,'J_POINT',138,56),p(6,'R_PEAK',280,50,'B2')]
        out=measure_points(pts,self.cal)
        got={x['metric']:x['value'] for x in out}
        self.assertEqual(got['QRS_DURATION'],140)
        self.assertEqual(got['R_AMPLITUDE'],.15)
        self.assertEqual(got['ST_AT_J'],.09)
        self.assertEqual(got['RR_INTERVAL'],660)
    def test_no_measure_without_human(self):
        pts=[dict(id='a',beat_id='B1',landmark_type='QRS_ONSET',x_pixel=1,y_pixel=5,source='AI_UNVERIFIED'),
             dict(id='b',beat_id='B1',landmark_type='QRS_OFFSET',x_pixel=25,y_pixel=5,source='AI_UNVERIFIED')]
        self.assertEqual(measure_points(pts,self.cal),[])
    def test_ai_malformed(self):
        raw={'landmarks':[{'landmark_type':'R_PEAK','x_pixel':3,'y_pixel':4,'beat_id':'B1'},
                          {'landmark_type':'R_PEAK','x_pixel':105,'y_pixel':4},
                          {'landmark_type':'FAKE','x_pixel':5,'y_pixel':3},
                          {'landmark_type':'J_POINT','x_pixel':None,'y_pixel':3}]}
        proposed=sanitize_ai_landmarks(raw,100,100)
        self.assertEqual(len(proposed),1)
        self.assertEqual(proposed[0]['source'],'AI_UNVERIFIED')
    def test_overlay(self):
        img=Image.new('RGB',(200,120),'white')
        out=overlay(img,self.cal,[dict(x_pixel=55,y_pixel=60,landmark_type='R_PEAK')])
        self.assertEqual(out.size,img.size)
    def test_roi(self):
        self.assertEqual(validated_roi((10,12,90,80),(200,100)),(10,12,90,80))
        with self.assertRaises(ValueError):validated_roi((90,12,10,80),(200,100))

if __name__=='__main__':unittest.main()
