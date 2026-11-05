import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image,ImageDraw
from core.qrs_polarity import detect_polarity_components, compare_polarity_to_references,polarity_overlay

class TestPolarity(unittest.TestCase):
    def test_blank_safe(self):
        r=detect_polarity_components(Image.new('RGB',(300,150),'white'))
        self.assertEqual(r['candidate_count'],0)
        self.assertEqual(r['status'],'REVIEW')
    def test_unresolved_not_silently_deleted(self):
        im=Image.new('RGB',(350,180),'white')
        r=detect_polarity_components(im)
        self.assertIsInstance(r.get('excluded_candidates'),list)
    def test_no_human_reference_feedback(self):
        im=Image.new('RGB',(350,180),'white')
        a=detect_polarity_components(im)
        compare_polarity_to_references(a,[{'reference_id':'R001','fiducial_type':'R_PEAK','x_pixel':30,'y_pixel':20}])
        b=detect_polarity_components(im)
        self.assertEqual(a,b)
    def test_xy_matching_strict(self):
        d={'positive_components':[{'x_pixel':30,'y_pixel':20}], 'negative_components':[]}
        out=compare_polarity_to_references(d,[{'reference_id':'1','fiducial_type':'R_PEAK','x_pixel':30,'y_pixel':65}],tol_x=10,tol_y=8)
        self.assertEqual(out['xy_match_count'],0)
        self.assertEqual(out['matches'][0]['status'],'X_ONLY_Y_MISMATCH')
    def test_overlay_no_mutation(self):
        im=Image.new('RGB',(350,180),'white')
        before=im.tobytes()
        out=polarity_overlay(im,{'positive_components':[],'negative_components':[]})
        self.assertEqual(im.tobytes(),before)
        self.assertEqual(out.size,im.size)

if __name__=='__main__':unittest.main()
