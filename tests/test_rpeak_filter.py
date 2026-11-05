import unittest
import numpy as np
from PIL import Image, ImageDraw
from core.rpeak_filter import filter_rpeak_hints, filter_overlay, compare_with_references

class TestFilter(unittest.TestCase):
    def setUp(self):
        im=Image.new('RGB',(310,100),'white')
        d=ImageDraw.Draw(im)
        for x in (45,118,212):
            d.line((x,72,x,25),fill='black',width=2)
            d.line((x,72,x+5,77),fill='black',width=2)
        self.im=im
        self.hints=[{'hint_id':f'H{i+1:03d}','x_pixel':x,'y_pixel':30,'strength_heuristic':1.,'flags':[]}
                    for i,x in enumerate((20,45,85,118,160,212,275))]
    def test_every_hint_retained(self):
        out=filter_rpeak_hints(self.im,self.hints,shortlist_fraction=.5)
        self.assertEqual(out['raw_hint_count'],7)
        self.assertEqual(len(out['hints']),7)
        self.assertEqual(out['priority_count']+out['unresolved_count'],7)
        self.assertTrue(all(p['status'] in ('UNRESOLVED','PRIORITY_REVIEW') for p in out['hints']))
    def test_zero_hints(self):
        out=filter_rpeak_hints(self.im,[])
        self.assertEqual(out['priority_count'],0)
    def test_no_human_data_enters_filter(self):
        a=filter_rpeak_hints(self.im,self.hints)
        b=filter_rpeak_hints(self.im,self.hints)
        self.assertEqual(a,b)
    def test_partial_reference_no_metrics(self):
        report=filter_rpeak_hints(self.im,self.hints)
        refs=[{'reference_id':'R001','x_pixel':45}]
        qa=compare_with_references(report,refs,reference_complete=False)
        self.assertIsNone(qa['metrics'])
    def test_one_to_one_matching(self):
        report={'hints':[{'hint_id':'H001','x_pixel':100,'status':'PRIORITY_REVIEW'},
                         {'hint_id':'H002','x_pixel':101,'status':'PRIORITY_REVIEW'}]}
        refs=[{'reference_id':'R001','x_pixel':100}]
        qa=compare_with_references(report,refs,tolerance_px=5,reference_complete=True)
        self.assertEqual(qa['spatially_matched_count'],1)
        self.assertEqual(qa['metrics']['spatial_precision'],.5)
    def test_overlay_preserves_dimensions(self):
        out=filter_rpeak_hints(self.im,self.hints)
        self.assertEqual(filter_overlay(self.im,out).size,self.im.size)
    def test_invalid_coordinates_and_fraction(self):
        with self.assertRaises(ValueError):filter_rpeak_hints(self.im,self.hints,shortlist_fraction=0)
        with self.assertRaises(ValueError):filter_rpeak_hints(self.im,[{'x_pixel':400}])
    def test_inverted_QRS_does_not_crash(self):
        im=Image.fromarray(255-np.asarray(self.im))
        self.assertEqual(filter_rpeak_hints(im,self.hints)['raw_hint_count'],7)

if __name__=='__main__':unittest.main()
