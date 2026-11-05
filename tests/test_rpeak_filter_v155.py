import unittest
import numpy as np
from PIL import Image, ImageDraw
from core.rpeak_filter_v155 import rank_rpeak_hints

class RPeakRank155Test(unittest.TestCase):
    def make_image(self):
        im=Image.new('RGB',(420,170),'white'); d=ImageDraw.Draw(im)
        d.line((0,100,419,100),fill=(45,45,45),width=2)
        for x in (65,210,355):
            d.line([(x-12,100),(x-4,96),(x,20),(x+4,104),(x+12,100)],fill=(24,24,24),width=3)
        return im
    def test_preserves_all_hints_and_protects_tall_spikes(self):
        hints=[]
        for i,(x,y,s) in enumerate([(65,20,1.1),(210,20,1.1),(355,20,1.1),(110,100,.2),(265,100,.2),(315,100,.2)]):
            hints.append({'hint_id':f'H{i}', 'x_pixel':x, 'y_pixel':y,'strength_heuristic':s})
        out=rank_rpeak_hints(self.make_image(),hints,shortlist_fraction=.5)
        priority={q['x_pixel'] for q in out['hints'] if q['status']=='PRIORITY_REVIEW'}
        self.assertEqual(priority,{65,210,355})
        self.assertEqual(out['raw_hint_count'],6)
        self.assertFalse(out['ranking_evidence']['uses_rr_regularization'])
    def test_missing_labels_not_needed(self):
        out=rank_rpeak_hints(self.make_image(),[{'x_pixel':65,'y_pixel':20}],shortlist_fraction=.5)
        self.assertEqual(out['priority_count'],1)
        self.assertFalse(out['ranking_evidence']['baseline_estimated'])
    def test_retains_outside_shortlist(self):
        hints=[{'x_pixel':x,'y_pixel':100,'strength_heuristic':.2} for x in range(30,230,20)]
        out=rank_rpeak_hints(self.make_image(),hints,shortlist_fraction=.3)
        self.assertEqual(out['priority_count']+out['unresolved_count'],len(hints))
    def test_invalid_fraction(self):
        with self.assertRaises(ValueError):rank_rpeak_hints(self.make_image(),[],shortlist_fraction=0)

if __name__=='__main__': unittest.main()
