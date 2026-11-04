import unittest
import numpy as np
from PIL import Image, ImageDraw
from core.rpeak_review import (add_reference_point, independent_rpeak_proposals,
                               match_references, mark_overlay)

class RPeakReviewTests(unittest.TestCase):
    def test_reference_click_and_sort(self):
        pts=add_reference_point([],90,10,width=200,height=50)
        pts=add_reference_point(pts,20,35,width=200,height=50)
        self.assertEqual([p['x_pixel'] for p in pts],[20,90])
        self.assertEqual([p['reference_id'] for p in pts],['R001','R002'])
    def test_duplicate_and_bounds(self):
        pts=add_reference_point([],30,20,width=100,height=50)
        with self.assertRaises(ValueError): add_reference_point(pts,31,30,width=100,height=50)
        with self.assertRaises(ValueError): add_reference_point([],101,30,width=100,height=50)
    def test_artifact_label_untouched_and_conflict(self):
        ev=[{'event_id':'E001','x_pixel':50,'review_label':'ARTIFACT'},
            {'event_id':'E002','x_pixel':80,'review_label':'QRS'}]
        refs=[{'reference_id':'R001','x_pixel':50},{'reference_id':'R002','x_pixel':80}]
        score=match_references(ev,refs,tolerance_px=8,review_complete=True)
        self.assertEqual(score['artifact_label_conflicts'],['E001'])
        self.assertFalse(score['exhaustive_scoring_allowed'])
        self.assertEqual(ev[0]['review_label'],'ARTIFACT')
    def test_not_scored_unless_complete(self):
        ev=[{'event_id':'E001','x_pixel':30,'review_label':'QRS'}]
        refs=[{'reference_id':'R001','x_pixel':30}]
        self.assertIsNone(match_references(ev,refs)['performance_metrics'])
    def test_one_to_one(self):
        ev=[{'event_id':'E001','x_pixel':30,'review_label':'QRS'},
            {'event_id':'E002','x_pixel':32,'review_label':'QRS'}]
        refs=[{'reference_id':'R001','x_pixel':31}]
        s=match_references(ev,refs,review_complete=True)
        self.assertEqual(s['counts']['matched_events'],1)
    def test_no_regular_rr_requirement(self):
        im=Image.new('RGB',(400,100),'white');d=ImageDraw.Draw(im)
        for x in [30,68,170,184,310]:d.line((x,25,x,70),fill='black',width=2)
        a=independent_rpeak_proposals(im)
        self.assertEqual(a['status'],'REVIEW')
        self.assertIn('hints',a)
        self.assertEqual(mark_overlay(im,[],[],[]).size,im.size)

if __name__ == '__main__':unittest.main()

class RPeakLocalizationTests(unittest.TestCase):
    def test_refinement_is_non_destructive(self):
        from core.rpeak_review import event_rpeak_localization
        im=Image.new('RGB',(250,100),'white'); d=ImageDraw.Draw(im)
        d.line((100,20,100,80),fill='black',width=3)
        ev=[{'event_id':'E003','x_pixel':110,'review_label':'ARTIFACT'}]
        out=event_rpeak_localization(im,ev)
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['x_event'],110)
        self.assertEqual(ev[0]['review_label'],'ARTIFACT')
        self.assertIsNotNone(out[0]['x_rpeak_hint'])
