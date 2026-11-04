import unittest
from PIL import Image
from core.beat_validation import cluster_validate, attach_review, validation_overlay

class BeatValidationTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (1000, 160), 'white')
    def _report(self, xs, flagged=()):
        return {'version':'test', 'candidates':[
            {'id':f'C{i:03d}', 'x_pixel':x, 'score_heuristic':.8,
             'shaft_top_pixel':30,'shaft_bottom_pixel':110,
             'flags':['edge_touch'] if i in flagged else []}
            for i,x in enumerate(xs,1)]}
    def test_close_shaft_clusters_without_rr(self):
        out=cluster_validate(self.image,self._report([120,132,400,800]))
        self.assertEqual(out['event_count'],3)
        self.assertEqual(out['events'][0]['source_candidate_count'],2)
    def test_irregular_rr_is_preserved(self):
        out=cluster_validate(self.image,self._report([100,260,640,700]))
        self.assertEqual(out['event_count'],4)
    def test_dense_zone_unresolved_not_discarded(self):
        out=cluster_validate(self.image,self._report([100,400,710,732,755,778,801,824]))
        self.assertGreater(out['machine_status_counts']['UNRESOLVED'],0)
        self.assertEqual(out['source_candidate_count'],8)
    def test_edge_flags_mark_unresolved(self):
        out=cluster_validate(self.image,self._report([300],flagged=[1]))
        self.assertEqual(out['events'][0]['machine_status'],'UNRESOLVED')
    def test_human_review_independent_of_machine(self):
        out=cluster_validate(self.image,self._report([100,400]))
        got=attach_review(out,{'E001':'QRS','E002':'ARTIFACT'},[230])
        self.assertEqual(got['human_confirmed_qrs_count'],1)
        self.assertEqual(got['reviewer_added_missed_qrs_count'],1)
        self.assertTrue(got['manual_review_completed'])
        self.assertEqual(out['events'][0]['review_label'],'UNREVIEWED')
    def test_reject_invalid_label_and_missing_position(self):
        out=cluster_validate(self.image,self._report([100]))
        with self.assertRaises(ValueError): attach_review(out,{'E001':'CONFIRMED_BY_AI'})
        with self.assertRaises(ValueError): attach_review(out,{},[-1])
    def test_overlay(self):
        out=cluster_validate(self.image,self._report([100]))
        self.assertEqual(validation_overlay(self.image,out).size,self.image.size)

if __name__=='__main__': unittest.main()
