import unittest
from PIL import Image
from core.fiducial_2d import (evaluate_fiducial_2d, normalize_references,
                              fiducial_2d_overlay)


def report(points):
    return {'hints':[{'hint_id':f'H{i:03d}', 'x_pixel':x,
        'y_hint_pixel':y,'status':s} for i,(x,y,s) in enumerate(points,1)]}


class Fiducial2DTest(unittest.TestCase):
    def test_x_only_not_2d(self):
        out=evaluate_fiducial_2d(report([(101,70,'PRIORITY_REVIEW')]),
              [{'reference_id':'R001','x_pixel':100,'y_pixel':10}],reference_complete=True)
        self.assertEqual(out['xy_within_tolerance_count'],0)
        self.assertEqual(out['x_only_y_mismatch_count'],1)
        self.assertEqual(out['metrics']['priority_xy_reference_coverage'],0)

    def test_2d_match(self):
        o=evaluate_fiducial_2d(report([(101,11,'PRIORITY_REVIEW')]),
             [{'reference_id':'R001','x_pixel':100,'y_pixel':10}],reference_complete=True)
        self.assertEqual(o['xy_within_tolerance_count'],1)
        self.assertEqual(o['priority_xy_within_tolerance_count'],1)
        self.assertEqual(o['matches'][0]['morphology_qc'],'UNVERIFIED')

    def test_one_to_one(self):
        o=evaluate_fiducial_2d(report([(100,10,'PRIORITY_REVIEW')]),
          [{'reference_id':'R001','x_pixel':100,'y_pixel':10},
           {'reference_id':'R002','x_pixel':101,'y_pixel':10}])
        self.assertEqual(o['xy_within_tolerance_count'],1)

    def test_x_only_cannot_consume_better_xy(self):
        o=evaluate_fiducial_2d(report([(100,80,'PRIORITY_REVIEW'),
                                         (103,11,'UNRESOLVED')]),
            [{'reference_id':'R001','x_pixel':100,'y_pixel':10}])
        self.assertEqual(o['xy_within_tolerance_count'],1)
        self.assertEqual(o['priority_xy_within_tolerance_count'],0)

    def test_partial_never_scores(self):
        o=evaluate_fiducial_2d(report([(100,10,'PRIORITY_REVIEW')]),
          [{'x_pixel':100,'y_pixel':10}],reference_complete=False)
        self.assertIsNone(o['metrics'])

    def test_qs_is_separate_type(self):
        p=normalize_references([{'x_pixel':100,'y_pixel':70,'fiducial_type':'QS_NADIR'}])
        self.assertEqual(p[0]['fiducial_type'],'QS_NADIR')
        self.assertRaises(ValueError,normalize_references,[{'fiducial_type':'T_PEAK'}])

    def test_overlay(self):
        r=report([(50,20,'UNRESOLVED')])
        ref=[{'x_pixel':50,'y_pixel':40}]
        out=evaluate_fiducial_2d(r,ref)
        im=fiducial_2d_overlay(Image.new('RGB',(100,100),'white'),r,ref,out)
        self.assertEqual(im.size,(100,100))

    def test_unresolved_preserved(self):
        r=report([(50,20,'UNRESOLVED')]);ref=[{'x_pixel':50,'y_pixel':20}]
        out=evaluate_fiducial_2d(r,ref)
        self.assertEqual(r['hints'][0]['status'],'UNRESOLVED')
        self.assertEqual(out['xy_within_tolerance_count'],1)

if __name__=='__main__':unittest.main()
