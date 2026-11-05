"""v1.5.3 geometry and safe full-ROI review gating."""
import unittest
from core.rpeak_windows import (make_windows, local_to_global, window_for_x,
                                global_to_local_markers, review_summary)

class TestRPeakWindows(unittest.TestCase):
    def test_nonoverlapping_complete_coverage(self):
        for width in [1, 100, 640, 641, 1100, 2556, 4096]:
            windows=make_windows(width)
            self.assertEqual(windows[0]['x_start'],0)
            self.assertEqual(windows[-1]['x_end'],width)
            self.assertTrue(all(a['x_end']==b['x_start'] for a,b in zip(windows,windows[1:])))
            self.assertEqual(sum(w['width'] for w in windows),width)
    def test_case_roi_2556_four_windows(self):
        w=make_windows(2556)
        self.assertEqual(len(w),4)
        self.assertEqual([p['width'] for p in w],[639]*4)
        self.assertEqual(window_for_x(w, 2000),3)
    def test_mapping_display_scale(self):
        w=make_windows(2556)[2]
        x,y=local_to_global(100,50,w,display_width=639,original_height=160)
        self.assertEqual(x,1378)
        self.assertEqual(y,50)
        x,y=local_to_global(50,25,w,display_width=319,original_height=160)
        self.assertAlmostEqual(x,1278+50*639/319)
        self.assertAlmostEqual(y,25*639/319)
    def test_no_wrong_window_crossing(self):
        w=make_windows(2556)
        refs=[{'reference_id':'R001','x_pixel':639,'y_pixel':25},
              {'reference_id':'R002','x_pixel':1278,'y_pixel':26}]
        self.assertEqual(global_to_local_markers(refs,w[0]),[])
        self.assertEqual(global_to_local_markers(refs,w[1])[0]['x_pixel'],0)
        self.assertEqual(global_to_local_markers(refs,w[2])[0]['x_pixel'],0)
        self.assertEqual(refs[0]['x_pixel'],639)
    def test_review_incomplete_gate(self):
        w=make_windows(2556)
        marks=[{'x_pixel':100,'y_pixel':20}]
        state=review_summary(w,{'0':True,'1':True,'2':True},marks)
        self.assertFalse(state['all_windows_reviewed'])
        self.assertEqual(state['reviewed_count'],3)
        state=review_summary(w,{'0':True,'1':True,'2':True,'3':True},marks)
        self.assertTrue(state['all_windows_reviewed'])
    def test_invalid_click_rejected(self):
        w=make_windows(2556)[0]
        with self.assertRaises(ValueError):
            local_to_global(639,10,w,639,160)
    def test_tiny_roi_support(self):
        w=make_windows(89)
        self.assertEqual(len(w),1)
        self.assertEqual(w[0]['width'],89)

if __name__=='__main__': unittest.main()
