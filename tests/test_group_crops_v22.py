import unittest
from PIL import Image
from core.visual_reader_v21 import layout_boxes
from core.group_crops_v22 import (all_panels, required_groups, group_panel, GROUPS,
                                  consistency_flags,validate_roi)
class GroupTests(unittest.TestCase):
    def setUp(self): self.im=Image.new('RGB',(2400,900),'white')
    def test_3x4_groups(self):
        b=layout_boxes(self.im.size,'3x4',rhythm=True)
        p=all_panels(self.im,b)
        self.assertEqual(len(p),5)
        self.assertEqual(p[2].leads,('V1','V2','V3'))
        self.assertEqual(p[-1].leads,('II-rhythm',))
    def test_6x2_without_rhythm(self):
        b=layout_boxes(self.im.size,'6x2',rhythm=False)
        p=all_panels(self.im,b)
        self.assertEqual(len(p),4)
        self.assertNotIn('II-rhythm',b)
        self.assertEqual(p[3].leads,('V4','V5','V6'))
    def test_6x2_with_real_rhythm(self):
        b=layout_boxes(self.im.size,'6x2',rhythm=True)
        self.assertEqual(len(all_panels(self.im,b)),5)
    def test_error_missing_lead(self):
        b=layout_boxes(self.im.size,'3x4',rhythm=False)
        b.pop('aVR')
        with self.assertRaises(ValueError): required_groups(b)
    def test_bad_roi(self):
        with self.assertRaises(ValueError):validate_roi((1,0,2600,30),self.im.size)
    def test_per_lead_override_reflected(self):
        b=layout_boxes(self.im.size,'3x4',rhythm=True)
        b['V1']=(1100,10,1300,300)
        p=group_panel(self.im,b,'PRECORDIAL_V1_V3')
        self.assertEqual(p.roi_boxes['V1'],(1100,10,1300,300))
    def test_consistency(self):
        first={'rhythm_observation':'nhịp đều','hypotheses':[{'type':'IRREGULAR_RHYTHM','evidence':'Nhịp đều đặn','leads':['II']} ]}
        second={'lead_observations':{'II':{'observation':'Nhịp đều'}}}
        flags=consistency_flags(first,second,['LIMB_I_III'])
        self.assertIn('CONTRADICTORY_RHYTHM_HYPOTHESIS',[x['code'] for x in flags])
    def test_no_fabrication(self):
        b=layout_boxes(self.im.size,'6x2',rhythm=False)
        self.assertNotIn('II-rhythm',b)
        self.assertNotIn('RHYTHM_II',required_groups(b))
if __name__=='__main__':unittest.main()
