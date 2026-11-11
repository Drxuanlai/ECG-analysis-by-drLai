import unittest
from PIL import Image
from core.visual_reader_v21 import layout_boxes
from core.group_crops_v231 import default_direct_boxes,direct_panels,validate_direct_group_boxes

class DirectGroupTests(unittest.TestCase):
    def setUp(self):
        self.image=Image.new('RGB',(2400,900),'white')
        pix=self.image.load()
        for y in range(900):
            for x in range(2400):
                pix[x,y]=((x//7)%256,(y//5)%256,(x+y)%256)
    def test_3x4_has_five_rectangles(self):
        g=default_direct_boxes(self.image.size,'3x4',True,.78)
        self.assertEqual(len(g),5)
        self.assertEqual(len(validate_direct_group_boxes(g,self.image.size,'3x4',True)),5)
        leads=layout_boxes(self.image.size,'3x4',rhythm=True)
        panels=direct_panels(self.image,g,leads)
        self.assertEqual(len(panels),5)
        for p in panels:
            self.assertEqual(p.image.size,(p.group_box[2]-p.group_box[0],p.group_box[3]-p.group_box[1]))
            self.assertEqual(p.image.getpixel((11,19)),self.image.getpixel((p.group_box[0]+11,p.group_box[1]+19)))
    def test_6x2_four_groups_no_fake_rhythm(self):
        g=default_direct_boxes(self.image.size,'6x2',False)
        self.assertEqual(len(g),4)
        p=direct_panels(self.image,g,layout_boxes(self.image.size,'6x2',rhythm=False))
        self.assertEqual(p[-1].leads,('V4','V5','V6'))
    def test_6x2_with_rhythm_five(self):
        self.assertEqual(len(default_direct_boxes(self.image.size,'6x2',True)),5)
    def test_overlap_rejected(self):
        g=default_direct_boxes(self.image.size,'3x4',True)
        g['AUGMENTED']=g['LIMB_I_III']
        with self.assertRaises(ValueError):validate_direct_group_boxes(g,self.image.size,'3x4',True)
    def test_group_does_not_change_with_lead_adjustment(self):
        g=default_direct_boxes(self.image.size,'3x4',True)
        lead_boxes=layout_boxes(self.image.size,'3x4',rhythm=True)
        original=direct_panels(self.image,g,lead_boxes)[0].image.tobytes()
        lead_boxes['II']=(200,100,400,300)
        self.assertEqual(original,direct_panels(self.image,g,lead_boxes)[0].image.tobytes())
    def test_wrong_layout(self):
        with self.assertRaises(ValueError): default_direct_boxes(self.image.size,'12x1')
if __name__=='__main__':unittest.main()
