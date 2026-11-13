import unittest
from core.qrs_inventory_v242 import strip_tiles,parse_tile,merge_tile_inventory,reconcile,inventory_prompt

class InventoryTests(unittest.TestCase):
 def test_tiles_cover(self):
  tiles=strip_tiles(2453,950,130)
  self.assertEqual((tiles[0]['x0'],tiles[-1]['x1']),(0,2453))
  self.assertTrue(all(b['x0']<a['x1'] for a,b in zip(tiles,tiles[1:])))
 def test_one_tile(self):self.assertEqual(len(strip_tiles(300)),1)
 def test_bad_overlaps(self):
  with self.assertRaises(ValueError):strip_tiles(1000,90,60)
 def test_prompt_no_rr(self):self.assertIn('Do NOT infer count',inventory_prompt('II',400,100,{'tile_id':'T01'}))
 def test_parse(self):
  t={'tile_id':'T01','x0':100,'x1':300}
  r=parse_tile({'qrs_count':1,'qrs':[{'x_pixel':30,'y_pixel':20}]},t,100)
  self.assertEqual(r['qrs'][0]['x_pixel'],130)
  self.assertTrue(r['count_consistent'])
 def test_bad_count(self):
  r=parse_tile({'qrs_count':3,'qrs':[{'x_pixel':30}]},{'tile_id':'T01','x0':0,'x1':100},50)
  self.assertFalse(r['count_consistent'])
 def test_null_not_bogus(self):
  r=parse_tile({'qrs_count':1,'qrs':[{'x_pixel':None}]},{'tile_id':'T01','x0':0,'x1':100},50)
  self.assertEqual(r['qrs'],[])
 def test_overlap_merge(self):
  a={'tile_id':'T01','count_consistent':True,'qrs':[{'x_pixel':100,'tile_id':'T01'}]}
  b={'tile_id':'T02','count_consistent':True,'qrs':[{'x_pixel':107,'tile_id':'T02'}]}
  m=merge_tile_inventory([a,b]);self.assertEqual(len(m['qrs']),1)
  self.assertTrue(m['qrs'][0]['overlap_merge_review'])
 def test_reconcile(self):
  inv={'qrs':[{'inventory_id':'Q001','x_pixel':100},{'inventory_id':'Q002','x_pixel':240}]}
  cs=[{'candidate_id':'C001','x_pixel':105},{'candidate_id':'C002','x_pixel':165}]
  r=reconcile(inv,cs);self.assertEqual(len(r['matches']),1)
  self.assertEqual(len(r['ai_only']),1);self.assertEqual(len(r['image_only']),1)
 def test_short_rr_retained(self):
  a={'tile_id':'T01','count_consistent':True,'qrs':[{'x_pixel':100,'tile_id':'T01'},{'x_pixel':130,'tile_id':'T01'}]}
  self.assertEqual(len(merge_tile_inventory([a])['qrs']),2)
if __name__=='__main__':unittest.main()
