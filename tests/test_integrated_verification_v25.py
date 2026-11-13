from core.integrated_verification_v25 import (tile_coverage,reconcile_proposals,qa_gate,visual_coverage)


def test_partial_and_retry():
    tiles=[{'tile_id':'T01','x0':0,'x1':100},{'tile_id':'T02','x0':80,'x1':180}]
    cov=tile_coverage(tiles,[{'tile_id':'T01','qrs':[{}],'count_consistent':True}])
    assert cov['retry_ids']==['T02'] and not cov['complete']
    assert tile_coverage(tiles,[{'tile_id':'T01','count_consistent':True}, {'tile_id':'T02','count_consistent':False}])['retry_ids']==['T02']
    assert tile_coverage(tiles,[{'tile_id':'T01','count_consistent':True}, {'tile_id':'T02','count_consistent':True}])['complete']


def test_union_and_gate():
    inventory={'qrs':[{'inventory_id':'Q001','x_pixel':100},{'inventory_id':'Q002','x_pixel':190}]}
    py=[{'candidate_id':'C01','x_pixel':105},{'candidate_id':'C02','x_pixel':250}]
    rows=reconcile_proposals(inventory,py)
    assert [r['origin'] for r in rows]==['AI_AND_PYTHON','AI_ONLY','PYTHON_ONLY']
    assert len({r['review_id'] for r in rows})==3
    good={'complete':True}
    assert qa_gate(good,[rows[0]['review_id']],rows,True,True)['can_build_windows']
    assert not qa_gate({'complete':False},[rows[0]['review_id']],rows,True,True)['can_build_windows']
    assert not qa_gate(good,['OLD'],rows,True,True)['can_build_windows']
    assert not qa_gate(good,[rows[0]['review_id']],rows,False,True)['can_build_windows']


def test_visual_rhythm_required():
    assert visual_coverage(['I','II-rhythm'],{'lead_observations':{'I':{'observation':'ok'}}})['missing']==['II-rhythm']
    assert visual_coverage(['I'],{'lead_observations':{'I':{'observation':'ok'}}})['complete']
    assert not visual_coverage(['I'],{'lead_observations':{'I':{'observation':'UNKNOWN'}}})['complete']


def test_no_truth_from_agreement():
    rows=reconcile_proposals({'qrs':[{'inventory_id':'Q001','x_pixel':50}]},[{'candidate_id':'C01','x_pixel':50}])
    assert rows[0]['review_status']=='UNVERIFIED'
    assert rows[0]['origin']=='AI_AND_PYTHON'
