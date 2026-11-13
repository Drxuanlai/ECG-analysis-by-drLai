from PIL import Image
from core.full_beat_v241 import (clean_candidates,propose_qrs_candidates,
    full_beat_windows,full_beat_prompt,validate_one_beat_landmarks,candidate_overlay)

def test_t_wave_does_not_become_beat_when_unselected():
    candidates=propose_qrs_candidates([
        {'x_pixel':133,'score':18},
        {'x_pixel':205,'score':6},  # T-wave candidate
        {'x_pixel':340,'score':18},
        {'x_pixel':412,'score':6},  # T-wave candidate
        {'x_pixel':547,'score':18}],900)
    chosen=[c for c in candidates if c['x_pixel'] in [133,340,547]]
    unresolved=[c for c in candidates if c['x_pixel'] not in [133,340,547]]
    windows=full_beat_windows(chosen,900)
    assert len(windows)==3
    assert [w['beat_id'] for w in windows]==['B1','B2','B3']
    assert len(unresolved)==2
    assert windows[0]['x0']<133<205<windows[0]['x1']
    assert windows[1]['x0']<340<412<windows[1]['x1']

def test_irregular_short_coupled_qrs_not_silently_removed():
    # A genuine early complex stays accepted; no forced regular RR.
    windows=full_beat_windows([{'x_pixel':133},{'x_pixel':190},
                               {'x_pixel':410}],800)
    assert len(windows)==3
    assert windows[1]['prev_rr_px']==57
    assert windows[0]['x1']>windows[1]['x0'] # overlap allowed

def test_missing_t_is_not_inferred():
    windows=full_beat_windows([{'x_pixel':70}],105)
    assert windows[0]['x1']==105
    assert 'RIGHT_TRUNCATED' in windows[0]['coverage_warning']
    prompt=full_beat_prompt('II-rhythm',windows[0],(windows[0]['x1']-windows[0]['x0'],100))
    assert 'T_END' in prompt and 'omit' in prompt.lower()
    assert 'NOT' in prompt

def test_candidates_retained_and_deduplicated():
    c=clean_candidates([{'x_pixel':110,'score':4},
         {'x_pixel':113,'score':3},{'x_pixel':170,'score':6}],300)
    assert [c0['x_pixel'] for c0 in c]==[110,170]
    assert all(x['event_status']=='UNRESOLVED_NOT_A_BEAT' for x in c)

def test_qrs_peak_far_from_anchor_flagged_not_discarded():
    w=full_beat_windows([{'x_pixel':100},{'x_pixel':305}],430)
    mark={'beat_id':'B1','landmark_type':'R_PEAK','x_pixel':170,'y_pixel':25}
    got=validate_one_beat_landmarks([mark],w[0])
    assert 'QRS_EXTREMUM_FAR_FROM_SELECTED_ANCHOR' in got[0]['qa_flags']
    assert got[0]['qa_state']=='LOCALIZATION_REVIEW'

def test_group_overlay_keeps_dimensions():
    img=Image.new('RGB',(320,110),'white')
    cand=[{'candidate_id':'C001','x_pixel':80},{'candidate_id':'C002','x_pixel':130}]
    assert candidate_overlay(img,cand,['C001']).size==img.size

def test_candidate_ranking_is_not_a_final_rhythm_classification():
    c=propose_qrs_candidates([{'x_pixel':100,'score':20},
        {'x_pixel':160,'score':4},{'x_pixel':205,'score':19}],300)
    assert len(c)==3
    assert all(x['event_status']=='UNRESOLVED_NOT_A_BEAT' for x in c)
