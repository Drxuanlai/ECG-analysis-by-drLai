"""ECG v2.5 workflow gates. Research-only; no automated clinical verification.

No inference of true QRS from the count, proximity or model agreement.
"""
from __future__ import annotations
from typing import Any


def tile_coverage(tiles: list[dict], reports: list[dict]) -> dict:
    """A tile is complete only if parsed, counts agree, and no parse errors."""
    by_id = {r.get('tile_id'): r for r in reports if isinstance(r, dict)}
    statuses = []
    for tile in tiles:
        key = tile['tile_id']
        r = by_id.get(key)
        if r is None:
            state = 'MISSING'
        elif not r.get('count_consistent', False):
            state = 'COUNT_OR_COORDINATE_ERROR'
        else:
            state = 'COMPLETE'
        statuses.append({'tile_id':key, 'x0':tile['x0'], 'x1':tile['x1'],
                         'status':state, 'qrs_proposed':len(r.get('qrs',[])) if r else 0})
    return {'tiles':statuses, 'complete':all(s['status']=='COMPLETE' for s in statuses),
            'retry_ids':[s['tile_id'] for s in statuses if s['status']!='COMPLETE'],
            'n_complete':sum(s['status']=='COMPLETE' for s in statuses), 'n_total':len(statuses)}


def reconcile_proposals(ai_inventory:dict, python_candidates:list[dict], tolerance_px:int=18)->list[dict]:
    """Union of AI and algorithm candidates for visual adjudication, not automatic confirmation.

    Every proposal receives stable value based on its origin; two nearby peaks are
    recorded as one review row without asserting either is an actual QRS.
    """
    ai = sorted(ai_inventory.get('qrs',[]),key=lambda e:e['x_pixel'])
    py = sorted(python_candidates,key=lambda e:e['x_pixel'])
    edges = sorted((abs(a['x_pixel']-c['x_pixel']),i,j)
                   for i,a in enumerate(ai) for j,c in enumerate(py)
                   if abs(a['x_pixel']-c['x_pixel']) <= tolerance_px)
    used_ai=set();used_py=set();out=[]
    for dist,i,j in edges:
        if i in used_ai or j in used_py:continue
        used_ai.add(i);used_py.add(j)
        a,c=ai[i],py[j]
        out.append({'review_id':f"BOTH_{a['inventory_id']}_{c['candidate_id']}",
                    'x_pixel':a['x_pixel'], 'x_ai':a['x_pixel'], 'x_python':c['x_pixel'],
                    'ai_id':a['inventory_id'], 'candidate_id':c['candidate_id'],
                    'delta_px':dist,'origin':'AI_AND_PYTHON','review_status':'UNVERIFIED'})
    for i,a in enumerate(ai):
        if i not in used_ai:
            out.append({'review_id':f"AI_{a['inventory_id']}",'x_pixel':a['x_pixel'],
                        'x_ai':a['x_pixel'],'x_python':None,'ai_id':a['inventory_id'],
                        'candidate_id':None,'delta_px':None,'origin':'AI_ONLY',
                        'review_status':'UNVERIFIED'})
    for j,c in enumerate(py):
        if j not in used_py:
            out.append({'review_id':f"PY_{c['candidate_id']}",'x_pixel':c['x_pixel'],
                        'x_ai':None,'x_python':c['x_pixel'],'ai_id':None,
                        'candidate_id':c['candidate_id'],'delta_px':None,
                        'origin':'PYTHON_ONLY','review_status':'UNVERIFIED'})
    return sorted(out,key=lambda x:(x['x_pixel'],x['review_id']))


def qa_gate(coverage:dict, selected_review_ids:list[str], union:list[dict],
            clinician_all_beats_reviewed:bool, confirmed_rhythm:bool)->dict:
    allowed={e['review_id'] for e in union}
    invalid=[x for x in selected_review_ids if x not in allowed]
    blockers=[]
    if not confirmed_rhythm:blockers.append('NO_LONG_RHYTHM_STRIP')
    if not coverage.get('complete'):blockers.append('INCOMPLETE_AI_INVENTORY')
    if invalid:blockers.append('STALE_SELECTION')
    if not clinician_all_beats_reviewed:blockers.append('CLINICIAN_REVIEW_NOT_COMPLETE')
    if not selected_review_ids:blockers.append('NO_QRS_SELECTED')
    return {'can_build_windows':not blockers,'blockers':blockers,
            'selected_count':len(selected_review_ids),
            'status':'CLINICIAN_GUIDED_QA_ONLY' if not blockers else 'BLOCKED'}


def visual_coverage(expected_leads:list[str],second:dict)->dict:
    observations=second.get('lead_observations',{}) if isinstance(second,dict) else {}
    missing=[];uncertain=[]
    for lead in expected_leads:
        ob=observations.get(lead)
        if not isinstance(ob,dict):missing.append(lead);continue
        values=[str(ob.get(k,'')).strip() for k in ('observation','p','qrs','st_t')]
        if not values[0] or values[0].upper() in ('UNKNOWN','NOT_REVIEWED','N/A'):
            uncertain.append(lead)
    return {'complete':not missing and not uncertain,'missing':missing,
            'not_observable':uncertain,'expected':list(expected_leads),
            'status':'VISUAL_REVIEW_READY' if not missing and not uncertain else 'VISUAL_INCOMPLETE'}
