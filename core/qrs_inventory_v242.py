"""v2.4.2 - visual QRS inventory and independent candidate reconciliation.

Research only. No counts/marks inferred from heart rate or regular rhythm.
AI visual proposals and image-detector proposals remain UNVERIFIED.
"""
from __future__ import annotations
import math
from PIL import Image


def strip_tiles(width:int, tile_width:int=950, overlap:int=130)->list[dict]:
    if width<1 or tile_width<80 or not 0<=overlap<tile_width//2:
        raise ValueError('Invalid tile parameters')
    if width<=tile_width:return [{'tile_id':'T01','x0':0,'x1':width}]
    out=[];x=0
    while x<width:
        end=min(width,x+tile_width)
        out.append({'tile_id':f'T{len(out)+1:02d}','x0':x,'x1':end})
        if end==width:break
        x=end-overlap
    return out


def inventory_prompt(lead:str, width:int, height:int, tile:dict)->str:
    # All coordinates asked within each tile. Never leak previous model answers or manual marks.
    return f'''ECG IMAGE OBSERVATION TASK: QRS INVENTORY ONLY. Research, not clinical diagnosis.
This image is one tile from {lead}, dimensions EXACTLY {width} by {height} pixels.
Local origin upper left (0,0); x grows right, y grows downward.
Tile {tile['tile_id']} is part of a longer rhythm strip. A waveform may be truncated
at a tile edge. Count ONLY complete, VISUALLY IDENTIFIABLE QRS complexes in THIS tile.
Do NOT infer count or locations from average RR, regularity, printed heart rate,
or previous descriptions. Do NOT count P waves, broad T waves, ECG grid, text,
calibration pulses, or photo artifacts as QRS. A truly early, wide, negative, or
low-voltage QRS can still be QRS: do not use an assumed minimum RR to discard it.
For EVERY claimed QRS, provide a LOCAL pixel x and y on the actual prominent
QRS extremum or morphological center (it need not be a positive R peak),
with morphology evidence. If ambiguous give an uncertain event separately.
Do not manufacture exact coordinates: null when localization is unsupported.
A QRS crossing a tile edge must be placed in partial_events, not counted as complete.
Scan from LEFT to RIGHT a second time. Output ONLY JSON:
{{"qrs_count":0,"qrs":[{{"x_pixel":100,"y_pixel":50,"morphology":"narrow sharp complex","evidence":"visible ..."}}],
"uncertain_events":[{{"x_pixel":null,"y_pixel":null,"reason":"QRS or T unclear"}}],
"partial_events":[{{"x_pixel":null,"reason":"truncated at edge"}}],"quality_notes":""}}
The number qrs_count MUST equal the length of qrs; if not sure, move ambiguous
entries to uncertain_events. Never return a diagnosis.'''


def _coord(value, upper):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):return None
    v=int(round(value))
    return v if 0<=v<upper else None


def parse_tile(payload:dict, tile:dict, height:int)->dict:
    w=tile['x1']-tile['x0']
    if not isinstance(payload,dict):payload={}
    raw=payload.get('qrs',[])
    if not isinstance(raw,list):raw=[]
    events=[]
    for item in raw[:80]:
        if not isinstance(item,dict):continue
        x=_coord(item.get('x_pixel'),w)
        y=_coord(item.get('y_pixel'),height)
        if x is None:continue
        events.append({'x_pixel':tile['x0']+x,'y_pixel':y,
                       'morphology':str(item.get('morphology',''))[:180],
                       'evidence':str(item.get('evidence',''))[:240],
                       'tile_id':tile['tile_id'],'source':'AI_VISUAL_INVENTORY_UNVERIFIED',
                       'status':'REVIEW_REQUIRED'})
    ambiguous=[]
    for item in payload.get('uncertain_events',[]) if isinstance(payload.get('uncertain_events'),list) else []:
        if not isinstance(item,dict):continue
        x=_coord(item.get('x_pixel'),w)
        ambiguous.append({'x_pixel':tile['x0']+x if x is not None else None,
                          'tile_id':tile['tile_id'],'reason':str(item.get('reason',''))[:240],
                          'status':'UNRESOLVED'})
    partial=[]
    for item in payload.get('partial_events',[]) if isinstance(payload.get('partial_events'),list) else []:
        if not isinstance(item,dict):continue
        x=_coord(item.get('x_pixel'),w)
        partial.append({'x_pixel':tile['x0']+x if x is not None else None,
                        'tile_id':tile['tile_id'],'reason':str(item.get('reason',''))[:240],
                        'status':'PARTIAL_NOT_COUNTED'})
    reported=payload.get('qrs_count')
    count_agrees=(isinstance(reported,int) and not isinstance(reported,bool) and reported==len(raw))
    return {'tile_id':tile['tile_id'],'x0':tile['x0'],'x1':tile['x1'],
            'count_reported':reported,'count_parsed':len(events),
            'count_consistent':count_agrees and len(events)==len(raw),
            'qrs':events,'uncertain_events':ambiguous,'partial_events':partial,
            'quality_notes':str(payload.get('quality_notes',''))[:350]}


def merge_tile_inventory(results:list[dict], tolerance_px:int=15)->dict:
    """Merge overlapping tile proposals; ambiguous collisions require review.

    Never silently adjudicate if two distinct true QRS are close: record merge group.
    """
    raw=sorted([dict(e) for r in results for e in r.get('qrs',[])],key=lambda p:p['x_pixel'])
    groups=[]
    for rec in raw:
        if groups and rec['x_pixel']-groups[-1][-1]['x_pixel']<=tolerance_px:
            groups[-1].append(rec)
        else:groups.append([rec])
    qrs=[]
    for idx,group in enumerate(groups,1):
        med=sorted(v['x_pixel'] for v in group)[len(group)//2]
        qrs.append({'inventory_id':f'Q{idx:03d}','x_pixel':med,
                    'y_pixel':next((g['y_pixel'] for g in group if g.get('y_pixel') is not None),None),
                    'tile_ids':sorted({g['tile_id'] for g in group}),
                    'observations':group,'status':'AI_UNVERIFIED',
                    'overlap_merge_review':len(group)>1})
    issues=[]
    for r in results:
        if not r.get('count_consistent',False):
            issues.append({'code':'TILE_COUNT_MISMATCH','tile_id':r.get('tile_id')})
    return {'qrs':qrs,'ai_count_unverified':len(qrs),'tile_results':results,
            'uncertain_events':[e for r in results for e in r.get('uncertain_events',[])],
            'partial_events':[e for r in results for e in r.get('partial_events',[])],
            'issues':issues,'status':'REVIEW_REQUIRED'}


def reconcile(inventory:dict, image_candidates:list[dict], tolerance_px:int=18)->dict:
    """One-to-one proximity matching, not TP/FP/FN without clinician ground truth."""
    ai=list(inventory.get('qrs',[]));img=list(image_candidates)
    edges=sorted((abs(a['x_pixel']-c['x_pixel']),i,j) for i,a in enumerate(ai)
                 for j,c in enumerate(img) if abs(a['x_pixel']-c['x_pixel'])<=tolerance_px)
    used_a=set();used_c=set();pairs=[]
    for d,i,j in edges:
        if i in used_a or j in used_c:continue
        used_a.add(i);used_c.add(j)
        pairs.append({'inventory_id':ai[i]['inventory_id'],
                      'ai_x':ai[i]['x_pixel'],'candidate_id':img[j]['candidate_id'],
                      'candidate_x':img[j]['x_pixel'],'delta_x':d,'status':'AGREEMENT_UNVERIFIED'})
    return {'matches':sorted(pairs,key=lambda p:p['ai_x']),
            'ai_only':[a for i,a in enumerate(ai) if i not in used_a],
            'image_only':[c for j,c in enumerate(img) if j not in used_c],
            'note':'Agreement is NOT proof of QRS. All unmatched events remain REVIEW/UNRESOLVED.'}


def inventory_overlay(roi:Image.Image, inventory:dict, image_candidates:list[dict])->Image.Image:
    from PIL import ImageDraw
    out=roi.convert('RGB').copy();d=ImageDraw.Draw(out)
    for c in image_candidates:
        x=c['x_pixel'];d.line((x,0,x,min(out.height-1,15)),fill=(210,135,35),width=2)
    for q in inventory.get('qrs',[]):
        x=q['x_pixel'];y=q.get('y_pixel')
        d.line((x,max(0,out.height-16),x,out.height-1),fill=(126,60,211),width=3)
        if isinstance(y,int):d.ellipse((x-4,y-4,x+4,y+4),outline=(126,60,211),width=2)
    return out
