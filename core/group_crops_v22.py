"""ECG v2.2: deterministic lead-group crops (research, no clinical validation).

All crop geometry derives from the confirmed per-lead ROI boxes. Group panels
are explicit composites, not falsely claimed to be simultaneous recordings.
"""
from __future__ import annotations
import io
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont

GROUPS = {
    'LIMB_I_III': ('I', 'II', 'III'),
    'AUGMENTED': ('aVR', 'aVL', 'aVF'),
    'PRECORDIAL_V1_V3': ('V1', 'V2', 'V3'),
    'PRECORDIAL_V4_V6': ('V4', 'V5', 'V6'),
    'RHYTHM_II': ('II-rhythm',),
}
MANDATORY_GROUPS = ('LIMB_I_III', 'AUGMENTED', 'PRECORDIAL_V1_V3', 'PRECORDIAL_V4_V6')

@dataclass(frozen=True)
class GroupPanel:
    group_id: str
    leads: tuple[str, ...]
    image: Image.Image
    roi_boxes: dict[str, tuple[int,int,int,int]]


def required_groups(boxes: dict) -> tuple[str, ...]:
    """Never invent rhythm from a short II; absence is explicit."""
    unavailable = [(group, lead) for group in MANDATORY_GROUPS for lead in GROUPS[group] if lead not in boxes]
    if unavailable:
        raise ValueError('Thiếu ROI: ' + ', '.join(f'{lead} ({group})' for group, lead in unavailable))
    return MANDATORY_GROUPS + (('RHYTHM_II',) if 'II-rhythm' in boxes else ())


def validate_roi(box, size):
    w,h = size
    if len(box)!=4 or any(not isinstance(v,(int,float)) for v in box):
        raise ValueError('ROI không hợp lệ')
    x0,y0,x1,y1=(int(v) for v in box)
    if not(0<=x0<x1<=w and 0<=y0<y1<=h):
        raise ValueError(f'ROI vượt ranh giới ảnh: {box}')
    return x0,y0,x1,y1


def group_panel(image: Image.Image, boxes: dict, group_id: str, max_width=1600) -> GroupPanel:
    if group_id not in GROUPS: raise ValueError('Nhóm không hợp lệ')
    leads = GROUPS[group_id]
    if any(l not in boxes for l in leads): raise ValueError('Thiếu chuyển đạo trong nhóm')
    crops=[]; orig_boxes={}
    for lead in leads:
        box=validate_roi(boxes[lead],image.size)
        orig_boxes[lead]=box
        crop=image.crop(box).convert('RGB')
        scale=min(1.0,max_width/crop.width)
        if scale<1:
            crop=crop.resize((int(crop.width*scale),max(1,int(crop.height*scale))), Image.Resampling.LANCZOS)
        crops.append(crop)
    # A group is a composite of *separate* lead rows. This intentionally does
    # not pretend that different columns correspond to the same instant.
    canvas_w=max(c.width for c in crops)
    label_h=28; spacing=10; outer=10
    canvas_h=outer*2+sum(c.height+label_h for c in crops)+spacing*(len(crops)-1)
    result=Image.new('RGB',(canvas_w+outer*2,canvas_h),'white')
    pen=ImageDraw.Draw(result)
    y=outer
    for i,(lead,crop) in enumerate(zip(leads,crops)):
        pen.rectangle((outer,y,canvas_w+outer,y+label_h),fill='#f3f6fa')
        pen.text((outer+7,y+6),f'{lead}   |   {group_id}',fill='#152c40')
        y+=label_h
        result.paste(crop,(outer,y))
        y+=crop.height
        if i<len(leads)-1:
            pen.line((outer,y+spacing//2,canvas_w+outer,y+spacing//2),fill='#b7c8d3',width=2)
            y+=spacing
    return GroupPanel(group_id,leads,result,orig_boxes)


def panel_as_jpeg(panel:GroupPanel,max_side=2600)->bytes:
    im=panel.image.copy()
    im.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    buf=io.BytesIO(); im.save(buf,format='JPEG',quality=91,optimize=True)
    return buf.getvalue()


def all_panels(image,boxes):
    return [group_panel(image,boxes,g) for g in required_groups(boxes)]


def selection_for_followup(report,all_ids):
    """Four mandatory groups always; rhythm iff physically present.

    The initial model does not remove a mandatory group. No diagnosis-driven
    selection is needed to achieve full 12-lead coverage.
    """
    return [g for g in MANDATORY_GROUPS if g in all_ids] + (["RHYTHM_II"] if "RHYTHM_II" in all_ids else [])


def consistency_flags(first,second,groups):
    flags=[]
    selected=set(l for group in groups for l in GROUPS[group])
    observed=set(second.get('lead_observations',{}))
    for lead in sorted(selected-observed):
        flags.append({'code':'MISSING_CROP_OBSERVATION','lead':lead,'details':'AI không mô tả chuyển đạo đã gửi'})
    rhythm_text=str(first.get('rhythm_observation','')).lower()
    for hypothesis in first.get('hypotheses',[]):
        kind=hypothesis.get('type','')
        evidence=str(hypothesis.get('evidence','')).lower()
        if kind=='IRREGULAR_RHYTHM':
            # A flag, not a silent reassignment; free text multilingual checks
            if any(t in evidence for t in ('nhịp đều', 'đều đặn', 'regular rhythm', 'regular sinus')) and not any(t in evidence for t in ('không đều','irregular', 'ngoại tâm thu')):
                flags.append({'code':'CONTRADICTORY_RHYTHM_HYPOTHESIS','lead':'II-rhythm','details':'Nhãn IRREGULAR_RHYTHM nhưng bằng chứng lại nói nhịp đều'})
            if 'không đều' not in rhythm_text and ('nhịp đều' in rhythm_text or 'đều đặn' in rhythm_text):
                flags.append({'code':'RHYTHM_CROSS_FIELD_CHECK','lead':'II-rhythm','details':'Mô tả nhịp đều nhưng AI sinh nghi ngờ IRREGULAR_RHYTHM; cần rà soát'})
        missed=[lead for lead in hypothesis.get('leads',[]) if lead not in observed and lead in selected]
        for lead in missed:
            flags.append({'code':'HYPOTHESIS_LEAD_NOT_RECHECKED','lead':lead,'details':f'{kind} có nhắc lead nhưng lượt 2 không mô tả'})
    return flags


def v22_report(first,second,panels,boxes,layout,image_hash):
    groups=[p.group_id for p in panels]
    flags=consistency_flags(first,second,groups)
    return {
        'schema_version':'2.2', 'status':'RESEARCH_REVIEW_REQUIRED',
        'image_sha256_prefix':image_hash, 'layout_user_confirmed':layout,
        'group_crops':[{'group_id':p.group_id,'leads':list(p.leads),'roi_boxes':{k:list(v) for k,v in p.roi_boxes.items()}} for p in panels],
        'all_boxes':{k:list(v) for k,v in boxes.items()},
        'rhythm_strip_available':'RHYTHM_II' in groups,
        'first_pass':first,'group_pass':second,'consistency_flags':flags,
        'quantitative_verification':{'status':'NOT_AVAILABLE','reason':'Các module đo ECG chưa được nghiệm thu độc lập'},
    }
