"""ECG v2.1 AI visual evidence workflow. Research-only: no automated diagnosis.

This module is self-contained, stateless and independent of legacy ECG code.
"""
from __future__ import annotations
import base64
import hashlib
import io
import json
import re
from dataclasses import dataclass
from typing import Any
from PIL import Image, ImageOps

LEADS = ('I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6')
PATTERNS_3X4 = (('I','aVR','V1','V4'),('II','aVL','V2','V5'),('III','aVF','V3','V6'))
PATTERNS_6X2 = (('I','V1'),('II','V2'),('III','V3'),('aVR','V4'),('aVL','V5'),('aVF','V6'))
ALLOWED_ISSUES = ('IRREGULAR_RHYTHM','P_WAVE','WIDE_QRS','ECTOPY','AF_SUSPECT',
                  'ST_ELEVATION','ST_DEPRESSION','BRUGADA_PATTERN','QT_CONCERN',
                  'CONDUCTION','OTHER')

@dataclass(frozen=True)
class CropSpec:
    lead: str
    box: tuple[int,int,int,int]
    source: str

def image_from_bytes(data: bytes) -> Image.Image:
    # PIL provides image validation. Pixel limits mitigate decompression bombs.
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        if im.width * im.height > 45_000_000:
            raise ValueError('Ảnh quá lớn (>45 triệu pixel); vui lòng thu nhỏ trước.')
        return im.convert('RGB').copy()

def image_identity(img: Image.Image) -> str:
    return hashlib.sha256(img.tobytes()).hexdigest()[:16]

def layout_boxes(size: tuple[int,int], layout: str, edges_x=None, edges_y=None,
                 rhythm: bool=False, rhythm_start: float=.78, margin: float=.025) -> dict[str,tuple[int,int,int,int]]:
    """Return suggested geometrical boxes only, not validated waveform isolation."""
    w,h = size
    patt = PATTERNS_3X4 if layout == '3x4' else PATTERNS_6X2 if layout == '6x2' else None
    if patt is None: raise ValueError('Bố cục phải là 3x4 hoặc 6x2')
    nr,nc=len(patt),len(patt[0]); bottom = rhythm_start if rhythm else .96
    if not 0.5 < bottom <= .99: raise ValueError('Giới hạn vùng nhịp không hợp lệ')
    xs=list(edges_x) if edges_x is not None else [0.02+(0.96*i/nc) for i in range(nc+1)]
    ys=list(edges_y) if edges_y is not None else [0.12+(bottom-.12)*i/nr for i in range(nr+1)]
    if len(xs)!=nc+1 or len(ys)!=nr+1 or not all(a<b for a,b in zip(xs,xs[1:])) or not all(a<b for a,b in zip(ys,ys[1:])): raise ValueError('Ranh giới bố cục không hợp lệ')
    if min(xs)<0 or max(xs)>1 or min(ys)<0 or max(ys)>1: raise ValueError('Ranh giới ngoài ảnh')
    boxes={}
    for row in range(nr):
        for col in range(nc):
            l,t,r,b=xs[col],ys[row],xs[col+1],ys[row+1]
            # Small margin to reduce adjacent leads, but may clip deep S: clinician must review.
            x0=int(w*(l+margin/nc)); x1=int(w*(r-margin/nc))
            y0=int(h*(t+margin/nr)); y1=int(h*(b-margin/nr))
            if x1<=x0 or y1<=y0: raise ValueError('ROI trống')
            boxes[patt[row][col]]=(x0,y0,x1,y1)
    if rhythm:
        boxes['II-rhythm']=(int(w*.02),int(h*(rhythm_start+.008)),int(w*.98),int(h*.97))
    return boxes

def image_for_model(img:Image.Image,max_side:int=3000) -> bytes:
    out=img.copy(); out.thumbnail((max_side,max_side),Image.Resampling.LANCZOS)
    buf=io.BytesIO(); out.save(buf,format='JPEG',quality=89,optimize=True)
    return buf.getvalue()

def crop_images(img:Image.Image,boxes:dict[str,tuple[int,int,int,int]],leads:list[str]) -> list[tuple[str,bytes]]:
    result=[]
    for lead in leads:
        if lead not in boxes: continue
        result.append((lead,image_for_model(img.crop(boxes[lead]),max_side=1800)))
    return result

def extract_json(text:str) -> dict[str,Any]:
    s=text.strip()
    if s.startswith('```'):
        s=re.sub(r'^```(?:json)?\s*','',s)
        s=re.sub(r'\s*```\s*$','',s)
    try: obj=json.loads(s)
    except json.JSONDecodeError:
        start=s.find('{'); end=s.rfind('}')
        if start<0 or end<start: raise ValueError('AI không trả JSON có thể đọc được.')
        obj=json.loads(s[start:end+1])
    if not isinstance(obj,dict): raise ValueError('Kết quả AI cần là đối tượng JSON')
    return obj

def sanitized_observations(raw:dict) -> dict:
    """Constrain AI output; never silently replace missing observations with 'normal'."""
    results={lead:{'visibility':'UNKNOWN','observation':'Chưa đủ bằng chứng đọc chuyển đạo này','p':'UNKNOWN','qrs':'UNKNOWN','st_t':'UNKNOWN'} for lead in LEADS}
    records=raw.get('lead_observations',[])
    if isinstance(records,list):
        for rec in records:
            if not isinstance(rec,dict) or rec.get('lead') not in results:continue
            lead=rec['lead']
            results[lead]={k:str(rec.get(k,'UNKNOWN'))[:750] for k in ('visibility','observation','p','qrs','st_t')}
    issues=[]
    records=raw.get('hypotheses',[])
    if isinstance(records,list):
        for rec in records[:30]:
            if not isinstance(rec,dict):continue
            issue=str(rec.get('type','OTHER')).upper()
            if issue not in ALLOWED_ISSUES: issue='OTHER'
            leads=[x for x in rec.get('leads',[]) if x in LEADS or x=='II-rhythm'] if isinstance(rec.get('leads'),list) else []
            issues.append({'type':issue,'leads':leads[:12], 'evidence':str(rec.get('evidence',''))[:900],
                           'uncertainty':str(rec.get('uncertainty',''))[:500],
                           'needs_verification':True})
    return {'lead_observations':results,'hypotheses':issues,
            'rhythm_observation':str(raw.get('rhythm_observation','UNKNOWN'))[:1000],
            'calibration':str(raw.get('calibration','UNKNOWN'))[:500],
            'image_limitations':str(raw.get('image_limitations',''))[:1000]}

def choose_leads(report:dict,available:set[str],max_leads:int=6) -> list[str]:
    """Checklist ensures some leads rechecked even if the model missed a finding."""
    choices=[]
    def add(lead):
        if lead in available and lead not in choices and len(choices)<max_leads:choices.append(lead)
    add('II-rhythm'); add('II'); add('V1'); add('V2')
    for hypo in report.get('hypotheses',[]):
        for lead in hypo.get('leads',[]): add(lead)
    for lead in ('V5','V6','aVR','III','aVF','I','V3','V4','aVL'):
        add(lead)
    return choices

FIRST_PASS_PROMPT = '''Bạn là AI đọc hình ECG cho một công cụ NGHIÊN CỨU có bác sĩ giám sát. Đọc CHỈ ảnh được gửi, không suy từ chẩn đoán máy in, không dùng kiến thức về hình ảnh khác. Đây là vòng quan sát độc lập. BẮT BUỘC mô tả đủ 12 lead I II III aVR aVL aVF V1-V6, từng lead phải ghi khi không thấy rõ. Đầu tiên phân biệt: chữ ghi tốc độ giấy/gain có nhìn thấy hay không, xung chuẩn có đầy đủ không; không tự khẳng định 25 mm/s hay 10 mm/mV. Xác định nhịp, P trước QRS, đều/không đều, độ rộng QRS bằng mô tả thị giác, ST-T theo từng lead và những bất thường cần đối chiếu. KHÔNG đo ms/mV từ ảnh bằng suy đoán. KHÔNG tự gọi hội chứng Brugada, STEMI hay rung nhĩ là chẩn đoán xác định. Không để câu hỏi hay gợi ý từ người dùng quyết định quan sát. Với mâu thuẫn/thiếu rõ ràng, ghi UNKNOWN. JSON duy nhất, không markdown, với schema:\n{"calibration":"...", "rhythm_observation":"...", "image_limitations":"...", "lead_observations":[{"lead":"I", "visibility":"CLEAR|LIMITED|UNKNOWN", "observation":"mô tả", "p":"...", "qrs":"...", "st_t":"..."}], "hypotheses":[{"type":"IRREGULAR_RHYTHM|P_WAVE|WIDE_QRS|ECTOPY|AF_SUSPECT|ST_ELEVATION|ST_DEPRESSION|BRUGADA_PATTERN|QT_CONCERN|CONDUCTION|OTHER", "leads":["V1"], "evidence":"các dấu hiệu cụ thể nhìn thấy", "uncertainty":"..."}]}'''

SECOND_PASS_PROMPT = '''Đây là lần quan sát độc lập ở ảnh PHÓNG TO của ECG, mỗi ảnh có nhãn chuyển đạo đặt trong mô tả input (nhãn do hệ thống gợi ý, cần nói nếu ảnh không phù hợp). Không được dựa vào kết luận từ lần đọc đầu; chỉ nhìn các ảnh crop đang gửi. Đọc riêng P, hình thái QRS (kể cả rS, QS, R'), J-ST và cực tính T. Khi không chắc, ghi UNKNOWN. Không tạo số đo QRS/QT/ST chính xác từ ảnh nếu hiệu chuẩn chưa chắc. Không tự chẩn đoán Brugada, STEMI, AF. Chỉ JSON:\n{"lead_observations":[{"lead":"V1","visibility":"CLEAR|LIMITED|UNKNOWN","observation":"...","p":"...","qrs":"...","st_t":"..."}],"conflicts_or_limits":"..."}'''

def normalize_second(raw:dict,expected:list[str])->dict:
    d={}
    for ob in raw.get('lead_observations',[]) if isinstance(raw.get('lead_observations'),list) else []:
        if isinstance(ob,dict) and ob.get('lead') in expected:
            d[ob['lead']]={k:str(ob.get(k,'UNKNOWN'))[:750] for k in ('visibility','observation','p','qrs','st_t')}
    return {'lead_observations':d,'conflicts_or_limits':str(raw.get('conflicts_or_limits',''))[:1500]}

def reconciliation(first:dict,second:dict,selected:list[str]) -> list[dict]:
    """Not an independent quantitative check. Two AI reads cannot verify ECG measurements."""
    rows=[]
    for lead in selected:
        a=first.get('lead_observations',{}).get(lead,{})
        b=second.get('lead_observations',{}).get(lead,{})
        if lead=='II-rhythm':a={'observation':first.get('rhythm_observation','UNKNOWN')}
        rows.append({'lead':lead,'first_read':str(a.get('observation','UNKNOWN')),
                     'crop_read':str(b.get('observation','NOT_REVIEWED')),
                     'verification_status':'AI_SECOND_VIEW_ONLY_NOT_INDEPENDENT_MEASUREMENT',
                     'clinical_review_required':True})
    return rows

def report_export(first,second,rows,boxes,layout,selected,img_hash):
    return {'schema_version':'2.1','layout_user_confirmed':layout,'roi_boxes_suggested':{k:list(v) for k,v in boxes.items()},
            'image_sha256_prefix':img_hash,'first_pass':first,'crop_pass':second,
            'selected_crops':selected,'reconciliation':rows,
            'quantitative_verification':{'status':'NOT_AVAILABLE','reason':'Digitizer/QRS/ST modules not validated for automatic confirmation'},
            'overall_status':'RESEARCH_REVIEW_REQUIRED'}
