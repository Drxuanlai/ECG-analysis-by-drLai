"""ECG Core 2.3: transparent, human-reviewed ECG pixel evidence and measurement.

No clinical decision or automatically verified landmark is produced here.
Pixel coordinates are always relative to the selected, unchanged lead ROI.
"""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
from typing import Any
from PIL import Image, ImageDraw

LANDMARKS = ('P_ONSET','P_PEAK','P_OFFSET','QRS_ONSET','Q_PEAK','R_PEAK',
             'S_NADIR','QRS_OFFSET','J_POINT','ST_POINT','T_PEAK','T_END','BASELINE')

@dataclass(frozen=True)
class GridCalibration:
    x0: float
    x5: float
    y0: float
    y5: float
    paper_speed_mm_s: float
    gain_mm_mv: float

    def __post_init__(self):
        if abs(self.x5-self.x0) < 8 or abs(self.y5-self.y0) < 8:
            raise ValueError('Hai đường lưới lớn cách nhau 5 mm phải khác nhau và đủ xa (>=8 px).')
        if not 8 <= abs(self.x5-self.x0) <= 300 or not 8 <= abs(self.y5-self.y0) <= 300:
            raise ValueError('Khoảng cách 5 mm phải trong 8..300 px; cần xác nhận lại ảnh/crop.')
        if self.paper_speed_mm_s not in (12.5, 25.0, 50.0, 100.0):
            raise ValueError('Tốc độ giấy chưa được hỗ trợ; không tự suy đoán.')
        if self.gain_mm_mv not in (5.0, 10.0, 20.0):
            raise ValueError('Gain chưa được hỗ trợ; không tự suy đoán.')

    @property
    def px_per_mm_x(self): return abs(self.x5-self.x0)/5.0
    @property
    def px_per_mm_y(self): return abs(self.y5-self.y0)/5.0
    @property
    def ms_per_px(self): return 1000.0/self.paper_speed_mm_s/self.px_per_mm_x
    @property
    def mv_per_px(self): return 1.0/self.gain_mm_mv/self.px_per_mm_y

    def to_dict(self):
        return {'x0':self.x0,'x5':self.x5,'y0':self.y0,'y5':self.y5,
                'paper_speed_mm_s':self.paper_speed_mm_s,'gain_mm_mv':self.gain_mm_mv,
                'px_per_mm_x':self.px_per_mm_x,'px_per_mm_y':self.px_per_mm_y,
                'ms_per_px':self.ms_per_px,'mv_per_px':self.mv_per_px,
                'verification':'MANUALLY_CONFIRMED_LOCAL_5MM_GRID'}


def paper_values(speed:float,gain:float)->dict:
    return {'small_box_ms':1000.0/speed,'large_box_ms':5000.0/speed,
            'small_box_mv':1.0/gain,'large_box_mv':5.0/gain}


def get_valid_points(points:list[dict], label:str, beat_id:str|None=None)->list[dict]:
    return [p for p in points if p.get('landmark_type')==label and p.get('source')=='HUMAN_CONFIRMED'
            and (beat_id is None or p.get('beat_id')==beat_id)]


def measure_points(points:list[dict], cal:GridCalibration)->list[dict]:
    """Compute only measurements supported by human-confirmed mappable landmarks."""
    events: list[dict] = []
    beat_ids=list(dict.fromkeys(p.get('beat_id') for p in points if p.get('source')=='HUMAN_CONFIRMED'))
    for beat in beat_ids:
        by_type={}
        for p in points:
            if p.get('beat_id')==beat and p.get('source')=='HUMAN_CONFIRMED':
                by_type.setdefault(p['landmark_type'],p)
        if 'QRS_ONSET' in by_type and 'QRS_OFFSET' in by_type:
            d=by_type['QRS_OFFSET']['x_pixel']-by_type['QRS_ONSET']['x_pixel']
            if d>0:
                events.append({'metric':'QRS_DURATION','beat_id':beat,'value':round(d*cal.ms_per_px,2),
                               'unit':'ms','point_ids':[by_type['QRS_ONSET']['id'],by_type['QRS_OFFSET']['id']],
                               'status':'HUMAN_LANDMARK_MEASUREMENT_REVIEW'})
        if 'R_PEAK' in by_type and 'BASELINE' in by_type:
            delta=by_type['BASELINE']['y_pixel']-by_type['R_PEAK']['y_pixel']
            events.append({'metric':'R_AMPLITUDE','beat_id':beat,'value':round(delta*cal.mv_per_px,4),
                           'unit':'mV','point_ids':[by_type['R_PEAK']['id'],by_type['BASELINE']['id']],
                           'status':'HUMAN_LANDMARK_MEASUREMENT_REVIEW'})
        if 'J_POINT' in by_type and 'BASELINE' in by_type:
            delta=by_type['BASELINE']['y_pixel']-by_type['J_POINT']['y_pixel']
            events.append({'metric':'ST_AT_J','beat_id':beat,'value':round(delta*cal.mv_per_px,4),
                           'unit':'mV','point_ids':[by_type['J_POINT']['id'],by_type['BASELINE']['id']],
                           'status':'HUMAN_LANDMARK_MEASUREMENT_REVIEW'})
    rpeaks=sorted(get_valid_points(points,'R_PEAK'),key=lambda p:p['x_pixel'])
    for p,q in zip(rpeaks,rpeaks[1:]):
        dx=q['x_pixel']-p['x_pixel']
        if dx>0:
            events.append({'metric':'RR_INTERVAL','beat_id':f"{p['beat_id']}->{q['beat_id']}",
                           'value':round(dx*cal.ms_per_px,2),'unit':'ms',
                           'point_ids':[p['id'],q['id']],
                           'status':'HUMAN_LANDMARK_MEASUREMENT_REVIEW'})
    return events


def overlay(roi:Image.Image,cal:GridCalibration|None,points:list[dict],
            proposals:list[dict]|None=None)->Image.Image:
    img=roi.convert('RGB').copy(); d=ImageDraw.Draw(img)
    if cal:
        sx=cal.px_per_mm_x; sy=cal.px_per_mm_y
        for step,anchor,dim,vertical in [(sx,cal.x0,img.width,True),(sy,cal.y0,img.height,False)]:
            start=math.floor((0-anchor)/step)
            end=math.ceil((dim-anchor)/step)
            for j in range(start,end+1):
                pos=round(anchor+j*step)
                if pos<0 or pos>=dim:continue
                color=(35,150,175) if j%5==0 else (100,195,204)
                if vertical:d.line([(pos,0),(pos,img.height)],fill=color,width=1)
                else:d.line([(0,pos),(img.width,pos)],fill=color,width=1)
    for p in proposals or []:
        if p.get('x_pixel') is None or p.get('y_pixel') is None:continue
        x=round(p['x_pixel']);y=round(p['y_pixel'])
        if 0<=x<img.width and 0<=y<img.height:
            d.ellipse((x-4,y-4,x+4,y+4),outline=(162,56,185),width=2)
    for p in points:
        x=round(p['x_pixel']);y=round(p['y_pixel'])
        if 0<=x<img.width and 0<=y<img.height:
            d.line((x-6,y,x+6,y),fill=(0,155,73),width=2)
            d.line((x,y-6,x,y+6),fill=(0,155,73),width=2)
            d.text((min(img.width-65,x+7),max(0,y-14)),p['landmark_type'][:9],fill=(0,105,50))
    return img


def sanitize_ai_landmarks(raw:Any,width:int,height:int)->list[dict]:
    """AI predictions remain unverified; malformed/out-of-bounds points are discarded."""
    if not isinstance(raw,dict):return []
    proposed=raw.get('landmarks',[])
    if not isinstance(proposed,list):return []
    output=[]
    for p in proposed[:80]:
        if not isinstance(p,dict):continue
        kind=str(p.get('landmark_type','')).upper()
        if kind not in LANDMARKS:continue
        x=p.get('x_pixel');y=p.get('y_pixel')
        if isinstance(x,bool) or isinstance(y,bool):continue
        if not isinstance(x,(int,float)) or not isinstance(y,(int,float)):continue
        if not all(map(math.isfinite,(x,y))) or not (0<=x<width and 0<=y<height):continue
        output.append({'landmark_type':kind,'beat_id':str(p.get('beat_id','UNKNOWN'))[:30],
                       'x_pixel':round(float(x),1),'y_pixel':round(float(y),1),
                       'evidence':str(p.get('evidence',''))[:300],
                       'source':'AI_UNVERIFIED','status':'REVIEW_REQUIRED'})
    return output


def validated_roi(box,photo_size):
    if len(box)!=4:raise ValueError('ROI cần 4 tọa độ.')
    x0,y0,x1,y1=map(int,box); w,h=photo_size
    if not (0<=x0<x1<=w and 0<=y0<y1<=h) or x1-x0<30 or y1-y0<30:
        raise ValueError('ROI không hợp lệ.')
    return x0,y0,x1,y1
