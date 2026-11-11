"""Direct, layout-aware group ROI for ECG v2.3.1 research UI.

A group is ONE untouched rectangular crop from the original ECG photo.
The clinical interpretation remains unverified; the surgeon/clinician checks
labels and morphology. Per-lead ROI is a separate layer used by measurement QA.
"""
from __future__ import annotations
from dataclasses import dataclass
from PIL import Image
from core.group_crops_v22 import GROUPS, MANDATORY_GROUPS, validate_roi, panel_as_jpeg

@dataclass(frozen=True)
class DirectGroupPanel:
    group_id: str
    leads: tuple[str, ...]
    image: Image.Image
    group_box: tuple[int, int, int, int]
    roi_boxes: dict[str, tuple[int, int, int, int]]


def default_direct_boxes(size, layout, rhythm=False, rhythm_start=.78, margin=.02):
    """Return 4 or 5 direct rectangles, not 12 lead rectangles.

    Layout assumes the standard 3x4 or 6x2 print sequence, with a separate
    bottom rhythm strip only when explicitly confirmed. Position is heuristic.
    """
    w, h = size
    if layout not in ('3x4', '6x2'):
        raise ValueError('Unsupported ECG layout')
    if not (.65 <= rhythm_start <= .92) or not (0 <= margin <= .06):
        raise ValueError('Invalid crop geometry')
    ncols = 4 if layout == '3x4' else 2
    y_top = .075
    y_bottom = rhythm_start - .007 if rhythm else .96
    if y_bottom <= y_top + .2:
        raise ValueError('Too little space for 12 leads')
    outer_left, outer_right = .02, .98
    colwidth = (outer_right - outer_left) / ncols
    # A narrow horizontal margin avoids spilling into the next printed column.
    inset = margin / ncols
    out = {}
    for i, group in enumerate(MANDATORY_GROUPS):
        if layout == '3x4':
            x0 = outer_left + i * colwidth + inset
            x1 = outer_left + (i + 1) * colwidth - inset
            y0, y1 = y_top, y_bottom
        else:
            col = 0 if i < 2 else 1
            half = (y_bottom-y_top) / 2
            x0 = outer_left + col * colwidth + inset
            x1 = outer_left + (col + 1) * colwidth - inset
            y0 = y_top + (i % 2) * half
            y1 = y_top + (i % 2 + 1) * half
        out[group] = validate_roi((round(w*x0),round(h*y0),round(w*x1),round(h*y1)),size)
    if rhythm:
        out['RHYTHM_II'] = validate_roi((round(w*.02),round(h*(rhythm_start+.008)),round(w*.98),round(h*.975)),size)
    return out


def direct_panels(image, group_boxes, lead_boxes):
    """Preserve pixel content, labels, relative rhythm and calibration marks."""
    names = list(MANDATORY_GROUPS) + (['RHYTHM_II'] if 'RHYTHM_II' in group_boxes else [])
    panels = []
    for name in names:
        box = validate_roi(group_boxes[name], image.size)
        leads = GROUPS[name]
        if any(lead not in lead_boxes for lead in leads):
            raise ValueError(f'Missing lead ROI in {name}')
        roi = {lead:validate_roi(lead_boxes[lead],image.size) for lead in leads}
        panels.append(DirectGroupPanel(name, leads, image.crop(box).convert('RGB'),box,roi))
    return panels


def validate_direct_group_boxes(boxes, size, layout, rhythm):
    """Ensure expected group count and reject unintended overlap between groups."""
    required = set(MANDATORY_GROUPS) | ({'RHYTHM_II'} if rhythm else set())
    if set(boxes) != required:
        raise ValueError('Group list does not match declared layout/rhythm')
    valid = {k:validate_roi(v,size) for k,v in boxes.items()}
    for i,a in enumerate(valid):
        x0,y0,x1,y1 = valid[a]
        for b in list(valid)[i+1:]:
            u0,v0,u1,v1 = valid[b]
            dx = max(0,min(x1,u1)-max(x0,u0))
            dy = max(0,min(y1,v1)-max(y0,v0))
            intersection = dx*dy
            smaller = min((x1-x0)*(y1-y0),(u1-u0)*(v1-v0))
            if smaller and intersection/smaller > .08:
                raise ValueError(f'Group ROIs {a} and {b} overlap substantially')
    return valid
