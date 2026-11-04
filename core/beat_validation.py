"""ECG Core v1.5.1: research-only clustering and artifact triage.

Candidates are *not* verified beats. No RR regularization and no QRS/HR
measurement. The only definitive labels are entered by a human reviewer.
"""
from __future__ import annotations

from copy import deepcopy
from statistics import median
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

VERSION = 'ecg-core-v1.5.1-beat-validation-research'
REVIEW_LABELS = ('UNREVIEWED', 'QRS', 'ARTIFACT', 'UNCERTAIN')


def _image_profile(image: Image.Image):
    """Crude photograph-quality evidence; not a clinical morphology classifier."""
    im = image.convert('L')
    w, h = im.size
    im.thumbnail((900, 280), Image.Resampling.BILINEAR)
    a = np.asarray(im, dtype=np.float32)
    # dark threshold robust to images with bright/dark background; measured per row
    local = np.percentile(a, 40, axis=1, keepdims=True)
    ink = a < np.clip(local - 6.0, 35.0, 235.0)
    vertical = np.mean(ink, axis=0)
    horizontal = np.mean(ink, axis=1)
    return {'vertical': vertical, 'horizontal': horizontal, 'w': a.shape[1],
            'h': a.shape[0], 'original_width': w, 'original_height': h}


def cluster_validate(image: Image.Image, report: dict[str, Any], *,
                     cluster_window_fraction: float = .022,
                     crowd_window_fraction: float = .075,
                     crowd_min_candidates: int = 5) -> dict[str, Any]:
    """Group nearby shafts, flag unreliable regions, but never assert QRS.

    Distances are image-local morphology grouping windows, not inferred RR.
    In crowded regions, keep every original candidate in audit membership.
    """
    if not (.006 <= cluster_window_fraction <= .07):
        raise ValueError('Invalid grouping window')
    if not (.035 <= crowd_window_fraction <= .2):
        raise ValueError('Invalid crowding window')
    raw = sorted(deepcopy(report.get('candidates', [])), key=lambda c: float(c['x_pixel']))
    w, h = image.size
    cluster_distance = max(7.0, w * cluster_window_fraction)
    crowd_radius = max(25.0, w * crowd_window_fraction)
    profile = _image_profile(image)
    groups: list[list[dict]] = []
    for event in raw:
        if groups and float(event['x_pixel']) - float(groups[-1][-1]['x_pixel']) <= cluster_distance:
            groups[-1].append(event)
        else:
            groups.append([event])
    results = []
    for group in groups:
        positions = [float(p['x_pixel']) for p in group]
        scores = [max(0.0, float(p.get('score_heuristic', 0))) for p in group]
        center = float(np.average(positions, weights=np.maximum(.05, scores)))
        nearby = [p for p in raw if abs(float(p['x_pixel']) - center) <= crowd_radius]
        original_flags = {f for p in group for f in p.get('flags', [])}
        flag_codes = []
        if len(nearby) >= crowd_min_candidates:
            flag_codes.append('candidate_crowding')
        if any(f in original_flags for f in ('edge_touch', 'near_roi_edge')):
            flag_codes.append('edge_or_adjacent_lead_risk')
        if 'long_vertical_artifact_possible' in original_flags:
            flag_codes.append('vertical_artifact_risk')
        if len(group) >= 3:
            flag_codes.append('multiple_shafts_same_event')
        ix = int(round(center / max(w, 1) * (profile['w'] - 1)))
        left, right = max(0, ix - 3), min(profile['w'], ix + 4)
        col_occupancy = float(np.median(profile['vertical'][left:right])) if right > left else 0.
        if col_occupancy >= .48:
            flag_codes.append('extended_dark_column_risk')
        if center <= .025 * w or center >= .975 * w:
            flag_codes.append('outer_roi_edge')
        # Strong local crowding -> defer instead of dropping potentially real ectopy/VT.
        decision = 'UNRESOLVED' if flag_codes else 'CANDIDATE_REVIEW'
        best = max(group, key=lambda p: float(p.get('score_heuristic', 0)))
        results.append({
            'event_id': f'E{len(results) + 1:03d}',
            'x_pixel': round(center, 2),
            'source_candidate_ids': [p.get('id', '') for p in group],
            'source_x_pixels': [float(p['x_pixel']) for p in group],
            'source_candidate_count': len(group),
            'nearby_candidate_count': len(nearby),
            'shaft_top_pixel': best.get('shaft_top_pixel'),
            'shaft_bottom_pixel': best.get('shaft_bottom_pixel'),
            'max_source_heuristic': max(scores) if scores else 0.0,
            'column_occupancy': round(col_occupancy, 4),
            'machine_status': decision,
            'reason_codes': flag_codes,
            'review_label': 'UNREVIEWED',
        })
    by_label = {'CANDIDATE_REVIEW': sum(e['machine_status'] == 'CANDIDATE_REVIEW' for e in results),
                'UNRESOLVED': sum(e['machine_status'] == 'UNRESOLVED' for e in results)}
    return {
        'version': VERSION,
        'status': 'REVIEW',
        'lead': report.get('lead'),
        'image_name': report.get('image_name'),
        'roi_box': report.get('roi_box'),
        'source_version': report.get('version'),
        'source_candidate_count': len(raw),
        'event_count': len(results),
        'machine_status_counts': by_label,
        'events': results,
        'parameters': {'cluster_window_fraction': cluster_window_fraction,
                       'crowd_window_fraction': crowd_window_fraction,
                       'crowd_min_candidates': crowd_min_candidates},
        'notes': ['No regular RR assumption or rhythm diagnosis.',
                  'UNRESOLVED events are not rejected as nonbeats.',
                  'Machine grouping is not verification of a QRS.',
                  'For sensitivity/precision, reviewer must also mark missed QRS on original image.']
    }


def attach_review(report: dict[str, Any], labels: dict[str, str],
                  missed_x_pixels: list[float] | None = None) -> dict[str, Any]:
    """Attach explicit human labels; retain machine statuses for audit."""
    out = deepcopy(report)
    for event in out['events']:
        value = labels.get(event['event_id'], 'UNREVIEWED')
        if value not in REVIEW_LABELS:
            raise ValueError(f'Invalid human label: {value}')
        event['review_label'] = value
    missed = []
    for x in missed_x_pixels or []:
        xx = float(x)
        if not np.isfinite(xx) or xx < 0:
            raise ValueError('Invalid missed beat coordinate')
        missed.append(round(xx, 2))
    out['missed_qrs_manual_x_pixels'] = missed
    out['manual_review_completed'] = all(e['review_label'] != 'UNREVIEWED' for e in out['events'])
    out['human_confirmed_qrs_count'] = sum(e['review_label'] == 'QRS' for e in out['events'])
    out['reviewer_added_missed_qrs_count'] = len(missed)
    out['status'] = 'REVIEWED_BY_USER' if out['manual_review_completed'] else 'REVIEW'
    return out


def validation_overlay(image: Image.Image, report: dict[str, Any]) -> Image.Image:
    out = image.convert('RGB').copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for e in report['events']:
        x = int(round(e['x_pixel']))
        status = e.get('review_label')
        color = ('#30a46c' if status == 'QRS' else '#cf4141' if status == 'ARTIFACT'
                 else '#e98c20' if e['machine_status'] == 'UNRESOLVED' else '#168ad1')
        draw.line((x, 0, x, h - 1), fill=color, width=max(1, w // 950))
        draw.text((max(0, x - 10), 3), e['event_id'], fill=color)
    for x in report.get('missed_qrs_manual_x_pixels', []):
        draw.ellipse((x - 5, h - 13, x + 5, h - 3), outline='#9128ac', width=2)
    return out
