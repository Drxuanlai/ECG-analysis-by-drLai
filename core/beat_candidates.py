"""ECG Core v1.5 -- candidate-only, beat-by-beat event proposals.

No regular RR assumption, no clinical verdict, no calibrated units. A dark
vertical trace or local morphological change is NOT proof of a QRS complex.
The output MUST be reviewed against the original ECG.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from core.waveform_reconstruction import _prep

VERSION = 'ecg-core-v1.5-candidate-only'


def _runs(a: np.ndarray):
    if a.size == 0:
        return []
    cuts = np.flatnonzero(np.diff(a) > 1) + 1
    return np.split(a, cuts)


def _residual_ink(image: Image.Image):
    im, rgb, dark, red, enhanced, original = _prep(image, max_width=1400, max_height=390)
    h, w = dark.shape
    # Local neutral darkness. Grid lines remain a competing hypothesis,
    # rather than being deleted at their intersections with an ECG stroke.
    threshold = max(7.0, float(np.percentile(dark, 77)))
    ink = (dark >= threshold) & (red < .66)
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE,
                           np.ones((3, 1), np.uint8)).astype(bool)
    # Full-height calibration strokes / paper boundaries receive penalties.
    col_fraction = ink.mean(axis=0)
    return im, ink, dark, col_fraction, original


def detect_candidates(image: Image.Image, *, sensitivity: float = 0.55,
                      exclude_margin_fraction: float = .028):
    """Rank morphological event candidates; do not infer that they are beats.

    Each event is sourced from image ink. The RR differences are descriptive
    and NEVER used to reject a candidate or fit a periodic clock.
    """
    if not .05 <= sensitivity <= .95:
        raise ValueError('sensitivity must be 0.05..0.95')
    im, ink, darkness, col_fraction, (ow, oh) = _residual_ink(image)
    h, w = ink.shape
    if h < 24 or w < 60:
        raise ValueError('Lead ROI is too small')

    min_run = max(7, int(round(h * .09)))
    max_run = max(min_run + 3, int(round(h * .79)))
    runscore = np.zeros(w, dtype=np.float32)
    miny = np.full(w, -1, dtype=np.int32)
    maxy = np.full(w, -1, dtype=np.int32)
    for x in range(w):
        candidates = [s for s in _runs(np.flatnonzero(ink[:, x]))
                      if min_run <= len(s) <= max_run]
        if candidates:
            s = max(candidates, key=len)
            runscore[x] = len(s) / float(h)
            miny[x], maxy[x] = int(s[0]), int(s[-1])

    # A very tall but also consistently dark column may be paper grid or a
    # calibration pulse. It stays a candidate with a flag, not a verified beat.
    k = max(7, (int(round(w / 45)) | 1))
    background = cv2.GaussianBlur(runscore.reshape(1, -1), (k, 1), 0).ravel()
    novelty = np.maximum(0, runscore - .55 * background)
    novelty = cv2.GaussianBlur(novelty.reshape(1, -1), (5, 1), 0).ravel()
    # Internal, scale-free selection threshold. Tunable for research QA only.
    positive = novelty[novelty > 0]
    if len(positive) == 0:
        limit = float('inf')
    else:
        limit = max(float(np.quantile(positive, .57 + .31 * (1 - sensitivity))),
                    .035 + .036 * (1 - sensitivity))
    margin = max(5, int(round(w * exclude_margin_fraction)))
    local_gap = max(7, round(w * .007))  # morphology deduplication, NOT RR period
    maxima = []
    for x in range(margin, w - margin):
        if novelty[x] < limit or runscore[x] <= 0:
            continue
        a, b = max(0, x - local_gap), min(w, x + local_gap + 1)
        if novelty[x] >= max(novelty[a:b]) and x == a + int(np.argmax(novelty[a:b])):
            maxima.append(x)
    # Merge duplicate detections within a narrow ink shaft, not by RR clock.
    kept = []
    for x in sorted(maxima, key=lambda v: float(novelty[v]), reverse=True):
        if all(abs(x - t) > max(5, round(w * .009)) for t in kept):
            kept.append(x)
    kept.sort()
    events = []
    norm = max(.05, float(np.quantile(novelty, .95)))
    for x in kept:
        y0, y1 = int(miny[x]), int(maxy[x])
        if y0 < 0:
            continue
        flags = []
        if y0 <= 2 or y1 >= h - 3:
            flags.append('edge_touch')
        if col_fraction[x] > .75:
            flags.append('long_vertical_artifact_possible')
        # Grid periodicity cannot be safely excluded from a single ROI;
        # every event remains UNVERIFIED regardless of its numeric score.
        if (x < margin * 2 or x > w - 2 * margin):
            flags.append('near_roi_edge')
        score = float(min(.95, .55 * novelty[x] / norm + .27 * runscore[x] / .35))
        if flags:
            score = min(score, .59)
        events.append({
            'id': f'C{len(events) + 1:03d}',
            'x_pixel': round((x + .5) * ow / w, 2),
            'x_work': int(x),
            'shaft_top_pixel': round(y0 * oh / h, 2),
            'shaft_bottom_pixel': round(y1 * oh / h, 2),
            'vertical_ink_px': round((y1 - y0 + 1) * oh / h, 2),
            'score_heuristic': round(score, 3),
            'status': 'CANDIDATE_REVIEW',
            'flags': flags + ['QRS_not_verified'],
        })
    xs = [e['x_pixel'] for e in events]
    rr = [round(xs[i] - xs[i - 1], 2) for i in range(1, len(xs))]
    return {
        'version': VERSION,
        'status': 'REVIEW',
        'candidates': events,
        'inter_candidate_deltas_px': rr,
        'metrics': {
            'candidate_count': len(events),
            'roi_width_px': ow,
            'roi_height_px': oh,
            'internal_image_width_px': w,
            'no_rr_regularization': True,
            'calibrated_ms_or_mV': False,
            'clinical_qrs_verified': 0,
        },
        'reason_codes': ['ink_morphology_not_clinically_validated',
                         'grid_and_calibration_may_create_false_candidates',
                         'no_ground_truth_label_comparison'],
        '_debug': {'preview': im, 'ink': ink, 'novelty': novelty},
    }


def candidate_overlay(image: Image.Image, report: dict):
    out = image.convert('RGB').copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    for event in report['candidates']:
        x = int(round(event['x_pixel']))
        draw.line((x, 0, x, h - 1), fill='#fa8e28', width=max(1, w // 650))
        draw.rectangle((x - 4, 2, x + 4, 10), fill='#c44b2c')
    return out


def evidence_panel(image: Image.Image, report: dict):
    original = image.convert('RGB')
    w, h = original.size
    ink = report['_debug']['ink']
    map_img = Image.fromarray((ink.astype(np.uint8) * 255), mode='L').convert('RGB')
    map_img = map_img.resize(original.size, Image.Resampling.NEAREST)
    overlay = candidate_overlay(image, report)
    canvas = Image.new('RGB', (w * 2, h), 'white')
    canvas.paste(original, (0, 0))
    canvas.paste(overlay, (w, 0))
    # Ink map is available separately to avoid suggesting it is a clean signal.
    return canvas, map_img


def public_report(report: dict):
    return {k: v for k, v in report.items() if k != '_debug'}
