"""Coordinate conversion for zoomed, contiguous ECG ROI windows (v2.3.2)."""
from __future__ import annotations


def window_bounds(width: int, segment: int, window_px: int = 520) -> tuple[int, int]:
    if width <= 0 or window_px <= 0 or segment < 0:
        raise ValueError('Invalid ROI / window segment')
    x0 = segment * window_px
    x1 = min(width, x0 + window_px)
    if x0 >= x1:
        raise ValueError('Window outside ROI')
    return x0, x1


def display_to_roi(x: float, y: float, roi_width: int, roi_height: int,
                   segment: int, displayed_width: int, displayed_height: int,
                   window_px: int = 520) -> tuple[int,int]:
    x0,x1=window_bounds(roi_width,segment,window_px)
    if not (0 <= x < displayed_width and 0 <= y < displayed_height):
        raise ValueError('Click outside image')
    return (max(0, min(roi_width-1, x0 + round(x*(x1-x0)/displayed_width))),
            max(0, min(roi_height-1, round(y*roi_height/displayed_height))))
