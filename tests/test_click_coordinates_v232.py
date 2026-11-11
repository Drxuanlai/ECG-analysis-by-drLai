import pytest
from core.click_coordinates_v232 import display_to_roi,window_bounds


def test_contiguous_windows():
    assert [window_bounds(2453,i) for i in range(5)] == [(0,520),(520,1040),(1040,1560),(1560,2080),(2080,2453)]


def test_first_window_scaled_click():
    assert display_to_roi(200,100,2453,163,0,1040,326)==(100,50)


def test_middle_window_global_offset():
    assert display_to_roi(200,100,2453,163,2,1040,326)==(1140,50)


def test_last_window_nonstandard_width():
    assert display_to_roi(500,100,2453,163,4,746,326)==(2330,50)


def test_invalid_click_rejected():
    with pytest.raises(ValueError):
        display_to_roi(-1,10,2453,163,0,1040,326)
    with pytest.raises(ValueError):
        window_bounds(2453,5)
