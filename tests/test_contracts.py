from core.contracts import ECGCalibration, LEADS
from core.measurement_guard import conversion_allowed

def test_twelve_leads():
    assert len(LEADS) == 12 and len(set(LEADS)) == 12

def test_unverified_calibration_cannot_convert():
    assert not conversion_allowed(ECGCalibration(speed_mm_s=25, limb_gain_mm_mv=10), lead='II')

def test_mixed_gain_requires_own_group():
    cal = ECGCalibration(speed_mm_s=25, limb_gain_mm_mv=10, verified=True)
    assert conversion_allowed(cal, lead='II')
    assert not conversion_allowed(cal, lead='V2')
