"""Guard for quantitative evidence. No ECG interpretation is performed here."""
from .contracts import ECGCalibration

def conversion_allowed(cal: ECGCalibration, *, lead: str) -> bool:
    if not cal.verified or cal.speed_mm_s is None or cal.speed_mm_s <= 0:
        return False
    gain = cal.chest_gain_mm_mv if lead.startswith('V') else cal.limb_gain_mm_mv
    return gain is not None and gain > 0
