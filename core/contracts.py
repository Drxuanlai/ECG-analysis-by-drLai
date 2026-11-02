"""Pure data contracts for future engines; NOT wired to legacy analysis yet."""
from dataclasses import dataclass, field
from typing import Literal
QCStatus = Literal["PASS", "REVIEW", "FAIL", "UNRESOLVED"]
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
@dataclass(frozen=True)
class ECGCalibration:
    speed_mm_s: float | None = None
    limb_gain_mm_mv: float | None = None
    chest_gain_mm_mv: float | None = None
    verified: bool = False
@dataclass(frozen=True)
class BeatEvidence:
    x_original_px: float
    source: str
    status: QCStatus = "REVIEW"
    evidence: dict = field(default_factory=dict)
