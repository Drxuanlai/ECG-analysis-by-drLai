# ECG Core v2.3 – Grid Calibration & Visual Landmark Evidence

**Research QA only.** The AI model is not validated as a clinical ECG measuring device. Do not use this tool alone to exclude dangerous disease or guide treatment.

## Installation (patch only)

Upload these files to `refactor/ecg-core-v1`, retaining their folder paths:

1. Replace `visual_reader_v21_app.py` (Streamlit's Main file path **stays unchanged**).
2. Add `core/grid_evidence_v23.py`.
3. Add `ui/grid_measurement_panel_v23.py`.
4. Add `tests/test_grid_evidence_v23.py`.
5. Add this README.
6. Follow `requirements_v23_additions.txt`: append `streamlit-image-coordinates>=0.1.9,<1` to the EXISTING root `requirements.txt` if absent. **Do not replace the root requirements file.**

This patch does not modify `app.py`, Gemini routes, `core/group_crops_v22.py`, or the benchmark app. If your repo has no `ui/__init__.py`, ensure the `ui/` directory is a usable Python package (namespace packages also work).

## Workflow

1. Upload one ECG, run first-pass AI, confirm 3x4 or 6x2 and the crop boxes. The existing 4/5 group workflow remains unchanged.
2. Open **4 · v2.3 Grid Map** after the group preview. Choose a single lead (II-rhythm preferred for rhythm measurements).
3. Verify speed (25 or 50 mm/s, etc.) and amplitude gain (5/10/20 mm/mV) from text/calibration pulse, **not only AI statements**.
4. Inspect the lead ROI and identify two **adjacent bold vertical gridlines** exactly 5 mm apart; input their X coordinates (relative to ROI). Likewise two adjacent bold **horizontal gridlines** 5 mm apart and their Y coordinates. Default numbers are placeholders and MUST NOT be accepted without inspection. If the grid is obscured or distorted, don't confirm.
5. Mark “Tôi đã kiểm tra...” only after validation. The app overlays a proposed fine grid. It does not automatically detect every local grid deformation; paper fold/tilt may invalidate calibration away from these anchors.
6. Optionally request **AI landmark suggestions** for that lead. They stay purple and `AI_UNVERIFIED`, are NEVER used directly for numerical measurements.
7. Click on the ECG image and select a landmark type plus Beat ID; save manually verified green landmarks. `streamlit-image-coordinates` converts display-pixel clicks to source ROI pixel coordinates. If the component is unavailable, enter X/Y directly.
8. The Python layer calculates only from pairs of **manual human landmarks**, with locally validated grid: RR interval (two consecutive R peaks), QRS duration (onset + offset), R amplitude (baseline + R peak), and ST at J (baseline + J point). If a required endpoint is missing, that measurement is unavailable.
9. Download ZIP `ecg_core_v2_3_grid_evidence_<lead>.zip`, which includes JSON, calibration evidence and human/AI landmarks **but no original ECG image or API secrets**.

## Coordinate conventions and measurement equations

- For a crop with bounding box `(x0,y0,x1,y1)`, local ROI `(x,y)` corresponds to original image `(x+x0, y+y0)`.
- Pixel X increases rightwards; pixel Y increases downwards.
- Two adjacent major gridlines are 5 mm apart. `px_per_mm_x=abs(X2-X1)/5`, and same for Y.
- `duration_ms = abs(dx_px)/px_per_mm_x * 1000 / paper_speed_mm_s`.
- `amplitude_mv = (baseline_y - peak_y)/px_per_mm_y / gain_mm_mv`.
- RR must be based on actual consecutive verified R peaks, not regularity assumptions; QRS must be onset-to-offset, not peak-to-peak.
- ST at J is a numerical *voltage comparison*, NOT a stand-alone STEMI or Brugada diagnosis. The selected reference baseline must be clinically and morphologically appropriate.
- Numbers should be interpreted with pixel-click uncertainty, paper distortion, calibration uncertainty and physiological context; UI does not display validated error bars.

## Limitations / honesty

The v2.3 grid is a **local user-calibrated overlay**, not automatic verified red-grid extraction or paper dewarping. It does not automatically find R, QRS onset, J point, ST, QT or heart rate. AI suggests coordinates only; human review is required. Group-crop quality and the legacy algorithm's limitations remain. Do not use results clinically without independent confirmation.

## Test

```bash
python -m unittest discover -s tests -p 'test_grid_evidence_v23.py' -v
```

Before accepting this implementation, test on images with: 25/50 mm/s, 5/10 mm/mV, black-and-white and red grid, 3x4/6x2, paper folds, deep S in V1, irregular rhythms, and blinded reference measurements. Verify component click coordinate mapping in your deployed Streamlit app.
