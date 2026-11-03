# ECG Core v1.3.3: Multi-Layout + Signal/Artifact Evidence

## Changes

- `core/multi_layout.py`: supports explicit 3×4, 6×2, experimental 12×1; unknown OCR evidence does not fall back silently to 3×4. For 6×2, mapping follows rows `I/V1`, `II/V2`, `III/V3`, `aVR/V4`, `aVL/V5`, `aVF/V6`. No rhythm row is invented.
- `core/signal_artifact.py`: distinct candidate-ECG-ink, confident colored-grid, likely text/calibration and uncertain intersection maps. **Not waveform extraction.**
- `ui/multi_layout_panel.py`: new dedicated QA panel with clinician-confirmed layout, shared row/column sliders, optional per-lead adjustment, ZIP validation export.
- Historical 3×4 QA panel marked LEGACY. Legacy AI, Gemini failover and `ecg_digitizer.py` unchanged.

## Limitations

- Automatic OCR layout discovery returns `UNKNOWN` on test images with insufficient labels; select 3×4 or 6×2 manually based on the original image.
- Row/column positions are initial proposals; especially on photographs, visually adjust them.
- The mask classifiers cannot guarantee perfect separation of gray grids or printing from a low-contrast ECG. All results remain REVIEW.
- No QRS detection or clinical measurements are introduced.

## Files and deployment

Unzip to the root of the **refactor/ecg-core-v1** branch, not over the production `main` branch. Entry point stays `app.py`. Preserve existing Streamlit Secrets.

Run `python -m pytest -q` or `streamlit run app.py`. Open the expander named **ECG Core v1.3.3 | Multi-Layout + Signal/Artifact QA**. Use `MULTILAYOUT_V1_3_3_GUIDE_VI.md` for instructions.
