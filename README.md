# ECG Core v1.3.1 — Lead-Aware ROI Refinement

**Giữ nguyên AI/failover legacy, QA riêng.**

Xem `ROI_REFINEMENT_V1_3_1_GUIDE_VI.md` để dùng hai lớp ROI và kiểm thử.

# ECG Core v1.0 — compatibility scaffold

**Status: structural refactor stage, not validated clinical software.**

## How to deploy
1. Create Git branch `refactor/ecg-core-v1` in your GitHub repository (leave main unchanged).
2. Upload all folders and files from this ZIP into that branch root. Preserve case and paths.
3. Streamlit Cloud: repository branch `refactor/ecg-core-v1`, entrypoint `app.py`.
4. Keep Streamlit Secrets configured **in Streamlit Cloud**, not in this repo.
5. Compare baseline app outputs on the same ECG image. Only merge into `main` after parity testing.

## Compatibility design
`app.py` delegates to the exact uploaded source `legacy/app_v17_1_original.py` via `runpy.run_path`. This deliberately retains all original UI/session-state logic while separating structure. It is **not** a completed algorithm refactor. `core/contracts.py` and `core/measurement_guard.py` are pure, testable starter modules, currently **not wired** into the legacy app.

## Known dependency gap
Original app optionally imports `ecg_digitizer` from `ecg_digitizer.py`. That source **was not included** in the two user uploads. The app's existing fallback will report it unavailable; digitizer-dependent features may not run. Add the **original compatible** `ecg_digitizer.py` to the project root if it is used in your current GitHub deployment. Do not create a dummy replacement.

## Tests
`python scripts/verify.py` — no installed app packages required.
`python -m pytest -q` — requires pytest (development dependency only).
A Streamlit browser smoke test and fixed-ECG regression set are required before calling this deploy ready.

## Next steps (separate, guarded commits)
1. Restore missing `ecg_digitizer.py` dependency and pin/test runtime dependencies.
2. Build baseline regression fixtures (images + exported JSON; de-identify patients).
3. Extract API routing/provider code to `ai/` with route/failover tests.
4. Extract orientation, OCR, geometry to `core/` with pixel-coordinate regression tests.
5. Extract v17 segmentation into `core/` with image-based quality metrics.
6. Add irregular rhythm + broad QRS benchmarks, not regular-RR filtering.

**Security:** never commit API keys, patient-identifiable photos, or secret files.


## v1.3 Adaptive Layout QA
Run `app.py`; expand **ECG Core v1.3 | Adaptive Layout & Lead ROI QA**. See `LAYOUT_V1_3_GUIDE_VI.md`. OCR and calibration-like stroke boundaries are unverified candidates; manual row/column edits are supported. Legacy AI/failover preserved.
