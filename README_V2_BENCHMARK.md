# ECG Core v2.0 — Benchmark & Independent QRS Detector (Research)

## Purpose
Separate algorithm development from clinical validation. This is a research tool, NOT a clinical ECG interpretation or verified QRS detector. The v1.x code, Gemini, API failover and `app.py` remain untouched.

## Installation (PATCH ONLY)
Copy `benchmark_v2_app.py`, `core/benchmark_v2.py`, `core/qrs_detector_v2.py`, and `tests/test_benchmark_v2.py` to the corresponding paths in the `refactor/ecg-core-v1` branch. Add `pandas>=2,<3`, `scipy>=1.11,<2` if absent in `requirements.txt` (both are common dependencies but require verification). Existing `numpy`, `opencv-python-headless`, `Pillow`, `streamlit` are used.

Deploy a **separate Streamlit app** with entrypoint `benchmark_v2_app.py` (same repository and branch), retaining the existing production and test apps intact. It can also be run locally via `streamlit run benchmark_v2_app.py`.

## Workflow
1. Upload one legacy ZIP validation, including `roi_original_v15.png` and `rpeak_reference_v153.json`.
2. Review imported human reference marks against original ROI. Labels are **UNVERIFIED** on import.
3. Explicitly assign TRAIN/VALIDATION/TEST. Prior optimization cases must **NOT** be marked TEST; use held-out images for honest performance measurement.
4. Run independent QRS detector; compare orange timing candidates to green human fiducials.
5. Export a self-contained ZIP containing image, reference, detector prediction, and QA report.

## Metric gates
Sensitivity and precision remain `null` unless all_windows_reviewed, reference_complete, review_status=VERIFIED, dataset_split=TEST and development_exposed=False. A click/checkbox is a human attestation, not an automatic proof of complete labels. Metrics are **per ROI/lead**, not evidence of clinical performance. Scoring uses one-to-one matching so duplicate candidates cannot count as two true positives.

The detector predicts QRS **event X only**, NOT R apex/S nadir Y. Do not interpret X matching as 2D fidelity. A separate 2D matching helper is provided for future localization tasks. Pixel tolerance must not be converted into milliseconds without confirmed paper/grid calibration.

## Roadmap gates
- A: Import and validate annotations, enforce completeness and independent split.
- B: Independent image-only QRS baseline; evaluate on hidden TEST after tuning frozen.
- C: Compare to frozen v1.5.9 on identical ROI and independently annotated QRS timing. Do not claim a win from unverified labels or tuned cases.

## Privacy
Do not commit patient-identifiable ECG scans or related metadata to a public GitHub repo. Upload validation ZIP to private QA session; locally back up ground truth securely.
