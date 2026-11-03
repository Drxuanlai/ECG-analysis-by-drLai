# ECG Core v1.3.2 - Lead Isolation QA

Research-only experimental feature. Source: ECG Core v1.3.1.

The historical ECG app lives unchanged under `legacy/app_v17_1_original.py`.
New module: `core/lead_isolation.py`; integration: `ui/layout_panel.py`.

Run `streamlit run app.py`.
Run `python -m pytest -q` for offline unit checks.

The v1.3.2 mask is NOT a validated ECG digitization, ECG rhythm detector,
or diagnostic measurement. All leads remain REVIEW by design. In overlapping
areas, uncertain pixels are withheld rather than attributed twice.

Three image smoke validations (image evidence only): see the separately
provided run report; no clinical accuracy is inferred.
