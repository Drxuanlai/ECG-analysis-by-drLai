"""Offline structural checks; does not claim clinical accuracy or test API calls."""
from pathlib import Path
import ast,hashlib,py_compile
base=Path(__file__).resolve().parents[1]
for name in ('app.py','legacy/app_v17_1_original.py','core/contracts.py','core/measurement_guard.py'):
    p=base/name; ast.parse(p.read_text(encoding='utf-8')); py_compile.compile(str(p),doraise=True)
actual=hashlib.sha256((base/'legacy/app_v17_1_original.py').read_bytes()).hexdigest()
expected=(base/'legacy'/'SOURCE_SHA256.txt').read_text().strip()
assert actual == expected, 'Legacy source was modified!'
print('STRUCTURAL PASS: source untouched, Python files compile.')
print('RUNTIME NOT TESTED: requires Streamlit + API dependencies and representative ECG inputs.')
