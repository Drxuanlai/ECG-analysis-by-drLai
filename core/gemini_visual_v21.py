"""Gemini visual adapter with Streamlit ECG_API_ROUTES compatibility.

Secrets and key values must never be logged or written to exports.
"""
from __future__ import annotations
from collections.abc import Mapping, Sequence


def routes_from_secrets(secrets) -> list[dict]:
    """Read ECG_API_ROUTES TOML array or a single conventional Gemini key."""
    routes = []
    try:
        configured = secrets.get('ECG_API_ROUTES', []) or []
    except Exception:
        configured = []
    for i, entry in enumerate(configured):
        if not hasattr(entry, 'get'):
            continue
        provider = str(entry.get('provider', 'Google Gemini'))
        key = str(entry.get('api_key', '') or '').strip()
        model = str(entry.get('model', '') or '').strip()
        if 'gemini' not in provider.lower() or not key or not model:
            continue
        routes.append({'name': str(entry.get('name', f'Route {i+1}')),
                       'model': model, 'api_key': key})
    if routes:
        return routes
    try:
        key = secrets.get('GEMINI_API_KEY') or secrets.get('GOOGLE_API_KEY') or ''
        model = secrets.get('GEMINI_MODEL', 'gemini-2.5-flash')
    except Exception:
        key, model = '', 'gemini-2.5-flash'
    if str(key).strip():
        routes.append({'name': 'Gemini mặc định', 'api_key': str(key).strip(), 'model': str(model)})
    return routes


def analyze_images(api_key: str, model: str, prompt: str,
                   images: Sequence[tuple[str, bytes]], temperature: float = .1) -> str:
    if not api_key or not api_key.strip():
        raise ValueError('Chưa có API key Gemini phù hợp trong Streamlit Secrets.')
    if not model or not model.startswith('gemini-') or '/' in model:
        raise ValueError('Tên model Gemini không hợp lệ trong cấu hình.')
    if not images or len(images) > 9:
        raise ValueError('Cần 1–9 hình ảnh cho một lượt phân tích.')
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)
    contents = [prompt]
    for title, raw in images:
        contents.append(f'Ảnh có nhãn ngữ cảnh: {title}. Chỉ kết luận lead nếu hình thực sự khớp.')
        contents.append(types.Part.from_bytes(data=raw, mime_type='image/jpeg'))
    response = client.models.generate_content(
        model=model, contents=contents,
        config=types.GenerateContentConfig(temperature=temperature,
                                           response_mime_type='application/json',
                                           max_output_tokens=11000))
    result = response.text
    if not result:
        raise RuntimeError('AI không trả nội dung; kiểm tra model/quota hoặc bộ lọc an toàn.')
    return result


def _safe_error_name(exc: Exception) -> str:
    """Never display str(exc): remote error may echo sensitive request metadata."""
    return type(exc).__name__


def analyze_with_routes(routes: Sequence[Mapping], prompt: str,
                        images: Sequence[tuple[str, bytes]], temperature: float = .1,
                        call=analyze_images) -> tuple[str, str]:
    """Use configured priority order. Try next route on failed call.

    Returns (response, public route label). Do not persist routes or keys.
    """
    if not routes:
        raise RuntimeError('Chưa tìm thấy ECG_API_ROUTES hoặc GEMINI_API_KEY trong Secrets của app này.')
    errors = []
    for i, route in enumerate(routes):
        model = str(route.get('model', ''))
        key = str(route.get('api_key', ''))
        try:
            result = call(key, model, prompt, images, temperature)
            return result, f'Route {i + 1} ({model})'
        except Exception as exc:
            errors.append(f'{i+1}:{_safe_error_name(exc)}')
    raise RuntimeError('Tất cả route Gemini đều thất bại (' + ', '.join(errors) + '). Xem Logs để kiểm tra quota, model và quyền API; không đăng key.')
