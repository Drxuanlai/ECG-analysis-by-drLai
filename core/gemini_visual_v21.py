"""Minimal Gemini adapter. No API key persisted to disk; API call only on button click."""
from __future__ import annotations
from typing import Sequence


def analyze_images(api_key: str, model: str, prompt: str,
                   images: Sequence[tuple[str,bytes]], temperature:float=.1) -> str:
    if not api_key or not api_key.strip():
        raise ValueError('Chưa cấu hình GEMINI_API_KEY trong Streamlit Secrets.')
    if not model or not model.startswith('gemini-') or '/' in model:
        raise ValueError('Tên model Gemini không hợp lệ.')
    if not images or len(images)>9:
        raise ValueError('Cần 1–9 hình ảnh cho một lượt phân tích.')
    from google import genai
    from google.genai import types
    client=genai.Client(api_key=api_key)
    contents=[prompt]
    for title,raw in images:
        contents.append(f'Ảnh có nhãn ngữ cảnh: {title}. Chỉ kết luận lead nếu hình thực sự khớp.')
        contents.append(types.Part.from_bytes(data=raw,mime_type='image/jpeg'))
    response=client.models.generate_content(
        model=model,contents=contents,
        config=types.GenerateContentConfig(temperature=temperature,response_mime_type='application/json',
                                           max_output_tokens=11000))
    text=response.text
    if not text: raise RuntimeError('AI không trả nội dung; kiểm tra model/quota hoặc bộ lọc an toàn.')
    return text
