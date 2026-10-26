import streamlit as st
from PIL import Image, ImageOps
import cv2
import numpy as np
from google import genai
from google.genai import types
from openai import OpenAI
import io
import base64
import time

# Hỗ trợ ảnh chụp iPhone (HEIC/HEIF)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

st.set_page_config(
    page_title="AI 12-Lead ECG Expert",
    page_icon="🫀",
    layout="wide"
)

# ----------------- BỘ TIỀN XỬ LÝ ẢNH (OPENCV) -----------------
def preprocess_ecg(pil_img, remove_shadow=True, apply_clahe=True, clahe_clip=2.0, sharpen_level=1.0, rotation_angle=0):
    img = np.array(pil_img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if rotation_angle != 0:
        (h, w) = img.shape[:2]
        M = cv2.getRotationMatrix2D((w // 2, h // 2), rotation_angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if remove_shadow:
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated, 21)
        diff_img = 255 - cv2.absdiff(gray, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        processed = norm_img
    else:
        processed = gray

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

    if sharpen_level > 0:
        gaussian = cv2.GaussianBlur(processed, (0, 0), 2.0)
        sharpened = cv2.addWeighted(processed, 1.0 + sharpen_level, gaussian, -sharpen_level, 0)
        processed = np.clip(sharpened, 0, 255).astype(np.uint8)

    return Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB))

# ----------------- PROMPT LÂM SÀNG -----------------
SYSTEM_PROMPT = """
Bạn là một chuyên gia Tim mạch can thiệp và Điện sinh lý học tim.
Hãy đọc và phân tích hình ảnh Điện tâm đồ 12 chuyển đạo tiêu chuẩn theo trình tự:

1. ĐÁNH GIÁ KỸ THUẬT: Test biên độ (mm/mV), Tốc độ kéo giấy (mm/s), Độ nhiễu, Mắc ngược điện cực (aVR).
2. TẦN SỐ & NHỊP: Tần số (ck/p), Nhịp xoang hay rối loạn nhịp (Rung nhĩ, cuồng nhĩ, ngoại tâm thu...).
3. TRỤC ĐIỆN TIM: Bình thường, lệch trái, lệch phải hay vô định.
4. HÌNH THÁI SÓNG & KHOẢNG:
   - Sóng P (dày nhĩ).
   - Khoảng PR (dài/ngắn, block nhĩ thất).
   - Phức bộ QRS (thời gian hẹp/rộng, biên độ Sokolow-Lyon/Cornell, sóng Q hoại tử).
   - Đoạn ST (đẳng điện, chênh lên, chênh xuống, vị trí thành cơ tim tổn thương, hình ảnh soi gương).
   - Sóng T và khoảng QTc.
5. KẾT LUẬN & GỢI Ý XỬ TRÍ:
   - Chẩn đoán ECG chính.
   - Mức độ khẩn cấp (TỐI KHẨN CẤP / KHẨN CẤP / THƯỜNG QUY).
   - Đề xuất cận lâm sàng kết hợp (Troponin hs, Siêu âm tim, Khí máu...).

Không bịa số đo khi lưới, nhãn chuyển đạo hoặc chuẩn biên độ/tốc độ không đọc rõ; ghi rõ không xác định được.
Trả lời ngắn gọn, tránh nhắc lại.
Lưu ý: Bắt đầu kết luận với lưu ý đây là công cụ AI hỗ trợ, quyết định lâm sàng cuối cùng thuộc về bác sĩ điều trị.
"""

# Timeout Google: milliseconds. One request, no nested automatic retries.
REQUEST_TIMEOUT_SECONDS = 45
STREAM_BUDGET_SECONDS = 90
MAX_IMAGE_EDGE = 2400


def gemini_config(model_name, thinking_level="medium"):
    options = {"max_output_tokens": 8192}
    # Only apply controls documented for this exact model.
    if model_name.removeprefix("models/") == "gemini-3.8-flash":
        options["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    return types.GenerateContentConfig(**options)


def probe_gemini(api_key, model_name, image=None):
    """Small inference on the same host, key and model as the ECG request."""
    started = time.monotonic()
    config = gemini_config(model_name, "low")
    config.max_output_tokens = 2048
    contents = "Reply with only OK."
    if image is not None:
        # Same image and encoding as analysis; short output isolates input handling.
        contents = [image, "Describe the image in one short sentence. Do not diagnose."]
    try:
        with genai.Client(
            api_key=api_key.strip(),
            http_options=types.HttpOptions(
                timeout=20000, retry_options=types.HttpRetryOptions(attempts=1)
            ),
        ) as client:
            result = client.models.generate_content(
                model=model_name, contents=contents, config=config
            )
            answer = result.text or ""
            if not answer.strip():
                raise RuntimeError("API không trả văn bản; kiểm tra giới hạn token hoặc bộ lọc.")
        return {"Model": model_name, "Bước": "Ảnh + yêu cầu ngắn" if image else "Văn bản ngắn",
                "Kết quả": "Thành công", "Giây": round(time.monotonic()-started, 1),
                "Chi tiết": answer}
    except Exception as error:
        return {"Model": model_name, "Bước": "Ảnh + yêu cầu ngắn" if image else "Văn bản ngắn",
                "Kết quả": "Thất bại", "Giây": round(time.monotonic()-started, 1),
                "Chi tiết": str(error)}


def analyze_with_gemini_stream(api_key, model_name, image, prompt, thinking_level="medium"):
    with genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_SECONDS * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) as client:
        img_to_send = image.convert("RGB")
        img_to_send.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        response = client.models.generate_content_stream(
            model=model_name,
            contents=[img_to_send, prompt],
            config=gemini_config(model_name, thinking_level),
        )
        try:
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                for candidate in chunk.candidates or []:
                    reason = getattr(candidate.finish_reason, "value", candidate.finish_reason)
                    if reason and reason != "STOP":
                        raise RuntimeError(
                            f"Báo cáo chưa hoàn tất (finish_reason={reason}). "
                            "Không sử dụng phần nội dung đang hiển thị như một kết luận đầy đủ."
                        )
        finally:
            response.close()


def friendly_api_error(error):
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    message = str(error).lower()
    if str(code) == "503" or "503" in message or "unavailable" in message:
        return "Google/model đang quá tải (503). Hãy thử lại sau hoặc chủ động chọn nhà cung cấp khác."
    if str(code) == "404" or "404" in message:
        return "Không tìm thấy model hoặc tài khoản không được truy cập (404). Kiểm tra tên model và quyền truy cập."
    if str(code) == "429" or "429" in message:
        return "Đã chạm hạn mức API/tần suất yêu cầu (429). Kiểm tra quota của dự án."
    if str(code) in ("401", "403"):
        return "API key, quyền truy cập hoặc điều kiện thanh toán không hợp lệ (401/403)."
    if isinstance(error, TimeoutError) or "timeout" in message or "timed out" in message:
        return "Yêu cầu chờ quá lâu. Hãy thử lại sau hoặc chủ động chọn model khác."
    return "Yêu cầu phân tích không hoàn tất. Xem chi tiết kỹ thuật bên dưới."


def display_analysis(generator):
    started = time.monotonic()
    first_text_seconds = None
    status = st.empty()
    status.caption("Đang chờ API trả nội dung; không tự gửi lại yêu cầu khi thất bại.")

    def timed_chunks():
        nonlocal first_text_seconds
        try:
            for text in generator:
                elapsed = time.monotonic() - started
                # Cooperative budget: a blocked network read is bounded separately
                # by the transport timeout; this is not a hard wall-clock deadline.
                if elapsed > STREAM_BUDGET_SECONDS:
                    raise TimeoutError("Đã vượt ngân sách thời gian stream.")
                if text:
                    if first_text_seconds is None:
                        first_text_seconds = elapsed
                    status.caption(f"Đang nhận báo cáo · {elapsed:.1f} giây")
                    yield text
            if first_text_seconds is None:
                raise RuntimeError("API kết thúc nhưng không trả nội dung báo cáo.")
        finally:
            generator.close()

    try:
        st.write_stream(timed_chunks())
        status.caption(
            f"Hoàn tất: {time.monotonic() - started:.1f} giây · "
            f"Có chữ đầu tiên sau: {first_text_seconds:.1f} giây"
        )
    except Exception as error:
        status.caption(f"Dừng sau {time.monotonic() - started:.1f} giây")
        st.error(friendly_api_error(error))
        if first_text_seconds is not None:
            st.warning("Báo cáo bị gián đoạn, nội dung phía trên chưa phải báo cáo hoàn chỉnh.")
        with st.expander("Chi tiết kỹ thuật"):
            st.code(str(error))


def encode_image_to_base64(pil_image):
    buffered = io.BytesIO()
    rgb_image = pil_image.convert('RGB')
    if max(rgb_image.size) > MAX_IMAGE_EDGE:
        rgb_image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    rgb_image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_with_openai_stream(api_key, model_name, base64_image, prompt):
    client = OpenAI(api_key=api_key.strip(), timeout=REQUEST_TIMEOUT_SECONDS, max_retries=0)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            max_tokens=2500,
            temperature=0.2,
            stream=True
        )
        try:
            for chunk in response:
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta.content:
                        yield choice.delta.content
                    if choice.finish_reason and choice.finish_reason != "stop":
                        raise RuntimeError(f"Báo cáo chưa hoàn tất ({choice.finish_reason}).")
        finally:
            response.close()
    finally:
        client.close()

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


# ----------------- GIAO DIỆN CHÍNH -----------------
st.title("🫀 Phân Tích Điện Tâm Đồ 12 Chuyển Đạo")
st.caption("Bản kiểm tra API v2 · AI hỗ trợ đọc ECG 12 chuyển đạo")

with st.sidebar:
    st.header("⚙️ Lựa chọn Mô hình AI")
    provider = st.radio(
        "Chọn chế độ phân tích:",
        ("Google Gemini (Theo quota tài khoản)", "OpenAI GPT-4o (Yêu cầu Key cá nhân)", "So sánh lần lượt (Cần Key OpenAI)"),
        index=0
    )
    
    system_gemini_key = get_secret("GEMINI_API_KEY", "")
    system_openai_key = get_secret("OPENAI_API_KEY", "")
    
    active_gemini_key = system_gemini_key
    active_openai_key = ""

    if provider in ["Google Gemini (Theo quota tài khoản)", "So sánh lần lượt (Cần Key OpenAI)"]:
        # Cho phép cấu hình model; không suy đoán model duy nhất từ một lỗi API.
        gemini_model = st.text_input(
            "Tên model Gemini:",
            value=get_secret("GEMINI_MODEL", "gemini-3.8-flash"),
            help="Dùng đúng model ID mà dự án của em được phép gọi. Danh sách model không đảm bảo còn quota suy luận.",
        ).strip()
        gemini_thinking = st.selectbox(
            "Mức suy luận (áp dụng cho Gemini 3.8 Flash):",
            ["medium", "low", "high"],
            help="Low có thể giảm thời gian chờ nhưng cần đánh giá lại chất lượng đọc ECG. Không đảm bảo độ chính xác tương đương.",
        )
        if system_gemini_key:
            st.success("🟢 Đang dùng Gemini API Key hệ thống")
        else:
            active_gemini_key = st.text_input("Nhập Google Gemini API Key:", type="password")

        if st.button("Lấy danh sách model từ API"):
            if not active_gemini_key:
                st.warning("Cần API key để lấy danh sách model.")
            else:
                try:
                    with genai.Client(api_key=active_gemini_key.strip(), http_options=types.HttpOptions(
                        timeout=15000, retry_options=types.HttpRetryOptions(attempts=1)
                    )) as client:
                        names = [m.name for m in client.models.list()
                                 if "generateContent" in (m.supported_actions or [])]
                    st.code("\n".join(names) or "Không có model generateContent trong danh sách.")
                    st.caption("Có tên trong danh sách chưa đảm bảo được phép phân tích ảnh hoặc còn quota.")
                except Exception as error:
                    st.error(friendly_api_error(error))
                    st.code(str(error))

    if provider in ["OpenAI GPT-4o (Yêu cầu Key cá nhân)", "So sánh lần lượt (Cần Key OpenAI)"]:
        openai_model = st.selectbox("Phiên bản OpenAI:", ["gpt-4o", "gpt-4o-mini"])
        if system_openai_key:
            use_sys_openai = st.checkbox("Dùng key OpenAI hệ thống", value=True)
            active_openai_key = system_openai_key if use_sys_openai else st.text_input("Nhập OpenAI API Key cá nhân:", type="password")
        else:
            active_openai_key = st.text_input("Nhập OpenAI API Key cá nhân (sk-...):", type="password")

    st.markdown("---")
    st.header("🛠️ Bộ Tiền Xử Lý Ảnh")
    enable_preprocessing = st.checkbox("Bật tiền xử lý ảnh nâng cao", value=False)
    if enable_preprocessing:
        remove_shadow = st.checkbox("Khử bóng mờ & Đều ánh sáng nền", value=True)
        apply_clahe = st.checkbox("Tăng tương phản lưới sóng (CLAHE)", value=True)
        clahe_clip = st.slider("Mức độ CLAHE:", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
        sharpen_lvl = st.slider("Độ sắc nét sóng:", min_value=0.0, max_value=3.0, value=1.0, step=0.5)
        rot_angle = st.slider("Cân góc xoay (°):", min_value=-15, max_value=15, value=0, step=1)
    else:
        remove_shadow, apply_clahe, clahe_clip, sharpen_lvl, rot_angle = False, False, 2.0, 0.0, 0

    st.markdown("---")
    st.header("📋 Thông tin Ca Bệnh (Tùy chọn)")
    patient_age = st.number_input("Tuổi bệnh nhân:", min_value=0, max_value=120, value=60)
    patient_sex = st.selectbox("Giới tính:", ["Nam", "Nữ", "Không xác định"])
    clinical_history = st.text_area("Bệnh sử / Triệu chứng lâm sàng:", placeholder="VD: Đau thắt ngực trái 2 giờ trước...")

st.subheader("1. Tải ảnh hoặc Chụp trực tiếp")
uploaded_file = st.file_uploader(
    "Chọn ảnh ECG 12 chuyển đạo", 
    type=["jpg", "jpeg", "png", "jfif", "webp", "bmp", "heic", "heif"]
)
if st.checkbox("Sử dụng Camera"):
    cam_file = st.camera_input("Chụp ảnh ECG")
    if cam_file:
        uploaded_file = cam_file

raw_image = None
final_image_to_analyze = None

if uploaded_file is not None:
    raw_image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    raw_image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    if enable_preprocessing:
        processed_image = preprocess_ecg(
            raw_image,
            remove_shadow=remove_shadow,
            apply_clahe=apply_clahe,
            clahe_clip=clahe_clip,
            sharpen_level=sharpen_lvl,
            rotation_angle=rot_angle
        )
        final_image_to_analyze = processed_image
        c_raw, c_proc = st.columns(2)
        with c_raw:
            st.image(raw_image, caption="Ảnh gốc (Raw Image)", width="stretch")
        with c_proc:
            st.image(processed_image, caption="Ảnh sau tiền xử lý (Sẽ gửi vào AI)", width="stretch")
    else:
        final_image_to_analyze = raw_image
        st.image(raw_image, caption="Ảnh ECG sử dụng trực tiếp", width="stretch")

if provider in ["Google Gemini (Theo quota tài khoản)", "So sánh lần lượt (Cần Key OpenAI)"]:
    with st.expander("🔎 Kiểm tra API khi phân tích bị timeout", expanded=False):
        st.caption("Thử từng bước. Mỗi nút gửi một yêu cầu đến cùng model đang chọn; có thể tính phí API.")
        if st.button("1. Thử văn bản ngắn (không gửi ảnh)"):
            if not active_gemini_key:
                st.warning("Vui lòng nhập API key trước.")
            else:
                with st.spinner("Đang thử API với văn bản ngắn..."):
                    st.session_state["api_probe_text"] = probe_gemini(active_gemini_key, gemini_model)
        if st.button("2. Thử ảnh đang tải + câu trả lời ngắn"):
            if not active_gemini_key or final_image_to_analyze is None:
                st.warning("Cần API key và ảnh trước khi thử bước 2.")
            else:
                with st.spinner("Đang thử API với ảnh..."):
                    st.session_state["api_probe_image"] = probe_gemini(active_gemini_key, gemini_model, final_image_to_analyze)
        for key in ("api_probe_text", "api_probe_image"):
            if key in st.session_state:
                st.json(st.session_state[key])
        st.caption("Nếu bước 1 cũng timeout: chưa thể quy lỗi cho ảnh ECG. Nếu cả hai bước thành công nhưng phân tích đầy đủ timeout: cần kiểm tra độ dài yêu cầu, suy luận và tải máy chủ. Kết quả chỉ phản ánh thời điểm thử.")

full_prompt = SYSTEM_PROMPT
if clinical_history or patient_age:
    full_prompt += f"\n\n### BỐI CẢNH LÂM SÀNG:\n- Tuổi: {patient_age}\n- Giới tính: {patient_sex}\n- Thông tin thêm: {clinical_history if clinical_history else 'Không có'}"

st.markdown("---")
if st.button("🚀 Bắt đầu Phân tích ECG", type="primary", width="stretch"):
    if final_image_to_analyze is None:
        st.warning("Vui lòng tải lên hoặc chụp một ảnh ECG trước khi phân tích!")
    else:
        if provider == "Google Gemini (Theo quota tài khoản)":
            if not active_gemini_key:
                st.error("Chưa có Gemini API Key! Hãy cấu hình trong Secrets hoặc nhập ở menu bên trái.")
            else:
                st.subheader(f"📋 Báo Cáo Phân Tích ({gemini_model})")
                try:
                    display_analysis(
                        analyze_with_gemini_stream(active_gemini_key, gemini_model, final_image_to_analyze, full_prompt, thinking_level=gemini_thinking)
                    )
                except Exception as e:
                    st.error(friendly_api_error(e))

        elif provider == "OpenAI GPT-4o (Yêu cầu Key cá nhân)":
            if not active_openai_key:
                st.warning("Vui lòng nhập OpenAI API Key cá nhân của bạn.")
            else:
                base64_img = encode_image_to_base64(final_image_to_analyze)
                st.subheader(f"📋 Báo Cáo Phân Tích ({openai_model})")
                try:
                    display_analysis(
                        analyze_with_openai_stream(active_openai_key, openai_model, base64_img, full_prompt)
                    )
                except Exception as e:
                    st.error(f"Lỗi OpenAI: {e}")

        elif provider == "So sánh lần lượt (Cần Key OpenAI)":
            if not active_gemini_key or not active_openai_key:
                st.warning("Chế độ so sánh cần cả Gemini Key và OpenAI Key.")
            else:
                base64_img = encode_image_to_base64(final_image_to_analyze)
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader(f"🔹 {gemini_model}")
                    try:
                        display_analysis(
                            analyze_with_gemini_stream(active_gemini_key, gemini_model, final_image_to_analyze, full_prompt, thinking_level=gemini_thinking)
                        )
                    except Exception as e:
                        st.error(f"Lỗi Gemini: {e}")
                with c2:
                    st.subheader(f"🔸 {openai_model}")
                    try:
                        display_analysis(
                            analyze_with_openai_stream(active_openai_key, openai_model, base64_img, full_prompt)
                        )
                    except Exception as e:
                        st.error(f"Lỗi OpenAI: {e}")
