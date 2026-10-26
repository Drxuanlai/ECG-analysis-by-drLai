import streamlit as st
from PIL import Image
import cv2
import numpy as np
from google import genai
from openai import OpenAI
import io
import base64

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

Lưu ý: Bắt đầu kết luận với lưu ý đây là công cụ AI hỗ trợ, quyết định lâm sàng cuối cùng thuộc về bác sĩ điều trị.
"""

# ----------------- HÀM GỌI GEMINI STREAM (CHUẨN TỐC ĐỘ CAO) -----------------
def analyze_with_gemini_stream(api_key, model_name, image, prompt):
    client = genai.Client(api_key=api_key.strip())
    
    # Nén nhanh kích thước ảnh về tối đa 1600px để truyền dữ liệu trong tích tắc
    img_to_send = image.copy()
    if max(img_to_send.size) > 1600:
        img_to_send.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    
    # Truyền phát dữ liệu trực tiếp, không can thiệp tham số gây xung đột
    response = client.models.generate_content_stream(
        model=model_name,
        contents=[img_to_send, prompt]
    )
    for chunk in response:
        if chunk.text:
            yield chunk.text

def encode_image_to_base64(pil_image):
    buffered = io.BytesIO()
    rgb_image = pil_image.convert('RGB')
    if max(rgb_image.size) > 1600:
        rgb_image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    rgb_image.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_with_openai_stream(api_key, model_name, base64_image, prompt):
    client = OpenAI(api_key=api_key.strip())
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
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# ----------------- GIAO DIỆN CHÍNH -----------------
st.title("🫀 Phân Tích Điện Tâm Đồ 12 Chuyển Đạo")
st.caption("Hệ thống AI chuyên khoa Tim mạch - Phân tích sóng thời gian thực & Tiền xử lý ảnh")

with st.sidebar:
    st.header("⚙️ Lựa chọn Mô hình AI")
    provider = st.radio(
        "Chọn chế độ phân tích:",
        ("Google Gemini (Mặc định - Miễn phí)", "OpenAI GPT-4o (Yêu cầu Key cá nhân)", "So sánh song song (Cần Key OpenAI)"),
        index=0
    )
    
    system_gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    system_openai_key = st.secrets.get("OPENAI_API_KEY", "")
    
    active_gemini_key = system_gemini_key
    active_openai_key = ""

    if provider in ["Google Gemini (Mặc định - Miễn phí)", "So sánh song song (Cần Key OpenAI)"]:
        gemini_model = st.selectbox(
            "Phiên bản Gemini:",
            ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"],
            index=0
        )
        if system_gemini_key:
            st.success("🟢 Đang dùng Gemini API Key hệ thống")
        else:
            active_gemini_key = st.text_input("Nhập Google Gemini API Key:", type="password")

    if provider in ["OpenAI GPT-4o (Yêu cầu Key cá nhân)", "So sánh song song (Cần Key OpenAI)"]:
        openai_model = st.selectbox("Phiên bản OpenAI:", ["gpt-4o", "gpt-4o-mini"])
        if system_openai_key:
            use_sys_openai = st.checkbox("Dùng key OpenAI hệ thống", value=True)
            active_openai_key = system_openai_key if use_sys_openai else st.text_input("Nhập OpenAI API Key cá nhân:", type="password")
        else:
            active_openai_key = st.text_input("Nhập OpenAI API Key cá nhân (sk-...):", type="password")

    st.markdown("---")
    st.header("🛠️ Bộ Tiền Xử Lý Ảnh")
    enable_preprocessing = st.checkbox("Bật tiền xử lý ảnh nâng cao", value=True)
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
    raw_image = Image.open(uploaded_file)
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

full_prompt = SYSTEM_PROMPT
if clinical_history or patient_age:
    full_prompt += f"\n\n### BỐI CẢNH LÂM SÀNG:\n- Tuổi: {patient_age}\n- Giới tính: {patient_sex}\n- Thông tin thêm: {clinical_history if clinical_history else 'Không có'}"

st.markdown("---")
if st.button("🚀 Bắt đầu Phân tích ECG", type="primary", width="stretch"):
    if final_image_to_analyze is None:
        st.warning("Vui lòng tải lên hoặc chụp một ảnh ECG trước khi phân tích!")
    else:
        if provider == "Google Gemini (Mặc định - Miễn phí)":
            if not active_gemini_key:
                st.error("Chưa có Gemini API Key! Hãy cấu hình trong Secrets hoặc nhập ở menu bên trái.")
            else:
                st.subheader(f"📋 Báo Cáo Phân Tích ({gemini_model})")
                try:
                    st.write_stream(
                        analyze_with_gemini_stream(active_gemini_key, gemini_model, final_image_to_analyze, full_prompt)
                    )
                except Exception as e:
                    st.error(f"Lỗi phân tích: {e}")

        elif provider == "OpenAI GPT-4o (Yêu cầu Key cá nhân)":
            if not active_openai_key:
                st.warning("Vui lòng nhập OpenAI API Key cá nhân của bạn.")
            else:
                base64_img = encode_image_to_base64(final_image_to_analyze)
                st.subheader(f"📋 Báo Cáo Phân Tích ({openai_model})")
                try:
                    st.write_stream(
                        analyze_with_openai_stream(active_openai_key, openai_model, base64_img, full_prompt)
                    )
                except Exception as e:
                    st.error(f"Lỗi OpenAI: {e}")

        elif provider == "So sánh song song (Cần Key OpenAI)":
            if not active_gemini_key or not active_openai_key:
                st.warning("Chế độ so sánh cần cả Gemini Key và OpenAI Key.")
            else:
                base64_img = encode_image_to_base64(final_image_to_analyze)
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader(f"🔹 {gemini_model}")
                    try:
                        st.write_stream(
                            analyze_with_gemini_stream(active_gemini_key, gemini_model, final_image_to_analyze, full_prompt)
                        )
                    except Exception as e:
                        st.error(f"Lỗi Gemini: {e}")
                with c2:
                    st.subheader(f"🔸 {openai_model}")
                    try:
                        st.write_stream(
                            analyze_with_openai_stream(active_openai_key, openai_model, base64_img, full_prompt)
                        )
                    except Exception as e:
                        st.error(f"Lỗi OpenAI: {e}")
