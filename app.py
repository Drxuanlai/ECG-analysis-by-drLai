import streamlit as st
from PIL import Image
import os
import io
import base64
import cv2
import numpy as np
from google import genai
from openai import OpenAI

# ----------------- CẤU HÌNH TRANG -----------------
st.set_page_config(
    page_title="AI 12-Lead ECG Expert",
    page_icon="🫀",
    layout="wide"
)

# ----------------- HÀM TIỀN XỬ LÝ ẢNH (OPENCV) -----------------
def preprocess_ecg(
    pil_img, 
    remove_shadow=True, 
    apply_clahe=True, 
    clahe_clip=2.0, 
    sharpen_level=1.0, 
    rotation_angle=0
):
    # Chuyển PIL Image sang OpenCV BGR
    img = np.array(pil_img)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # 1. Xoay ảnh cân chỉnh góc
    if rotation_angle != 0:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Chuyển sang ảnh Grayscale để xử lý nét sóng
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Khử bóng đổ và cân bằng ánh sáng nền
    if remove_shadow:
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated, 21)
        diff_img = 255 - cv2.absdiff(gray, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        processed = norm_img
    else:
        processed = gray

    # 3. Tăng tương phản cục bộ thích ứng (CLAHE)
    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        processed = clahe.apply(processed)

    # 4. Làm sắc nét viền sóng (Unsharp Masking)
    if sharpen_level > 0:
        gaussian = cv2.GaussianBlur(processed, (0, 0), 2.0)
        sharpened = cv2.addWeighted(processed, 1.0 + sharpen_level, gaussian, -sharpen_level, 0)
        processed = np.clip(sharpened, 0, 255).astype(np.uint8)

    # Chuyển ngược lại RGB dạng PIL để hiển thị và gửi API
    processed_rgb = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(processed_rgb)

# ----------------- PROMPT CHUYÊN KHOA TIM MẠCH -----------------
SYSTEM_PROMPT = """
Bạn là một chuyên gia Tim mạch can thiệp và Điện sinh lý học tim hàng đầu.
Nhiệm vụ của bạn là đọc và phân tích hình ảnh Điện tâm đồ 12 chuyển đạo tiêu chuẩn được cung cấp.

Hãy tiến hành đọc một cách chặt chẽ, tuần tự theo các bước chuẩn lâm sàng sau:

### 1. ĐÁNH GIÁ CHẤT LƯỢNG KỸ THUẬT & DẪN TRUYỀN
- Kiểm tra test biên độ (Calibration): Thường là 10 mm/mV (1 mV = 2 ô lớn).
- Tốc độ kéo giấy (Paper speed): Thường là 25 mm/s (1 ô nhỏ = 0.04s, 1 ô lớn = 0.2s).
- Đánh giá nhiễu: Nhiễu cơ vân, đường đẳng điện trôi, mắc ngược điện cực (aVR có P và QRS dương bất thường không?).

### 2. TẦN SỐ & NHỊP
- Tần số tim: Khoảng bao nhiêu nhịp/phút?
- Nhịp: Nhịp xoang đều/không đều? Hay loạn nhịp (Rung nhĩ, Cuồng nhĩ, PSVT, VT, Ngoại tâm thu nhĩ/thất...)?

### 3. TRỤC ĐIỆN TIM & GÓC ALPHA
- Trục: Bình thường (0° đến +90°), Lệch trái (-30° đến -90°), Lệch phải (+90° đến +180°), hay Vô định?
- Phân tích dựa trên D1, D2, aVF.

### 4. ĐO ĐẠC HÌNH THÁI CÁC SÓNG & KHOẢNG
- **Sóng P**: Biên độ (dày nhĩ phải nếu > 2.5mm ở DII), thời gian (dày nhĩ trái nếu > 0.12s, P hai đỉnh ở DII hoặc pha âm sâu ở V1).
- **Khoảng PR**: Bình thường (0.12 - 0.20s), Ngắn (Hội chứng WPW, LGL), Dài (Block AV độ I, Mobitz I, Mobitz II, Block AV độ III).
- **Phức bộ QRS**: 
  - Thời gian: Hẹp (< 0.12s) hay Rộng (Block nhánh Trái/Phải, nhịp tự thất)?
  - Biên độ & Dày thất: Tiêu chuẩn Sokolow-Lyon (SV1 + RV5/V6 >= 35mm), Cornell (RaVL + SV3).
  - Sóng Q hoại tử: Có sóng Q bệnh lý (sâu > 1/4 R, rộng >= 0.04s) ở vùng nào không?
- **Đoạn ST**: 
  - Đẳng điện, chênh lên hay chênh xuống? Dạng vòm, dốc lên, dốc xuống?
  - Định vị thành tổn thương: Trước vách (V1-V2), Trước mỏm (V3-V4), Bên (DI, aVL, V5-V6), Thành dưới (DII, DIII, aVF), Thành sau...
  - Hình ảnh soi gương (reciprocal changes)?
- **Sóng T**: Chiều, biên độ, hình dạng (T âm nhọn đối xứng, T cao nhọn đối xứng...).
- **Khoảng QTc**: Kéo dài hay bình thường?

### 5. KẾT LUẬN & GỢI Ý XỬ TRÍ
- **Kết luận điện tâm đồ**: Tóm tắt chẩn đoán ngắn gọn.
- **Mức độ khẩn cấp**: [TỐI KHẨN CẤP / KHẨN CẤP / THEO DÕI THƯỜNG QUY].
- **Đề xuất cận lâm sàng kết hợp**: (Troponin hs, Khí máu, Siêu âm tim tại giường, Điện giải đồ...).

*Lưu ý: Luôn bắt đầu phần kết luận bằng lời nhắc: Đây là công cụ AI hỗ trợ đọc dữ liệu thô, quyết định lâm sàng cuối cùng thuộc về bác sĩ trực tiếp khám bệnh.*
"""

def encode_image_to_base64(pil_image):
    buffered = io.BytesIO()
    rgb_image = pil_image.convert('RGB')
    rgb_image.save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_with_gemini(api_key, model_name, image, prompt):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=[image, prompt]
    )
    return response.text

def analyze_with_openai(api_key, model_name, base64_image, prompt):
    client = OpenAI(api_key=api_key)
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
        temperature=0.2
    )
    return response.choices[0].message.content

# ----------------- GIAO DIỆN CHÍNH -----------------
st.title("🫀 Phân Tích Điện Tâm Đồ 12 Chuyển Đạo")
st.caption("Ứng dụng AI phân tích ECG tích hợp bộ lọc tăng nét và khử nhiễu ảnh y tế")

# --- SIDEBAR: Cấu hình API & Tiền xử lý ---
with st.sidebar:
    st.header("⚙️ Cấu hình Mô hình")
    provider = st.radio(
        "Nhà cung cấp AI:",
        ("Google Gemini", "OpenAI (ChatGPT)", "So sánh cả 2 mô hình"),
        index=0
    )
    
    default_gemini_key = st.secrets.get("GEMINI_API_KEY", "")
    default_openai_key = st.secrets.get("OPENAI_API_KEY", "")
    gemini_key = default_gemini_key
    openai_key = default_openai_key
    
    if provider in ["Google Gemini", "So sánh cả 2 mô hình"]:
        gemini_model = st.selectbox("Phiên bản Gemini:", ["gemini-2.5-flash", "gemini-1.5-pro", "gemini-2.5-pro"])
        if not default_gemini_key:
            gemini_key = st.text_input("Gemini API Key:", type="password")

    if provider in ["OpenAI (ChatGPT)", "So sánh cả 2 mô hình"]:
        openai_model = st.selectbox("Phiên bản OpenAI:", ["gpt-4o", "gpt-4o-mini"])
        if not default_openai_key:
            openai_key = st.text_input("OpenAI API Key:", type="password")
            
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
        remove_shadow = False
        apply_clahe = False
        clahe_clip = 2.0
        sharpen_lvl = 0.0
        rot_angle = 0

    st.markdown("---")
    st.header("📋 Thông tin Ca Bệnh")
    patient_age = st.number_input("Tuổi bệnh nhân:", min_value=0, max_value=120, value=60)
    patient_sex = st.selectbox("Giới tính:", ["Nam", "Nữ", "Không xác định"])
    clinical_history = st.text_area(
        "Bệnh sử / Triệu chứng lâm sàng:",
        placeholder="VD: Đau thắt ngực kiểu mạch vành kèm vã mồ hôi 1h, tiền căn rung nhĩ đang dùng NOAC..."
    )

# --- KHUNG TẢI ẢNH & HIỂN THỊ TRƯỚC / SAU ---
st.subheader("1. Tải ảnh hoặc Chụp trực tiếp")
uploaded_file = st.file_uploader("Chọn ảnh ECG 12 chuyển đạo (JPG/PNG)", type=["jpg", "jpeg", "png"])
enable_cam = st.checkbox("Sử dụng Camera")
if enable_cam:
    cam_file = st.camera_input("Chụp ảnh ECG")
    if cam_file:
        uploaded_file = cam_file

raw_image = None
final_image_to_analyze = None

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    
    # Thực hiện tiền xử lý nếu được bật
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
        
        # Hiển thị song song ảnh gốc và ảnh sau xử lý
        col_raw, col_proc = st.columns(2)
        with col_raw:
            st.image(raw_image, caption="Ảnh gốc (Raw Image)", use_container_width=True)
        with col_proc:
            st.image(processed_image, caption="Ảnh sau tiền xử lý (Sẽ gửi vào AI)", use_container_width=True)
    else:
        final_image_to_analyze = raw_image
        st.image(raw_image, caption="Ảnh ECG sử dụng trực tiếp", use_container_width=True)

# Chuẩn bị Prompt
full_prompt = SYSTEM_PROMPT
if clinical_history or patient_age:
    full_prompt += f"\n\n### BỐI CẢNH LÂM SÀNG CỦA CA BỆNH:\n- Tuổi: {patient_age}\n- Giới tính: {patient_sex}\n- Thông tin thêm: {clinical_history if clinical_history else 'Không có'}\nHãy kết hợp bối cảnh này để biện luận các biến đổi trên ECG."

# --- NÚT PHÂN TÍCH ---
st.markdown("---")
if st.button("🚀 Bắt đầu Phân tích ECG", type="primary", use_container_width=True):
    if final_image_to_analyze is None:
        st.warning("Vui lòng tải lên hoặc chụp một ảnh ECG trước!")
    else:
        base64_img = encode_image_to_base64(final_image_to_analyze)
        
        if provider == "Google Gemini":
            if not gemini_key:
                st.error("Thiếu Gemini API Key!")
            else:
                with st.spinner(f"Đang phân tích qua {gemini_model}..."):
                    try:
                        res = analyze_with_gemini(gemini_key, gemini_model, final_image_to_analyze, full_prompt)
                        st.subheader(f"📋 Báo Cáo Phân Tích ({gemini_model})")
                        st.markdown(res)
                    except Exception as e:
                        st.error(f"Lỗi: {e}")

        elif provider == "OpenAI (ChatGPT)":
            if not openai_key:
                st.error("Thiếu OpenAI API Key!")
            else:
                with st.spinner(f"Đang phân tích qua {openai_model}..."):
                    try:
                        res = analyze_with_openai(openai_key, openai_model, base64_img, full_prompt)
                        st.subheader(f"📋 Báo Cáo Phân Tích ({openai_model})")
                        st.markdown(res)
                    except Exception as e:
                        st.error(f"Lỗi: {e}")

        elif provider == "So sánh cả 2 mô hình":
            if not gemini_key or not openai_key:
                st.error("Cần nhập cả 2 API Key để so sánh!")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    with st.spinner(f"Đang tải {gemini_model}..."):
                        try:
                            gemini_res = analyze_with_gemini(gemini_key, gemini_model, final_image_to_analyze, full_prompt)
                            st.subheader(f"🔹 {gemini_model}")
                            st.markdown(gemini_res)
                        except Exception as e:
                            st.error(f"Gemini Lỗi: {e}")
                with c2:
                    with st.spinner(f"Đang tải {openai_model}..."):
                        try:
                            openai_res = analyze_with_openai(openai_key, openai_model, base64_img, full_prompt)
                            st.subheader(f"🔸 {openai_model}")
                            st.markdown(openai_res)
                        except Exception as e:
                            st.error(f"OpenAI Lỗi: {e}")
