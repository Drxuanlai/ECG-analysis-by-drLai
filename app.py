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
import csv
import hashlib
import re
import shutil
import subprocess
import json
from datetime import datetime, timezone

# Hỗ trợ ảnh chụp iPhone (HEIC/HEIF)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

st.set_page_config(
    page_title="Phân tích ECG — by dr. Lai",
    page_icon="🫀",
    layout="wide"
)

# OCR runs locally. It detects readable ECG labels, not waveform polarity.
# Angles are COUNTERCLOCKWISE relative to the EXIF-corrected image.
def rotate_quarter_turn(image, angle):
    rotations = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }
    return image.transpose(rotations[angle]) if angle in rotations else image.copy()


def score_ecg_labels(tsv):
    weights = {"avr": 4, "avl": 4, "avf": 4,
               "v1": 2.5, "v2": 2.5, "v3": 2.5,
               "v4": 2.5, "v5": 2.5, "v6": 2.5,
               "rhythm": 5, "bpm": 4, "mmmv": 4, "mv": 2,
               "filter": 3, "mms": 3}
    hits = {}
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        try:
            confidence = float(row.get("conf", -1))
        except (TypeError, ValueError):
            continue
        token = re.sub(r"[^a-z0-9]", "", row.get("text", "").lower())
        if confidence >= 30 and token in weights:
            hits[token] = max(hits.get(token, 0), weights[token] * confidence / 100)
    return sum(hits.values()), len(hits)


def detect_ecg_orientation(image):
    executable = shutil.which("tesseract")
    if not executable:
        return {"angle": 0, "detected": False,
                "message": "Chưa có bộ nhận diện chiều ảnh. Thêm packages.txt rồi Reboot app; hiện có thể xoay tay."}
    probe = image.convert("RGB")
    probe.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    candidates = []
    failed = 0
    for angle in (0, 90, 180, 270):
        buffer = io.BytesIO()
        rotate_quarter_turn(probe, angle).save(buffer, format="PNG")
        try:
            result = subprocess.run(
                [executable, "stdin", "stdout", "--psm", "11", "-l", "eng", "tsv"],
                input=buffer.getvalue(), capture_output=True, timeout=5, check=True,
            )
            score, count = score_ecg_labels(result.stdout.decode("utf-8", errors="replace"))
            candidates.append((score, count, angle))
        except (subprocess.SubprocessError, OSError):
            failed += 1
    candidates.sort(reverse=True)
    # If a candidate could not be tested, don't silently choose among the rest.
    if failed or len(candidates) != 4:
        return {"angle": 0, "detected": False,
                "message": "Nhận diện chữ chưa hoàn tất. Giữ nguyên ảnh; em có thể xoay tay bên dưới."}
    best, second = candidates[0], candidates[1]
    if best[0] < 6 or best[1] < 2 or best[0] - second[0] < 3:
        return {"angle": 0, "detected": False,
                "message": "Chưa đủ dấu hiệu để tự xác định chiều. Giữ nguyên ảnh; hãy kiểm tra và xoay tay."}
    return {"angle": best[2], "detected": True,
            "message": "Đã nhận diện chiều dựa trên chữ và nhãn ECG. Hãy kiểm tra ảnh trước khi phân tích."}


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
APP_VERSION = "v7"
PROMPT_VERSION = "ecg-evidence-v7"
SYSTEM_PROMPT = """
Bạn hỗ trợ bác sĩ mô tả ECG từ ảnh. Đây là bản phân tích chưa được bác sĩ xác minh.
Không có chẩn đoán đích được gợi sẵn. Không mặc định ECG bình thường hay bệnh lý.
Chỉ mô tả bằng chứng nhìn thấy; không tạo số đo để khớp một chẩn đoán.

QUY TẮC NGUỒN DỮ LIỆU
- Phân biệt: quan sát dạng sóng; số đo ước lượng từ ảnh; chữ/số do máy in;
  thông tin người dùng đã xác nhận. Không gọi số máy in là phép đo độc lập.
- Đọc riêng nhãn chuyển đạo, tốc độ và chuẩn khuếch đại từng nhóm. Không đoán
  chuẩn ghi từ vị trí chuyển đạo hoặc từ một cấu hình thông dụng.
- Chữ chuẩn ghi và xung chuẩn là hai nguồn cần đối chiếu. Nếu chữ bị che và
  không xác định được biên độ tín hiệu chuẩn của máy, ghi chưa xác định mm/mV.
  Không tự mặc định độ cao xung tương ứng 1 mV khi điều này chưa rõ.
- Nếu không xác định chuẩn biên độ, không quy đổi sang mV và không tính chỉ số
  điện thế phì đại thất. Tốc độ không rõ thì không đưa thời gian ms chắc chắn.
- Biên độ phải gắn đúng chuyển đạo và cực tính. Kiểm tra đó là R, S hay T;
  không lấy sóng lớn ở một chuyển đạo làm số đo của chuyển đạo khác.
- Số ước lượng phải có chuyển đạo, số ô lưới, mốc đo, chuẩn ghi và sai số/giới hạn.
  Không thấy đủ các mốc thì ghi không đo được, không bịa con số.

TRÌNH TỰ BÁO CÁO (trình bày ngắn gọn, không viết suy luận nội bộ)
1. CHẤT LƯỢNG & CHUẨN GHI
   Nhãn nào đọc được, vị trí bị che/cắt/nhiễu, tốc độ và chuẩn từng nhóm với nguồn.
   Không khẳng định mắc điện cực hoàn toàn đúng chỉ từ dấu P/QRS ở aVR.
2. BẢNG QUAN SÁT TRƯỚC CHẨN ĐOÁN
   Cột: Hạng mục | Quan sát/số đo | Chuyển đạo và mốc đo | Nguồn | Giới hạn.
   Gồm: độ đều RR; liên hệ P-QRS; tần số; QRS từ khởi đầu sớm nhất đến kết thúc
   muộn nhất có thể xác định ở chuyển đạo ghi đồng thời; hình thái tận cùng QRS;
   cực tính QRS để xét trục; PR; ST và T tại từng chuyển đạo bất thường; QT nếu đo được.
   Không chỉ đo đỉnh nhọn của QRS. Không coi thành phần tận cùng QRS là ST/T.
   Nếu có thay đổi RR/hình thái giữa các nhát, mô tả trước khi gọi tên nhịp.
3. DIỄN GIẢI CÓ ĐIỀU KIỆN
   Đánh giá dẫn truyền thất trước khi diễn giải ST-T. Đối chiếu chuyển đạo trước tim
   với chuyển đạo bên; không bỏ qua biến đổi tái cực thứ phát do dẫn truyền bất thường.
   Trục lệch đơn độc không tự động xác nhận block phân nhánh.
   Không kết luận hội chứng vành cụ thể hoặc vị trí hẹp mạch chỉ từ T âm;
   không loại trừ thiếu máu cơ tim chỉ vì có rối loạn dẫn truyền.
   Phân biệt hình thái ECG nghi ngờ với hội chứng cần bệnh sử và xét nghiệm.
   QT/QTc phải nêu nguồn, công thức nếu tự tính và ảnh hưởng QRS rộng.
4. KẾT LUẬN & ĐIỂM CẦN BÁC SĨ XÁC MINH
   Mỗi kết luận gắn bằng chứng ở mục 2; phân biệt có cơ sở/nghi ngờ/chưa xác định.
   Không gọi ECG bình thường nếu chưa xác định được QRS và chưa đánh giá đủ chuyển đạo.
   Không tự gán phần trăm tin cậy. Liệt kê số đo hoặc ảnh cần bổ sung.
   Nếu thiếu triệu chứng/sinh hiệu, ghi chưa đủ dữ kiện phân tầng khẩn cấp. Nếu có
   dấu hiệu nguy hiểm, mô tả cụ thể dấu hiệu và đề nghị bác sĩ đánh giá ngay.
   Không đưa đơn thuốc, liều thuốc, năng lượng sốc hoặc chỉ định can thiệp bắt buộc.
   Chỉ nêu bước xác minh phù hợp, có điều kiện. Không coi thiếu bệnh sử là không có triệu chứng.

QUY TẮC SỐ ĐO
- Mọi số do AI ước lượng đều phải ghi “AI ước lượng — chưa kiểm chứng”.
- Không xác định được tốc độ giấy từ ảnh hoặc thông tin người dùng đã xác nhận:
  chỉ mô tả ô lưới/hình thái; không giả định 25 mm/s và không kết luận thời gian
  ms hoặc tần số bpm. Không tự đưa giả định chuẩn vào phần kết luận.
- Chuẩn do người dùng xác nhận chỉ xác nhận chuẩn ghi, không xác nhận số đo AI.
- Chỉ nói “hình thái gợi ý” nếu chưa xác định đủ độ rộng QRS, không loại trừ block
  chỉ từ phần đỉnh nhọn. Nếu không phân biệt được tận cùng QRS với ST/T, ghi rõ.
- Không tự ghi “Bác sĩ đã xác minh” cho bất kỳ số đo hoặc kết luận nào.
- Tần số và RR phải nhất quán (bpm = 60000 / RR_ms). Khi không xác định được
  một chu kỳ hoặc các số mâu thuẫn, ghi chưa đo được; không đếm T như QRS.

Kết thúc bằng: Báo cáo AI chưa được xác minh; bác sĩ cần kiểm tra lại dạng sóng và số đo.
"""


# Timeout Google: milliseconds. One request, no nested automatic retries.
REQUEST_TIMEOUT_SECONDS = 45
STREAM_BUDGET_SECONDS = 90
MAX_IMAGE_EDGE = 2400
GEMINI_MODEL = "gemini-3.6-flash"


def image_to_png(image):
    prepared = image.convert("RGB")
    prepared.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    prepared.save(buffer, format="PNG")
    return buffer.getvalue()


def build_verified_context(speed, limb_gain, chest_gain, qrs_ms):
    items = {"Tốc độ giấy (mm/s)": speed, "Chuẩn chi (mm/mV)": limb_gain,
             "Chuẩn ngực (mm/mV, chỉ dùng nếu đồng nhất V1–V6)": chest_gain,
             "QRS (ms)": qrs_ms}
    rows = [f"- {label}: {value}" for label, value in items.items() if value is not None]
    if not rows:
        return "\n\nKhông có số đo/chuẩn ghi do người dùng xác nhận."
    return ("\n\n### SỐ LIỆU DO NGƯỜI DÙNG XÁC NHẬN, KHÔNG PHẢI AI TỰ ĐO:\n"
            + "\n".join(rows) + "\nNếu thấy mâu thuẫn với ảnh phải nêu mâu thuẫn; không âm thầm thay số.")




def build_clinical_context(age, sex, history):
    age_text = f"{age} tuổi" if age is not None else "Chưa cung cấp"
    sex_text = sex if sex in ("Nam", "Nữ", "Khác") else "Chưa cung cấp"
    history_text = history.strip() or "Chưa cung cấp"
    return (
        "\n\n### THÔNG TIN DO NGƯỜI DÙNG CUNG CẤP:\n"
        f"- Tuổi: {age_text}\n- Giới tính: {sex_text}\n"
        f"- Bệnh sử / triệu chứng: {history_text}\n"
        "Không tự điền tuổi hay giới tính khi chưa được cung cấp. Nếu đọc được thông tin "
        "trên ảnh, phải ghi rõ nguồn là chữ in trên bản ghi, chưa được người dùng xác nhận. "
        "Chưa cung cấp triệu chứng không có nghĩa là không có triệu chứng. "
        "Không suy diễn dữ liệu còn thiếu từ giá trị mặc định hoặc từ hình thái ECG."
    )


def gemini_config(model_name, thinking_level="medium"):
    options = {"max_output_tokens": 8192}
    # Only apply controls documented for this exact model.
    if model_name.removeprefix("models/") == GEMINI_MODEL:
        options["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)
    return types.GenerateContentConfig(**options)




def analyze_with_gemini_stream(api_key, model_name, image_bytes, prompt, thinking_level="medium"):
    with genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(
            timeout=REQUEST_TIMEOUT_SECONDS * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) as client:
        response = client.models.generate_content_stream(
            model=model_name,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/png"), prompt],
            config=gemini_config(model_name, thinking_level),
        )
        finished = False
        try:
            for chunk in response:
                if chunk.text:
                    yield chunk.text
                for candidate in chunk.candidates or []:
                    reason = getattr(candidate.finish_reason, "value", candidate.finish_reason)
                    if reason == "STOP":
                        finished = True
                    if reason and reason != "STOP":
                        raise RuntimeError(
                            f"Báo cáo chưa hoàn tất (finish_reason={reason}). "
                            "Không sử dụng phần nội dung đang hiển thị như một kết luận đầy đủ."
                        )
            if not finished:
                raise RuntimeError("Luồng kết thúc thiếu tín hiệu hoàn tất; báo cáo có thể bị thiếu.")
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
    fragments = []
    completed = False
    error_message = None
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
                    fragments.append(text)
                    yield text
            if first_text_seconds is None:
                raise RuntimeError("API kết thúc nhưng không trả nội dung báo cáo.")
        finally:
            generator.close()

    try:
        st.write_stream(timed_chunks())
        completed = True
        status.caption(
            f"Hoàn tất: {time.monotonic() - started:.1f} giây · "
            f"Có chữ đầu tiên sau: {first_text_seconds:.1f} giây"
        )
    except Exception as error:
        error_message = friendly_api_error(error)
        status.caption(f"Dừng sau {time.monotonic() - started:.1f} giây")
        st.error(friendly_api_error(error))
        if first_text_seconds is not None:
            st.warning("Báo cáo bị gián đoạn, nội dung phía trên chưa phải báo cáo hoàn chỉnh.")

    return {"report": "".join(fragments), "completed": completed,
            "error": error_message, "total_seconds": round(time.monotonic()-started, 2),
            "first_text_seconds": round(first_text_seconds, 2) if first_text_seconds is not None else None}


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
                                "url": f"data:image/png;base64,{base64_image}",
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
        finished = False
        try:
            for chunk in response:
                if chunk.choices:
                    choice = chunk.choices[0]
                    if choice.delta.content:
                        yield choice.delta.content
                    if choice.finish_reason == "stop":
                        finished = True
                    if choice.finish_reason and choice.finish_reason != "stop":
                        raise RuntimeError(f"Báo cáo chưa hoàn tất ({choice.finish_reason}).")
            if not finished:
                raise RuntimeError("Luồng kết thúc thiếu tín hiệu hoàn tất; báo cáo có thể bị thiếu.")
        finally:
            response.close()
    finally:
        client.close()

def get_secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


# ----------------- GIAO DIỆN ĐỒNG NGHIỆP -----------------
def crop_box_percent(size, horizontal, vertical):
    w, h = size
    left, right = horizontal
    top, bottom = vertical
    if not (0 <= left < right <= 100 and 0 <= top < bottom <= 100):
        raise ValueError("Vùng cắt phải có chiều rộng và chiều cao lớn hơn 0.")
    box = (int(w * left / 100), int(h * top / 100), int(w * right / 100), int(h * bottom / 100))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Vùng cắt quá nhỏ.")
    return box


def request_fingerprint(image_bytes, prompt, service, model, thinking):
    settings = json.dumps([service, model, thinking, prompt], ensure_ascii=False).encode()
    return hashlib.sha256(image_bytes + settings).hexdigest()


def reset_case():
    # Only this browser session. Never cache patient images across users.
    for key in list(st.session_state):
        if key.startswith("case_") or key.startswith("image_") or key in ("last_report_v7", "orientation_result"):
            del st.session_state[key]


st.title("🫀 Phân tích ECG — by dr. Lai")
st.caption("Hỗ trợ bác sĩ mô tả điện tâm đồ 12 chuyển đạo.")
st.info("Kết quả AI có thể sai. Bác sĩ cần đối chiếu ECG gốc và lâm sàng trước khi sử dụng; các số đo AI chưa được kiểm chứng.")

with st.sidebar:
    st.header("Thông tin ca bệnh")
    st.caption("Không nhập họ tên, mã hồ sơ hoặc thông tin định danh người bệnh.")
    patient_age = st.number_input("Tuổi", 0, 120, value=None, step=1, key="case_age")
    patient_sex = st.selectbox("Giới tính", ["Chưa cung cấp", "Nam", "Nữ", "Khác"], key="case_sex")
    clinical_history = st.text_area("Triệu chứng / bệnh sử / sinh hiệu", placeholder="Chưa cung cấp", key="case_history")
    with st.expander("Chuẩn ghi đã kiểm tra (tùy chọn)"):
        st.caption("Để trống nếu chưa xác định. Có thể nhập chuẩn chi và ngực khác nhau; các giá trị này do người dùng cung cấp, không phải AI tự đo.")
        known_speed = st.number_input("Tốc độ giấy (mm/s)", 1.0, 200.0, value=None, key="case_speed")
        known_limb_gain = st.number_input("Chuẩn chuyển đạo chi (mm/mV)", 0.1, 100.0, value=None, key="case_limb")
        known_chest_gain = st.number_input("Chuẩn ngực nếu đồng nhất V1–V6 (mm/mV)", 0.1, 100.0, value=None, key="case_chest")
        st.caption("Nếu chuẩn khác nhau giữa các chuyển đạo ngực, ghi rõ trong thông tin ca bệnh thay vì nhập một giá trị chung.")
    with st.expander("Thiết lập AI"):
        provider = st.selectbox("Dịch vụ", ["Google Gemini", "OpenAI"], key="service_v7")
        if provider == "Google Gemini":
            model = st.text_input("Tên model", value=get_secret("GEMINI_MODEL", GEMINI_MODEL), key="gmodel_v7").strip()
            thinking = st.selectbox("Mức suy luận", ["medium", "low", "high"], key="thinking_v7")
            if model.removeprefix("models/") != GEMINI_MODEL:
                st.caption("Điều khiển mức suy luận chỉ áp dụng cho Gemini 3.6 Flash trong bản này; model khác dùng thiết lập mặc định của dịch vụ.")
            api_key = get_secret("GEMINI_API_KEY", "")
            if not api_key:
                api_key = st.text_input("Gemini API key", type="password", key="gkey_v7")
        else:
            model = st.text_input("Tên model", value=get_secret("OPENAI_MODEL", "gpt-4o"), key="omodel_v7").strip()
            thinking = None
            api_key = get_secret("OPENAI_API_KEY", "")
            if not api_key:
                api_key = st.text_input("OpenAI API key", type="password", key="okey_v7")
        st.caption("Model cần hỗ trợ nhận ảnh và được tài khoản API cấp quyền truy cập.")
    st.button("Xóa thông tin và kết quả ca hiện tại", on_click=reset_case, key="reset_v7")

st.subheader("Ảnh điện tâm đồ")
st.caption("Che thông tin định danh trước khi tải. Chọn ảnh rõ, đúng chiều và giữ đủ nhãn chuyển đạo, lưới, xung chuẩn cùng đầu–cuối đường sóng.")
uploaded_file = st.file_uploader("Tải ảnh ECG", type=["jpg", "jpeg", "png", "jfif", "webp", "bmp", "heic", "heif"], key="upload_v7", on_change=reset_case)
if st.checkbox("Chụp bằng camera", key="camera_enabled_v7"):
    camera_file = st.camera_input("Chụp ECG", key="camera_v7", on_change=reset_case)
    if camera_file is not None:
        uploaded_file = camera_file

if uploaded_file is not None:
    try:
        original = ImageOps.exif_transpose(Image.open(io.BytesIO(uploaded_file.getvalue()))).convert("RGB")
    except Exception:
        st.error("Không đọc được ảnh. Hãy xuất JPEG/PNG rồi tải lại.")
        st.stop()
    image_id = hashlib.sha256(uploaded_file.getvalue()).hexdigest()[:16]
    with st.expander("Chỉnh chiều và vùng ảnh"):
        auto_orientation = st.checkbox("Tự nhận diện chiều ảnh", value=True, key="image_auto_" + image_id)
        auto_angle = 0
        if auto_orientation:
            cache = st.session_state.get("orientation_result", {})
            if cache.get("image_id") != image_id:
                with st.spinner("Đang kiểm tra chiều ảnh..."):
                    cache = {"image_id": image_id, **detect_ecg_orientation(original)}
                st.session_state["orientation_result"] = cache
            auto_angle = cache["angle"]
            if not cache["detected"]:
                st.caption("Chưa tự xác định được chiều. Hãy xem ảnh và xoay tay nếu cần.")
        manual_angle = st.selectbox("Xoay thêm ngược chiều kim đồng hồ", [0, 90, 180, 270], key="image_rotation_" + image_id)
        crop_enabled = st.checkbox("Cắt vùng tờ ECG", value=False, key="image_crop_" + image_id)
        horizontal, vertical = (0, 100), (0, 100)
        if crop_enabled:
            horizontal = st.slider("Biên trái — phải (%)", 0, 100, (0, 100), key="image_x_" + image_id)
            vertical = st.slider("Biên trên — dưới (%)", 0, 100, (0, 100), key="image_y_" + image_id)
        enhance = st.checkbox("Tăng tương phản ảnh", value=False, key="image_enhance_" + image_id)
        if enhance:
            st.caption("Hãy đối chiếu ảnh trước/sau vì tăng tương phản có thể làm thay đổi chi tiết đường sóng.")
    oriented = rotate_quarter_turn(original, (auto_angle + manual_angle) % 360)
    try:
        box = crop_box_percent(oriented.size, horizontal, vertical)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    raw_image = oriented.crop(box)
    raw_image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    final_image = preprocess_ecg(raw_image, True, True, 2.0, 1.0, 0) if enhance else raw_image
    payload = image_to_png(final_image)
    st.image(payload, caption="Ảnh sẽ gửi để phân tích", width="stretch")
    if enhance:
        with st.expander("Đối chiếu ảnh trước tăng tương phản"):
            st.image(raw_image, width="stretch")
    with st.expander("Phóng to để xem"):
        zoom_x = st.slider("Vùng xem ngang (%)", 0, 100, (0, 100), key="image_zoom_x_" + image_id)
        zoom_y = st.slider("Vùng xem dọc (%)", 0, 100, (0, 100), key="image_zoom_y_" + image_id)
        try:
            st.image(final_image.crop(crop_box_percent(final_image.size, zoom_x, zoom_y)), width="stretch")
        except ValueError:
            st.caption("Chọn vùng có chiều rộng và chiều cao lớn hơn 0.")
        st.caption("Phóng to chỉ để xem, không thay ảnh gửi phân tích.")

    prompt = SYSTEM_PROMPT + build_clinical_context(patient_age, patient_sex, clinical_history)
    prompt += build_verified_context(known_speed, known_limb_gain, known_chest_gain, None)
    prompt += "\nĐọc dạng sóng độc lập. Không dùng kết luận máy, chẩn đoán viết tay hoặc các số máy in để suy ra đáp án. Được đọc nhãn chuyển đạo và chuẩn ghi. Không coi nhịp nhanh QRS rộng đồng nghĩa nhịp nhanh thất; tách mô tả, chẩn đoán phân biệt và mức độ khẩn cấp."
    signature = request_fingerprint(payload, prompt, provider, model, thinking)
    st.caption("Khi bấm Phân tích, ảnh và thông tin ca bệnh được gửi đến dịch vụ AI đã chọn. App giữ kết quả trong phiên làm việc; nên tải về trước khi đóng.")
    if st.button("Phân tích ECG", type="primary", key="analyze_v7"):
        if not api_key or not model:
            st.warning("Cần thiết lập dịch vụ và API key trong mục Thiết lập AI.")
        else:
            st.session_state.pop("last_report_v7", None)
            st.subheader("Đang phân tích")
            st.caption("Nội dung đang nhận là báo cáo AI chưa được bác sĩ kiểm tra.")
            if provider == "Google Gemini":
                generator = analyze_with_gemini_stream(api_key, model, payload, prompt, thinking)
            else:
                generator = analyze_with_openai_stream(api_key, model, base64.b64encode(payload).decode("ascii"), prompt)
            outcome = display_analysis(generator)
            st.session_state["last_report_v7"] = {**outcome, "signature": signature,
                "provider": provider, "model": model, "image_png": payload,
                "time_utc": datetime.now(timezone.utc).isoformat(),
                "image_sha256": hashlib.sha256(payload).hexdigest(),
                "app_version": APP_VERSION}
            st.rerun()

    result = st.session_state.get("last_report_v7")
    if result and result["signature"] != signature:
        st.info("Ảnh, thông tin ca bệnh hoặc thiết lập AI đã thay đổi. Bấm Phân tích để tạo báo cáo phù hợp với dữ liệu hiện tại.")
    elif result:
        st.subheader("Kết quả phân tích")
        st.caption(f"{result['provider']} · {result['model']} · {result['total_seconds']:.1f} giây")
        if not result["completed"]:
            st.error(result.get("error") or "Phân tích chưa hoàn tất.")
            if result["report"]:
                st.warning("Nội dung dưới đây bị gián đoạn, không phải báo cáo hoàn chỉnh.")
        st.markdown(result["report"] or "Chưa có nội dung trả về.")
        st.caption("Báo cáo do AI tạo; cần bác sĩ xác minh số đo, dạng sóng và kết luận.")
        if result["report"]:
            state = "API đã trả nội dung hoàn chỉnh; chưa được bác sĩ xác minh" if result["completed"] else "BÁO CÁO BỊ GIÁN ĐOẠN — KHÔNG HOÀN CHỈNH"
            download = (f"# Phân tích ECG — by dr. Lai\n\n{state}\n\n"
                f"Dịch vụ: {result['provider']} | Model: {result['model']}\n\n"
                f"Thời gian tạo (UTC): {result['time_utc']}\n\n" + result["report"])
            st.download_button("Tải báo cáo", download, "ecg_by_dr_lai.md", "text/markdown", key="download_v7")
else:
    st.caption("Tải ảnh hoặc chụp điện tâm đồ để bắt đầu.")
