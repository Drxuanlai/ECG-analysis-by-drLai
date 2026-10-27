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
import zipfile
import uuid
import importlib.metadata
from datetime import datetime, timezone

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
APP_VERSION = "v6"
PROMPT_VERSION = "ecg-evidence-v6"
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

QUY TẮC SỐ ĐO V6
- Mọi số do AI ước lượng đều phải ghi “AI ước lượng — chưa kiểm chứng”.
- Không xác định được tốc độ giấy từ ảnh hoặc thông tin người dùng đã xác nhận:
  chỉ mô tả ô lưới/hình thái; không giả định 25 mm/s và không kết luận thời gian
  ms hoặc tần số bpm. Không tự đưa giả định chuẩn vào phần kết luận.
- Chuẩn do người dùng xác nhận chỉ xác nhận chuẩn ghi, không xác nhận số đo AI.
- Chỉ nói “hình thái gợi ý” nếu chưa xác định đủ độ rộng QRS, không loại trừ block
  chỉ từ phần đỉnh nhọn. Nếu không phân biệt được tận cùng QRS với ST/T, ghi rõ.
- Không tự ghi “Bác sĩ đã xác minh” cho bất kỳ số đo hoặc kết luận nào.

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


def trial_zip(records):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for record in records:
            directory = record["run_id"]
            public = {key: value for key, value in record.items() if key != "image_png"}
            archive.writestr(f"{directory}/record.json", json.dumps(public, ensure_ascii=False, indent=2))
            archive.writestr(f"{directory}/image_sent.png", record["image_png"])
            archive.writestr(f"{directory}/report.txt", record["report"])
            archive.writestr(f"{directory}/prompt.txt", record["prompt"])
    return buffer.getvalue()


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
        with st.expander("Chi tiết kỹ thuật"):
            st.code(str(error))

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


# ----------------- QUẢN LÝ CẤU HÌNH VÀ ĐỐI CHIẾU -----------------
def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def crop_box_percent(size, horizontal, vertical):
    w, h = size
    left, right = horizontal
    top, bottom = vertical
    if not (0 <= left < right <= 100 and 0 <= top < bottom <= 100):
        raise ValueError("Vùng cắt phải có chiều rộng và chiều cao lớn hơn 0.")
    box = (int(w * left / 100), int(h * top / 100),
           int(w * right / 100), int(h * bottom / 100))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("Vùng cắt quá nhỏ.")
    return box


def add_measurement_review(record, values, reviewer, attested, note):
    """Human review is append-only and never validates the AI narrative."""
    if not reviewer.strip() or not attested:
        raise ValueError("Cần tên người kiểm tra và xác nhận đã đối chiếu số đo.")
    present = {k: v for k, v in values.items() if v is not None}
    if not present:
        raise ValueError("Chưa có số đo để lưu.")
    if not note.strip():
        raise ValueError("Cần ghi chuyển đạo, mốc đo và cách đối chiếu; QTc cần công thức/nguồn.")
    limits = {"hr_bpm": (1, 400), "pr_ms": (1, 1000), "qrs_ms": (1, 500),
              "qt_ms": (1, 1500), "qtc_ms": (1, 1500)}
    for key, value in present.items():
        if key not in limits or not limits[key][0] <= value <= limits[key][1]:
            raise ValueError("Số đo ngoài phạm vi nhập liệu.")
    entry = {"review_id": uuid.uuid4().hex[:12], "reviewer": reviewer.strip(),
             "utc_time": datetime.now(timezone.utc).isoformat(), "values": present,
             "source": "Bác sĩ tự xác nhận đã đối chiếu", "note": note.strip(),
             "ai_report_validated": False}
    record.setdefault("measurement_reviews", []).append(entry)
    return entry


st.title("🫀 Phân tích điện tâm đồ 12 chuyển đạo")
st.caption("Bản v6 · Gemini 3.6 Flash · Cấu hình có thể khóa, số đo có nguồn")
st.info("Ứng dụng thử nghiệm. AI chưa có bộ đo tín hiệu ECG đã được kiểm chứng. Mọi số đo và chẩn đoán AI trả về đều chưa được xác minh.")

locked = st.session_state.get("experiment_lock_v6") is not None
if locked:
    st.success("Đang khóa cấu hình nhóm: " + st.session_state["experiment_lock_v6"]["name"])
    if st.button("Mở khóa để thiết lập nhóm mới", key="unlock_v6"):
        st.session_state.pop("experiment_lock_v6", None)
        st.rerun()

with st.sidebar:
    test_mode = st.checkbox("Chế độ thử nghiệm có khóa cấu hình", value=True, disabled=locked, key="test_mode_v6")
    group_name = st.text_input("Tên nhóm cấu hình", value="case01_A", disabled=locked, key="group_v6")
    st.caption("Mỗi nhóm khóa một cấu hình. Mở khóa khi chuyển A sang B hoặc đổi chuẩn ghi. Thay API key không được ghi vào nhật ký.")
    provider = st.radio("Chọn chế độ phân tích:",
        ["Google Gemini", "OpenAI", "So sánh lần lượt"], disabled=locked, key="provider_v6")
    gemini_model = GEMINI_MODEL
    gemini_thinking = st.selectbox("Mức suy luận Gemini 3.6 Flash:",
        ["medium", "low", "high"], disabled=locked, key="thinking_v6")
    st.caption("Model Google cố định: " + GEMINI_MODEL)
    active_gemini_key = get_secret("GEMINI_API_KEY", "")
    if provider != "OpenAI":
        if active_gemini_key:
            st.caption("Đang dùng Gemini key trong Secrets.")
        else:
            active_gemini_key = st.text_input("Gemini API key", type="password", key="gkey_v6")
    openai_model = "gpt-4o"
    active_openai_key = get_secret("OPENAI_API_KEY", "")
    if provider != "Google Gemini":
        openai_model = st.selectbox("Model OpenAI", ["gpt-4o", "gpt-4o-mini"], disabled=locked)
        if not active_openai_key:
            active_openai_key = st.text_input("OpenAI API key", type="password", key="okey_v6")

    st.subheader("Thiết lập lần thử")
    reading_mode = st.radio("Cách đọc:", ["Đọc sóng độc lập", "Đối chiếu cả chữ/số máy in"], disabled=locked)
    trial_label = st.text_input("Mã ca / lần thử", placeholder="VD: case01_A1", key="trial_v6")
    mask_condition = st.selectbox("Thông tin che trên ảnh (nhật ký, không tự che ảnh):", [
        "Chưa gắn nhãn", "Không che", "Che kết luận/chữ chẩn đoán",
        "Che kết luận và số đo máy", "Che thêm chữ chuẩn ghi", "Khác"])

    st.subheader("Tiền xử lý ảnh")
    enable_preprocessing = st.checkbox("Bật tiền xử lý ảnh nâng cao", value=False, disabled=locked)
    remove_shadow = apply_clahe = False
    clahe_clip, sharpen_lvl, rot_angle = 2.0, 0.0, 0
    if enable_preprocessing:
        remove_shadow = st.checkbox("Khử bóng và đều nền", value=True, disabled=locked)
        apply_clahe = st.checkbox("Tăng tương phản CLAHE", value=True, disabled=locked)
        clahe_clip = st.slider("Mức CLAHE", 1.0, 5.0, 2.0, 0.5, disabled=locked)
        sharpen_lvl = st.slider("Độ sắc nét", 0.0, 3.0, 1.0, 0.5, disabled=locked)
        st.caption("Xoay nhỏ góc bị tắt để tránh nội suy/cắt mất bờ tín hiệu. Có thể xoay 90° ở vùng ảnh.")

    st.subheader("Thông tin người bệnh")
    patient_age = st.number_input("Tuổi", 0, 120, value=None, step=1, disabled=locked)
    patient_sex = st.selectbox("Giới tính", ["Chưa cung cấp", "Nam", "Nữ", "Khác"], disabled=locked)
    clinical_history = st.text_area("Bệnh sử / triệu chứng", disabled=locked,
        placeholder="Để trống nếu chưa được cung cấp")

    st.subheader("Chuẩn ghi đã kiểm tra")
    st.caption("Không có giá trị mặc định. Trong bài test che chữ, để Chưa xác định nếu muốn thử AI độc lập.")
    speed_choice = st.selectbox("Tốc độ giấy (mm/s)", ["Chưa xác định", "25", "50", "Khác"], disabled=locked)
    known_speed = None
    if speed_choice == "Khác":
        known_speed = st.number_input("Tốc độ khác (mm/s)", 1.0, 200.0, value=None, disabled=locked)
    elif speed_choice != "Chưa xác định":
        known_speed = float(speed_choice)
    known_limb_gain = st.number_input("Chuẩn chuyển đạo chi (mm/mV)", 0.1, 100.0, value=None, disabled=locked)
    known_chest_gain = st.number_input("Chuẩn ngực đồng nhất V1–V6 (mm/mV)", 0.1, 100.0, value=None, disabled=locked)
    calibration_source = st.selectbox("Nguồn kiểm tra chuẩn ghi", ["Chưa xác định", "Đọc từ bản ECG gốc", "Cài đặt máy đã kiểm tra", "Khác"], disabled=locked)
    confirm_calibration = st.checkbox("Tôi đã kiểm tra các chuẩn ghi vừa nhập", disabled=locked)
    st.caption("Nếu V1–V6 có chuẩn khác nhau, không nhập một giá trị chung. Ghi rõ khác biệt trong bệnh sử/thông tin kỹ thuật.")

st.subheader("1. Chọn ảnh, chiều đọc và vùng ECG")
uploaded_file = st.file_uploader("Ảnh ECG", type=["jpg", "jpeg", "png", "jfif", "webp", "bmp", "heic", "heif"])
if st.checkbox("Sử dụng camera"):
    cam = st.camera_input("Chụp ECG")
    if cam is not None:
        uploaded_file = cam

final_image_to_analyze = None
payload = None
settings = None
current_config = None
if uploaded_file is not None:
    try:
        uploaded_bytes = uploaded_file.getvalue()
        upload_sha = hashlib.sha256(uploaded_bytes).hexdigest()
        original_image = ImageOps.exif_transpose(Image.open(io.BytesIO(uploaded_bytes))).convert("RGB")
    except Exception:
        st.error("Không đọc được ảnh. Hãy xuất JPEG/PNG rồi tải lại.")
        st.stop()
    auto_orientation = st.checkbox("Tự nhận diện chiều (OCR cục bộ)", value=True, disabled=locked)
    auto_angle = 0
    if auto_orientation:
        cache = st.session_state.get("orientation_result", {})
        if cache.get("image_id") != upload_sha:
            with st.spinner("Đang kiểm tra chiều chữ, không gọi API..."):
                cache = {"image_id": upload_sha, **detect_ecg_orientation(original_image)}
            st.session_state["orientation_result"] = cache
        auto_angle = cache["angle"]
        st.caption(cache["message"] + f" Góc tự xoay: {auto_angle}°.")
    manual_angle = st.selectbox("Xoay thêm ngược chiều kim đồng hồ", [0, 90, 180, 270], disabled=locked)
    oriented = rotate_quarter_turn(original_image, (auto_angle + manual_angle) % 360)
    crop_enabled = st.checkbox("Chọn vùng tờ ECG để gửi AI", value=False, disabled=locked)
    horizontal, vertical = (0, 100), (0, 100)
    if crop_enabled:
        st.caption("Tọa độ % tính trên ảnh đã xoay. Giữ đủ 12 chuyển đạo, nhãn, dải nhịp và xung chuẩn. Ảnh cắt sẽ được gửi thật vào AI.")
        horizontal = st.slider("Biên trái — phải (%)", 0, 100, (0, 100), disabled=locked)
        vertical = st.slider("Biên trên — dưới (%)", 0, 100, (0, 100), disabled=locked)
    try:
        crop_box = crop_box_percent(oriented.size, horizontal, vertical)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    raw_image = oriented.crop(crop_box)
    raw_image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
    final_image_to_analyze = preprocess_ecg(raw_image, remove_shadow, apply_clahe, clahe_clip, sharpen_lvl, 0) if enable_preprocessing else raw_image
    payload = image_to_png(final_image_to_analyze)
    image_sha = hashlib.sha256(payload).hexdigest()
    with st.expander("Xem ảnh toàn cảnh đã xoay"):
        st.image(oriented, caption="Ảnh toàn cảnh; có thể khác vùng gửi AI", width="stretch")
    before, after = st.columns(2)
    with before:
        st.image(raw_image, caption="Vùng đã chọn, trước tăng tương phản", width="stretch")
    with after:
        st.image(payload, caption="ĐÚNG ẢNH SẼ GỬI AI", width="stretch")
    st.caption(f"Ảnh gửi {final_image_to_analyze.width} × {final_image_to_analyze.height} px · SHA256 {image_sha[:16]}")
    with st.expander("Phóng to để bác sĩ xem — KHÔNG gửi thêm ảnh này"):
        zoom_x = st.slider("Vùng xem ngang (%)", 0, 100, (0, 100), key="zoom_x_v6")
        zoom_y = st.slider("Vùng xem dọc (%)", 0, 100, (0, 100), key="zoom_y_v6")
        try:
            zoom_box = crop_box_percent(final_image_to_analyze.size, zoom_x, zoom_y)
            st.image(final_image_to_analyze.crop(zoom_box), width="stretch", caption="Chỉ để xem; không thay ảnh hoặc cấu hình gửi AI")
        except ValueError:
            st.caption("Chọn vùng xem có chiều rộng và chiều cao lớn hơn 0.")

has_calibration = any(v is not None for v in (known_speed, known_limb_gain, known_chest_gain))
calibration_valid = not has_calibration or (confirm_calibration and calibration_source != "Chưa xác định")
if has_calibration and not calibration_valid:
    st.warning("Đã nhập chuẩn ghi: cần chọn nguồn và xác nhận đã kiểm tra trước khi phân tích.")
verified = {"speed_mm_s": known_speed, "limb_mm_mv": known_limb_gain,
            "chest_mm_mv": known_chest_gain, "qrs_ms": None}
full_prompt = SYSTEM_PROMPT + build_clinical_context(patient_age, patient_sex, clinical_history)
full_prompt += build_verified_context(known_speed, known_limb_gain, known_chest_gain, None)
if has_calibration:
    full_prompt += "\nNguồn chuẩn ghi do người dùng xác nhận: " + calibration_source
full_prompt += ("\nCHẾ ĐỘ ĐỌC SÓNG ĐỘC LẬP: Không dùng kết luận máy, chữ chẩn đoán viết tay hay số PR/QRS/QTc/trục/tần số máy in. Được đọc nhãn và chuẩn ghi. Đây là chỉ dẫn, không phải công cụ che chữ tự động."
                if reading_mode == "Đọc sóng độc lập" else
                "\nCHẾ ĐỘ ĐỐI CHIẾU: Tách số máy in/chữ chẩn đoán khỏi quan sát dạng sóng, không coi là đáp án đúng.")

if payload is not None:
    current_config = {
        "app_version": APP_VERSION, "prompt_version": PROMPT_VERSION,
        "provider": provider, "gemini_model": gemini_model, "thinking_level": gemini_thinking,
        "openai_model": openai_model if provider != "Google Gemini" else None,
        "reading_mode": reading_mode, "prompt_sha256": hashlib.sha256(full_prompt.encode()).hexdigest(),
        "uploaded_size": list(original_image.size), "oriented_size": list(oriented.size),
        "sent_size": list(final_image_to_analyze.size), "max_image_edge": MAX_IMAGE_EDGE,
        "auto_orientation_enabled": auto_orientation, "auto_rotation_ccw": auto_angle,
        "manual_rotation_ccw": manual_angle,
        "crop": {"enabled": crop_enabled, "horizontal_percent": list(horizontal),
                 "vertical_percent": list(vertical), "box_pixels": list(crop_box)},
        "preprocessing": {"enabled": enable_preprocessing, "remove_shadow": remove_shadow,
            "clahe": apply_clahe, "clahe_clip": clahe_clip, "sharpen": sharpen_lvl, "fine_rotation": 0},
        "verified_values": verified, "calibration_source": calibration_source if has_calibration else None,
        "calibration_confirmed": bool(has_calibration and calibration_valid),
        "clinical_context": {"age": patient_age, "sex": patient_sex, "history": clinical_history},
        "generation_options": {"Gemini": gemini_config(gemini_model, gemini_thinking).model_dump(mode="json", exclude_none=True),
                               "OpenAI": {"max_tokens": 2500, "temperature": 0.2}},
    }
    current_hash = config_hash(current_config)
    st.subheader("2. Kiểm tra và khóa cấu hình")
    with st.expander("Chi tiết cấu hình hiện tại"):
        st.json(current_config)
    lock = st.session_state.get("experiment_lock_v6")
    if test_mode and not locked:
        if st.button("Khóa cấu hình nhóm này", disabled=not calibration_valid):
            if not group_name.strip():
                st.warning("Cần tên nhóm cấu hình.")
            else:
                st.session_state["experiment_lock_v6"] = {"name": group_name.strip(),
                    "config_id": uuid.uuid4().hex[:12], "config_sha256": current_hash,
                    "locked_utc": datetime.now(timezone.utc).isoformat(), "config": current_config}
                st.rerun()
    lock_matches = bool(lock and lock["config_sha256"] == current_hash)
    if locked and not lock_matches:
        st.error("Ảnh hoặc cấu hình hiện tại khác nhóm đã khóa (ví dụ kích thước/góc OCR). Không gửi API. Hãy mở khóa và tạo nhóm mới.")
    if lock:
        st.caption(f"Mã cấu hình: {lock['config_id']} · {lock['config_sha256'][:16]}")
        st.download_button("Tải cấu hình nhóm JSON", json.dumps(lock, ensure_ascii=False, indent=2),
                           "ecg_config_v6.json", "application/json")
        st.caption("Cấu hình có thể chứa bệnh sử đã nhập. Không chứa API key. Khóa chỉ tồn tại trong phiên hiện tại; tải nhật ký trước khi đóng.")
    confirmed_image = st.checkbox("Tôi đã xem đúng ảnh sẽ gửi, đủ vùng cần đọc và đúng chiều", key="confirmed_" + image_sha)
    can_analyze = calibration_valid and confirmed_image and (not test_mode or (locked and lock_matches))
    st.subheader("3. Phân tích")
    st.warning("Số đo trong báo cáo AI = ước lượng chưa kiểm chứng. Xác nhận chuẩn ghi không xác nhận các số đo hay chẩn đoán AI.")
    if st.button("🚀 Bắt đầu phân tích ECG", type="primary", disabled=not can_analyze, key="analyze_v6"):
        if test_mode and not trial_label.strip():
            st.warning("Cần mã ca/lần thử để lưu nhật ký.")
        elif ((provider != "OpenAI" and not active_gemini_key) or
              (provider != "Google Gemini" and not active_openai_key)):
            st.warning("Thiếu API key của dịch vụ đã chọn.")
        elif len(st.session_state.get("ecg_runs_v6", [])) >= 8:
            st.warning("Đã đủ 8 lượt. Tải ZIP và xóa nhật ký trước khi chạy tiếp; app không tự xóa lượt cũ.")
        elif provider == "So sánh lần lượt" and len(st.session_state.get("ecg_runs_v6", [])) > 6:
            st.warning("So sánh cần 2 vị trí nhật ký trống. Hãy tải ZIP và xóa nhật ký trước.")
        else:
            settings = {**current_config, "configuration": current_config,
                "trial_label": trial_label.strip(), "mask_condition": mask_condition,
                "uploaded_file_sha256": upload_sha, "config_sha256": current_hash,
                "experiment": lock if test_mode else None, "measurement_reviews": [],
                "ai_measurement_status": "Chưa được bác sĩ xác minh", "image_confirmed_by_user": True}
            services = []
            if provider != "OpenAI":
                services.append(("Gemini", gemini_model, active_gemini_key))
            if provider != "Google Gemini":
                services.append(("OpenAI", openai_model, active_openai_key))
            for service, model, api_key in services:
                st.subheader(f"Báo cáo AI chưa xác minh ({model})")
                if service == "Gemini":
                    generator = analyze_with_gemini_stream(api_key, model, payload, full_prompt, gemini_thinking)
                else:
                    generator = analyze_with_openai_stream(api_key, model, base64.b64encode(payload).decode("ascii"), full_prompt)
                outcome = display_analysis(generator)
                record = {**settings, **outcome, "measurement_reviews": [], "run_id": uuid.uuid4().hex[:12],
                    "utc_time": datetime.now(timezone.utc).isoformat(), "service": service,
                    "requested_model": model, "thinking_level": gemini_thinking if service == "Gemini" else None,
                    "generation_options": current_config["generation_options"][service],
                    "prompt": full_prompt, "image_png": payload, "image_sha256": image_sha,
                    "sdk_versions": {p: importlib.metadata.version(p) for p in ("google-genai", "openai", "streamlit")}}
                st.session_state.setdefault("ecg_runs_v6", []).append(record)
else:
    st.caption("Tải ảnh để chọn vùng ECG, kiểm tra ảnh gửi và khóa cấu hình.")

records = st.session_state.get("ecg_runs_v6", [])
if st.session_state.get("ecg_runs_v5"):
    st.download_button("Tải nhật ký v5 còn trong phiên", trial_zip(st.session_state["ecg_runs_v5"]), "ecg_trials_v5.zip", "application/zip")
if records:
    st.subheader("4. Nhật ký và số đo bác sĩ đối chiếu")
    st.caption("Tối đa 8 lượt/phiên. ZIP chứa ảnh, bệnh sử, cấu hình và kết quả; không chứa API key. Tải trước khi đóng phiên hoặc Reboot.")
    st.dataframe([{"Lượt": r["trial_label"], "ID": r["run_id"], "Model": r["requested_model"],
        "Cấu hình": r["config_sha256"][:12], "Tiền xử lý": r["preprocessing"]["enabled"],
        "Hoàn tất API": r["completed"], "Giây": r["total_seconds"], "Ảnh": r["image_sha256"][:12]}
        for r in records], hide_index=True)
    index = st.selectbox("Chọn lượt để đối chiếu", range(len(records)),
        format_func=lambda i: f"{records[i]['trial_label']} · {records[i]['run_id']}", key="selected_run_v6")
    selected = records[index]
    rid = selected["run_id"]
    with st.expander("Ảnh thực sự đã gửi của lượt được chọn"):
        st.image(selected["image_png"], width="stretch")
    st.warning("Báo cáo AI: toàn bộ số đo/diễn giải chưa kiểm chứng, kể cả khi AI tự viết 'có cơ sở'.")
    if not selected["completed"]:
        st.error("Luồng API không hoàn tất. " + (selected.get("error") or ""))
    st.markdown(selected["report"] or "Không có nội dung trả về.")
    values = selected["verified_values"]
    source_rows = [{"Thông tin": label, "Giá trị": str(values.get(key)) if values.get(key) is not None else "Chưa xác định",
        "Nguồn / trạng thái": "Người dùng xác nhận trước phân tích" if values.get(key) is not None else "Chưa được người dùng xác nhận"}
        for key, label in [("speed_mm_s", "Tốc độ giấy (mm/s)"), ("limb_mm_mv", "Chuẩn chi (mm/mV)"), ("chest_mm_mv", "Chuẩn ngực (mm/mV)")]]
    st.dataframe(source_rows, hide_index=True)
    with st.expander("Bác sĩ ghi số đo đã tự đối chiếu — không xác nhận toàn bộ báo cáo"):
        st.caption("Nhập kết quả tự đo/đối chiếu với ECG gốc. Để trống số chưa kiểm tra. Các lần lưu được giữ lại trong lịch sử; đây là ghi nhận tự xác nhận, không xác thực danh tính bác sĩ.")
        with st.form("review_" + rid):
            reviewer = st.text_input("Tên bác sĩ kiểm tra")
            hr = st.number_input("Tần số đã kiểm tra (bpm)", 1, 400, value=None)
            pr = st.number_input("PR đã kiểm tra (ms)", 1, 1000, value=None)
            qrs = st.number_input("QRS đã kiểm tra (ms)", 1, 500, value=None)
            qt = st.number_input("QT đã kiểm tra (ms)", 1, 1500, value=None)
            qtc = st.number_input("QTc đã kiểm tra (ms)", 1, 1500, value=None)
            note = st.text_area("Chuyển đạo, mốc đo, tốc độ và cách đo; công thức QTc nếu có")
            attested = st.checkbox("Tôi là bác sĩ đã đối chiếu các số vừa nhập với ECG gốc")
            save_review = st.form_submit_button("Lưu lần đối chiếu số đo")
        if save_review:
            try:
                add_measurement_review(selected, {"hr_bpm": hr, "pr_ms": pr, "qrs_ms": qrs, "qt_ms": qt, "qtc_ms": qtc}, reviewer, attested, note)
                st.success("Đã lưu số đo đối chiếu riêng. Nội dung AI và trạng thái chưa xác minh được giữ nguyên.")
            except ValueError as error:
                st.error(str(error))
    for review in selected.get("measurement_reviews", []):
        st.caption(f"Người tự xác nhận: {review['reviewer']} · {review['utc_time']} (UTC)")
        st.json(review)
    st.download_button("Tải ZIP: ảnh gửi + cấu hình + prompt + báo cáo + đối chiếu", trial_zip(records), "ecg_trials_v6.zip", "application/zip", key="zip_v6")
    if st.button("Xóa nhật ký trong phiên sau khi đã tải ZIP", key="clear_v6"):
        st.session_state.pop("ecg_runs_v6", None)
        st.session_state.pop("selected_run_v6", None)
        st.rerun()
