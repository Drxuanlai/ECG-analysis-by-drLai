"""Single-lead trace extraction prototype. No API, learned weights or ECG diagnosis.
Deploy separately from app.py. Uses existing Streamlit/Pillow/NumPy/OpenCV only.
"""
import csv
import hashlib
import io
import json
import time
from datetime import datetime, timezone

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass

VERSION = "Digitizer 0.1"
METHOD = "dark-ink-dp-v1"
MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_SOURCE_PIXELS = 25_000_000
MAX_WORK_WIDTH = 2000
MAX_WORK_HEIGHT = 800


def png_bytes(image):
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def orient(image, angle):
    turns = {90:Image.Transpose.ROTATE_90,180:Image.Transpose.ROTATE_180,270:Image.Transpose.ROTATE_270}
    return image.transpose(turns[angle]) if angle in turns else image.copy()


def crop_region(image, x_range, y_range):
    left,right = x_range
    top,bottom = y_range
    if not (0 <= left < right <= 100 and 0 <= top < bottom <= 100):
        raise ValueError("Vùng chọn cần có chiều rộng và chiều cao lớn hơn 0.")
    box = (int(image.width*left/100),int(image.height*top/100),
           int(image.width*right/100),int(image.height*bottom/100))
    if box[2]-box[0] < 32 or box[3]-box[1] < 16:
        raise ValueError("Vùng quá nhỏ. Chọn một đoạn có nhiều nhát tim và đủ toàn bộ chiều cao sóng.")
    return image.crop(box),box


def roi_preview(image, box):
    preview = image.copy()
    preview.thumbnail((1500,1000),Image.Resampling.LANCZOS)
    sx,sy = preview.width/image.width,preview.height/image.height
    draw = ImageDraw.Draw(preview)
    draw.rectangle(tuple((v*sx if i%2==0 else v*sy) for i,v in enumerate(box)),outline="#e53935",width=3)
    return preview


def prepare_working_image(crop):
    work = crop.convert("RGB").copy()
    work.thumbnail((MAX_WORK_WIDTH,MAX_WORK_HEIGHT),Image.Resampling.LANCZOS)
    return work


def ink_mask(image, max_gray=185, local_contrast=10, suppress_color=True, remove_grid=True):
    rgb = np.asarray(image.convert("RGB"))
    # A black trace stays dark while colored grid pixels are lightened.
    gray = np.max(rgb,axis=2).astype(np.uint8) if suppress_color else cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    h,w = gray.shape
    block = min(41,min(h,w) if min(h,w)%2 else min(h,w)-1)
    block = max(3,block)
    adaptive = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV,block,float(local_contrast)) > 0
    original = adaptive & (gray < max_gray)
    removed = np.zeros_like(original)
    if remove_grid:
        # Remove only very long straight structures. May also remove ECG segments;
        # those pixels are not silently reconstructed/interpolated.
        horizontal = cv2.morphologyEx(original.astype(np.uint8),cv2.MORPH_OPEN,
                                      np.ones((1,max(24,int(w*.65))),np.uint8)) > 0
        vertical = cv2.morphologyEx(original.astype(np.uint8),cv2.MORPH_OPEN,
                                    np.ones((max(12,int(h*.88)),1),np.uint8)) > 0
        removed = horizontal | vertical
    return original & ~removed,gray,removed


def trace_candidates(mask, gray, continuity=.12, seed_percent=50.):
    """Find a low-cost ink path, then reject unsupported/ambiguous samples.

    This path is an image-processing candidate, not a confirmed ECG signal.
    No spline, gap fill, waveform synthesis or physical unit conversion.
    """
    h,w = mask.shape
    if not np.any(mask):
        raise ValueError("Không có đủ nét tối để dò. Chọn vùng khác hoặc chỉnh ngưỡng tối.")
    rows = np.arange(h,dtype=np.float32)
    indices = np.arange(h,dtype=np.int32)
    parent = np.empty((w,h),dtype=np.uint16)
    intensity = gray.astype(np.float32)/255
    cost = np.where(mask[:,0],intensity[:,0],20.) + continuity*.2*np.abs(rows-seed_percent*(h-1)/100)
    parent[0] = indices
    for x in range(1,w):
        forward_raw = cost-continuity*rows
        forward_min = np.minimum.accumulate(forward_raw)
        forward_parent = np.maximum.accumulate(np.where(forward_raw <= forward_min,indices,0))
        forward = forward_min+continuity*rows
        backward_raw = (cost+continuity*rows)[::-1]
        backward_min = np.minimum.accumulate(backward_raw)
        backward_parent_rev = np.maximum.accumulate(np.where(backward_raw <= backward_min,indices,0))
        backward_parent = (h-1-backward_parent_rev)[::-1]
        backward = backward_min[::-1]-continuity*rows
        use_forward = forward <= backward
        parent[x] = np.where(use_forward,forward_parent,backward_parent)
        cost = np.minimum(forward,backward)+np.where(mask[:,x],intensity[:,x],20.)
    path = np.empty(w,dtype=np.int32)
    path[-1] = int(np.argmin(cost))
    for x in range(w-1,0,-1):
        path[x-1] = parent[x,path[x]]
    supported = mask[path,np.arange(w)]
    run_count = mask[0].astype(np.int32)+np.sum(mask[1:] & ~mask[:-1],axis=0)
    # A nearly vertical ink stroke may contain many possible y values at one x.
    # Do not pretend one chosen y uniquely recovers its peak.
    vertical_span = np.zeros(w,dtype=np.int32)
    for x,y in enumerate(path):
        if not supported[x]:
            continue
        lo,hi = y,y
        while lo > 0 and mask[lo-1,x]: lo -= 1
        while hi+1 < h and mask[hi+1,x]: hi += 1
        vertical_span[x] = hi-lo+1
        # Center of the selected connected ink run reduces contour-edge bias.
        # Long near-vertical runs remain ambiguous and are rejected below.
        path[x] = (lo+hi)//2
    status = np.full(w,"ung_vien_net_toi",dtype=object)
    status[~supported] = "khong_co_net_tai_diem_do"
    status[supported & (run_count > 3)] = "nhieu_net_trong_cot"
    status[supported & (vertical_span > max(10,int(h*.14)))] = "net_doc_mo_ho"
    status[supported & ((path <= 1) | (path >= h-2))] = "cham_bien_vung_cat"
    accepted = status == "ung_vien_net_toi"
    values = path.astype(float)
    values[~accepted] = np.nan
    return {"y":values,"status":status,"accepted":accepted,
            "support_fraction":float(np.mean(accepted)),"run_count":run_count,
            "vertical_span":vertical_span}


def trace_overlay(image, values):
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    previous = None
    for x,y in enumerate(values):
        if not np.isfinite(y):
            previous = None
            continue
        point = (x,float(y))
        if previous is not None and abs(y-previous[1]) <= max(12,image.height*.12):
            draw.line((previous,point),fill="#e60073",width=2)
        else:
            draw.ellipse((x-1,y-1,x+1,y+1),fill="#e60073")
        previous = point
    return overlay


def pixels_csv(result, crop_box, work_size, crop_size):
    # Map to the oriented image; NOT back to the camera's pre-EXIF coordinates.
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(["x_work_px","y_work_px_candidate","x_oriented_px","y_oriented_px_candidate","status"])
    ww,wh = work_size
    cw,ch = crop_size
    for x,y in enumerate(result["y"]):
        x_source = crop_box[0]+(x+.5)*cw/ww-.5
        y_source = crop_box[1]+(y+.5)*ch/wh-.5 if np.isfinite(y) else None
        writer.writerow([x,round(float(y),3) if np.isfinite(y) else "",round(x_source,3),
                         round(float(y_source),3) if y_source is not None else "",result["status"][x]])
    return out.getvalue().encode("utf-8-sig")


def analyze_crop(crop, params):
    work = prepare_working_image(crop)
    mask,gray,removed = ink_mask(work,params["max_gray"],params["local_contrast"],params["suppress_color"],params["remove_grid"])
    trace = trace_candidates(mask,gray,params["continuity"],params["seed_percent"])
    statuses,counts = np.unique(trace["status"],return_counts=True)
    return {"work":work,"mask":mask,"removed":removed,**trace,
            "counts":dict(zip(statuses.tolist(),counts.tolist()))}


def clear_result():
    st.session_state.pop("digitizer_result",None)


def main():
    st.set_page_config(page_title="ECG Digitizer — by dr. Lai",page_icon="📈",layout="wide")
    st.title("📈 ECG Digitizer — by dr. Lai")
    st.caption(VERSION+" · Thử nghiệm trích nét của MỘT chuyển đạo · Không cần API")
    st.info("Đường màu là nét tối ứng viên do xử lý ảnh tìm được, chưa được xác nhận là tín hiệu ECG. Bản này chưa đo PR/QRS/ST, chưa chẩn đoán và chưa dùng mô hình số hóa có trọng số.")
    st.caption("Ảnh được xử lý trên máy chủ Streamlit của app. Code này không gửi ảnh đến Gemini/GPT hay dịch vụ số hóa bên ngoài; không ghi ảnh vào đĩa hay bộ nhớ dùng chung giữa các phiên.")
    uploaded = st.file_uploader("Tải ảnh ECG",type=["jpg","jpeg","png","webp","heic","heif"],key="dg_upload",on_change=clear_result)
    if uploaded is None:
        st.write("Bắt đầu với ảnh thẳng, rõ. Chọn một chuyển đạo, bỏ chữ và xung chuẩn khỏi vùng dò, giữ đủ chiều cao P–QRS–T.")
        return
    blob = uploaded.getvalue()
    if len(blob) > MAX_FILE_BYTES:
        st.error("Ảnh vượt 20 MB. Xuất lại JPEG/PNG nhỏ hơn rồi tải lại.")
        return
    try:
        source = Image.open(io.BytesIO(blob))
        if source.width*source.height > MAX_SOURCE_PIXELS:
            raise ValueError("Ảnh vượt 25 triệu pixel; xuất lại ảnh nhỏ hơn.")
        source = ImageOps.exif_transpose(source).convert("RGB")
    except Exception as exc:
        st.error("Không đọc được ảnh. Thử xuất JPEG/PNG. " + (str(exc) if isinstance(exc,ValueError) else ""))
        return
    sha = hashlib.sha256(blob).hexdigest()
    suffix = sha[:12]
    st.subheader("1. Chọn đúng một chuyển đạo")
    rotation = st.selectbox("Xoay ngược chiều kim đồng hồ",[0,90,180,270],key="dg_rot_"+suffix)
    oriented = orient(source,rotation)
    frame = suffix+str(rotation)
    lead = st.selectbox("Nhãn chuyển đạo do em chọn",["II","I","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6","Chưa xác định"],key="dg_lead_"+frame)
    x_range = st.slider("Vùng dò: trái — phải (%)",0.,100.,(0.,95.),.1,key="dg_roi_x_"+frame)
    y_range = st.slider("Vùng dò: trên — dưới (%)",0.,100.,(40.,58.),.1,key="dg_roi_y_"+frame)
    try:
        crop,box = crop_region(oriented,x_range,y_range)
    except ValueError as exc:
        st.error(str(exc));return
    st.image(roi_preview(oriented,box),caption="Khung đỏ là vùng sẽ dò. Kiểm tra cả đỉnh R và đáy S không bị cắt.",width="stretch")
    st.image(crop,caption="Ảnh gốc của vùng chọn. Nên có 2–5 nhát, một hàng sóng; tránh chữ, hàng sóng khác và xung chuẩn.",width="stretch")
    st.caption("Nhãn do người dùng chọn, app chưa tự xác nhận đúng chuyển đạo. Vùng quá cao có thể khiến thuật toán bám nhầm hàng khác.")
    st.subheader("2. Trích nét thử nghiệm")
    with st.expander("Thiết lập xử lý ảnh (chỉ chỉnh khi nét bị sai)"):
        suppress_color = st.checkbox("Nét đen: giảm lưới màu",value=True,key="dg_color_"+frame)
        remove_grid = st.checkbox("Thử gỡ đường lưới thẳng dài",value=True,key="dg_grid_"+frame)
        st.caption("Gỡ lưới có thể làm mất cả đoạn ECG trùng lưới. Hãy thử tắt nếu đường màu bị đứt; không tự điền đoạn đã mất.")
        max_gray = st.slider("Ngưỡng tối tối đa",30,245,185,key="dg_gray_"+frame)
        local_contrast = st.slider("Tương phản cục bộ tối thiểu",2,35,10,key="dg_contrast_"+frame)
        continuity = st.slider("Ưu tiên đường liên tục",.02,.4,.12,.01,key="dg_cont_"+frame)
        seed_percent = st.slider("Gợi ý độ cao bắt đầu trong vùng (%)",0.,100.,50.,1.,key="dg_seed_"+frame)
    params = dict(max_gray=max_gray,local_contrast=local_contrast,suppress_color=suppress_color,
                  remove_grid=remove_grid,continuity=continuity,seed_percent=seed_percent)
    signature = hashlib.sha256(json.dumps([sha,rotation,lead,box,params,VERSION],sort_keys=True).encode()).hexdigest()
    if st.button("Trích đường ứng viên",type="primary",key="dg_run"):
        clear_result()
        started = time.monotonic()
        try:
            with st.spinner("Đang xử lý ảnh tại app, không gọi API..."):
                candidate = analyze_crop(crop,params)
            work = candidate["work"]
            metadata = {"version":VERSION,"method":METHOD,"validation_status":"unverified_candidate",
                "time_utc":datetime.now(timezone.utc).isoformat(),"source_sha256":sha,
                "original_exif_corrected_size":list(source.size),"rotation_ccw":rotation,
                "oriented_size":list(oriented.size),"crop_box_oriented_pixels":list(box),
                "crop_size":list(crop.size),"working_size":list(work.size),"chosen_lead":lead,
                "parameters":params,"column_status_counts":candidate["counts"],
                "candidate_column_fraction":candidate["support_fraction"],
                "seconds":round(time.monotonic()-started,3),
                "coordinates":"CSV coordinates refer to working crop and oriented image; not milliseconds or mV.",
                "limits":"Dark ink is not proven to be ECG. No gap filling. No P/QRS/T landmarks. No clinical validation."}
            st.session_state["digitizer_result"] = {"signature":signature,"metadata":metadata,
                "original_png":png_bytes(work),"overlay_png":png_bytes(trace_overlay(work,candidate["y"])),
                "mask_png":png_bytes(Image.fromarray(candidate["mask"].astype(np.uint8)*255)),
                "csv":pixels_csv(candidate,box,work.size,crop.size)}
        except (ValueError,cv2.error) as exc:
            st.error(str(exc))
    result = st.session_state.get("digitizer_result")
    if not result:
        return
    if result["signature"] != signature:
        st.info("Ảnh, vùng chọn hoặc thiết lập đã đổi. Kết quả cũ không áp dụng; bấm trích lại.")
        return
    st.subheader("3. Đối chiếu với ảnh gốc")
    tabs = st.tabs(["Đường màu phủ lên ảnh","Ảnh gốc để so sánh","Mặt nạ nét tối"])
    with tabs[0]:st.image(result["overlay_png"],caption="Màu hồng: ứng viên nằm trên nét tối. Khoảng trống: không trả điểm; chưa có nghĩa phần còn lại đã đúng ECG.",width="stretch")
    with tabs[1]:st.image(result["original_png"],width="stretch")
    with tabs[2]:st.image(result["mask_png"],caption="Trắng: nét còn lại sau xử lý. Có thể còn lưới/chữ hoặc mất đoạn sóng thật.",width="stretch")
    meta = result["metadata"]
    st.write(f"Thời gian xử lý: {meta['seconds']:.2f} giây · Cột có ứng viên: {meta['candidate_column_fraction']:.1%}")
    st.caption("Tỷ lệ trên chỉ là số cột có điểm ứng viên, KHÔNG phải độ chính xác hay mức tin cậy lâm sàng.")
    if meta["column_status_counts"].get("net_doc_mo_ho",0):
        st.warning("Có nét dọc mơ hồ đã để trống; đỉnh R hoặc đáy S có thể chưa được trích. Tỷ lệ cột có ứng viên cao không bù được việc bỏ sót phần sóng quan trọng.")
    if meta["candidate_column_fraction"] < .7:
        st.warning("Nhiều cột chưa có ứng viên. Chọn vùng sát hơn hoặc điều chỉnh gỡ lưới/ngưỡng; không dùng để đo.")
    st.dataframe([{"Trạng thái kỹ thuật":k,"Số cột":v} for k,v in meta["column_status_counts"].items()],hide_index=True)
    st.warning("Dừng ở bước kiểm tra đường nét. Nếu đường màu bám lưới, chữ, bỏ đỉnh R/đáy S hoặc nhảy sang sóng khác, kết quả này chưa đạt. Điểm nằm trên nét tối vẫn có thể không phải ECG.")
    st.subheader("4. Tải kết quả thử nghiệm — từng file riêng")
    st.caption("CSV chỉ chứa tọa độ pixel ứng viên. Cột không chắc để trống, không nội suy. Ảnh có thể chứa thông tin ca bệnh; che định danh trước khi tải lên.")
    st.download_button("Tải ảnh phủ màu",result["overlay_png"],"digitizer_overlay.png","image/png",key="dg_overlay_download")
    st.download_button("Tải tọa độ CSV",result["csv"],"digitizer_candidates.csv","text/csv",key="dg_csv_download")
    st.download_button("Tải cấu hình JSON",json.dumps(meta,ensure_ascii=False,indent=2),"digitizer_record.json","application/json",key="dg_json_download")


if __name__ == "__main__":
    main()
