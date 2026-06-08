import os, cv2, numpy as np, tempfile
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS

MODEL_PATH = "best.pt"
CONF = 0.15

print("모델 로딩 중...")
model = YOLO(MODEL_PATH)
NAMES = [model.names[i] for i in sorted(model.names)]
print(f"모델 로딩 완료! 클래스: {len(NAMES)}개")

def get_font(size=18):
    for fp in [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]:
        if os.path.exists(fp):
            try: return ImageFont.truetype(fp, size)
            except: pass
    return ImageFont.load_default()

font_label = get_font(18)

def make_tts(text: str):
    """gTTS로 음성 파일 생성, 경로 반환"""
    try:
        tts = gTTS(text=text, lang="ko")
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tts.save(tmp.name)
        return tmp.name
    except:
        return None

def draw_boxes(frame_bgr):
    results = model(frame_bgr, verbose=False, conf=CONF)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    detected = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls  = NAMES[int(box.cls[0])]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        short = cls.split("/")[-1]
        label = f"{short}  {conf:.0%}"
        draw.rectangle([x1, y1, x2, y2], outline=(0, 180, 216), width=2)
        try:
            bbox = draw.textbbox((x1, y1), label, font=font_label)
            tw, th = bbox[2]-bbox[0], bbox[3]-bbox[1]
        except:
            tw, th = 120, 18
        draw.rectangle([x1, y1-th-8, x1+tw+8, y1], fill=(30, 39, 97))
        draw.text((x1+4, y1-th-4), label, font=font_label, fill=(255, 255, 255))
        detected.append((cls, conf))
    return np.array(img), detected

def predict_image(img):
    if img is None:
        return None, "이미지를 업로드해주세요.", None
    frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    result_arr, detected = draw_boxes(frame_bgr)

    if detected:
        lines = [f"✅  {cls.split('/')[-1]}  (신뢰도: {conf:.0%})" for cls, conf in detected]
        # 음성 안내: 탐지된 표지판 이름 읽어주기
        speak_text = ", ".join([cls.split("/")[-1] for cls, _ in detected[:3]])  # 최대 3개
        audio_path = make_tts(speak_text)
        return result_arr, "\n".join(lines), audio_path
    return result_arr, "❌  표지판을 탐지하지 못했습니다.", None

def predict_webcam(frame):
    if frame is None:
        return None
    frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    result_arr, _ = draw_boxes(frame_bgr)
    return result_arr

with gr.Blocks(title="교통 표지판 인식 데모") as demo:
    gr.Markdown("""
# 🚦 한국 교통 표지판 인식 데모
**YOLOv8 | AI Hub 도로환경 파노라마 | 57클래스**
    """)

    with gr.Tabs():
        with gr.Tab("📁 이미지 업로드"):
            with gr.Row():
                with gr.Column():
                    img_input  = gr.Image(label="이미지 업로드", type="pil")
                    btn_detect = gr.Button("🔍  탐지하기", variant="primary", size="lg")
                with gr.Column():
                    img_output = gr.Image(label="탐지 결과")
                    txt_output = gr.Textbox(label="탐지된 표지판 목록", lines=6)
                    audio_out  = gr.Audio(label="🔊 음성 안내", autoplay=True)
            btn_detect.click(
                predict_image,
                inputs=img_input,
                outputs=[img_output, txt_output, audio_out]
            )

        with gr.Tab("📷 웹캠 실시간"):
            gr.Markdown("웹캠을 연결하면 실시간으로 탐지합니다.")
            with gr.Row():
                webcam_in  = gr.Image(label="웹캠", sources=["webcam"], streaming=True, type="numpy")
                webcam_out = gr.Image(label="실시간 탐지 결과")
            webcam_in.stream(predict_webcam, inputs=webcam_in, outputs=webcam_out)

    gr.Markdown("---\n딥러닝 기초 프로젝트  |  이상혁 2022100890  |  김기윤 2022100835")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
