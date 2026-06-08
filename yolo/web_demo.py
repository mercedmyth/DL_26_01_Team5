import os, sys, threading, cv2, numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

import gradio as gr
from ultralytics import YOLO
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CKPT  = Path(r"C:\Users\lkkmj\딥러닝 기초\프로젝트\yolo\checkpoints\traffic_sign\weights\best.pt")
CONF  = 0.15
FONT_PATH = r"C:\Windows\Fonts\malgun.ttf"

model = YOLO(str(CKPT))
NAMES = [model.names[i] for i in sorted(model.names)]

try:
    font_label = ImageFont.truetype(FONT_PATH, 18)
    font_title = ImageFont.truetype(FONT_PATH, 22)
except:
    font_label = ImageFont.load_default()
    font_title = font_label

def draw_boxes_pil(frame_bgr):
    """PIL로 한글 텍스트 그리기"""
    results = model(frame_bgr, verbose=False, conf=CONF)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(img)
    detected = []

    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls  = NAMES[int(box.cls[0])]
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        short = cls.split("/")[-1]
        label = f"{short}  {conf:.0%}"

        # 바운딩박스
        draw.rectangle([x1,y1,x2,y2], outline=(0,180,216), width=2)

        # 라벨 배경
        bbox = draw.textbbox((x1, y1), label, font=font_label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.rectangle([x1, y1-th-8, x1+tw+8, y1], fill=(30,39,97))
        draw.text((x1+4, y1-th-4), label, font=font_label, fill=(255,255,255))
        detected.append((cls, conf))

    return np.array(img), detected

def speak(text):
    def _run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            for v in engine.getProperty("voices"):
                if "korean" in v.name.lower() or "ko" in v.id.lower():
                    engine.setProperty("voice", v.id); break
            engine.say(text); engine.runAndWait()
        except: pass
    threading.Thread(target=_run, daemon=True).start()

# ── 이미지 업로드 ──
def predict_image(img):
    if img is None:
        return None, "이미지를 업로드해주세요."
    frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    result_arr, detected = draw_boxes_pil(frame_bgr)
    if detected:
        lines = [f"✅  {cls.split('/')[-1]}  (신뢰도: {conf:.0%})" for cls,conf in detected]
        speak("  ".join([cls.split('/')[-1] for cls,_ in detected]))
        return result_arr, "\n".join(lines)
    return result_arr, "❌  표지판을 탐지하지 못했습니다."

# ── 웹캠 실시간 ──
def predict_webcam(frame):
    if frame is None: return None
    frame_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    result_arr, _ = draw_boxes_pil(frame_bgr)
    return result_arr

# ── UI ──
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
                    txt_output = gr.Textbox(label="탐지된 표지판 목록", lines=8)
            btn_detect.click(predict_image, inputs=img_input, outputs=[img_output, txt_output])

        with gr.Tab("📷 웹캠 실시간"):
            gr.Markdown("웹캠을 연결하면 실시간으로 탐지합니다. 'Turn on' 버튼을 눌러주세요.")
            with gr.Row():
                webcam_in  = gr.Image(label="웹캠", sources=["webcam"], streaming=True, type="numpy")
                webcam_out = gr.Image(label="실시간 탐지 결과")
            webcam_in.stream(predict_webcam, inputs=webcam_in, outputs=webcam_out)

    gr.Markdown("---\n딥러닝 기초 프로젝트  |  이상혁 2022100890  |  김기윤 2022100835")

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=False, theme=gr.themes.Base())