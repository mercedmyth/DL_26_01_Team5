"""
실시간 데모 스크립트
사용법:
  웹캠: python yolo/demo.py --mode webcam
  이미지: python yolo/demo.py --mode image --file <경로>
"""
import os, sys, argparse, threading, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from pathlib import Path

CKPT       = Path(r"C:\Users\lkkmj\딥러닝 기초\프로젝트\yolo\checkpoints\traffic_sign\weights\best.pt")
CONF       = 0.15
COLOR_BOX  = (0, 180, 216)
COLOR_TEXT = (255, 255, 255)
COLOR_BG   = (30, 39, 97)


def speak_async(text):
    def _run():
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
            for v in engine.getProperty("voices"):
                if "korean" in v.name.lower() or "ko" in v.id.lower():
                    engine.setProperty("voice", v.id)
                    break
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[TTS] {e}")
    threading.Thread(target=_run, daemon=True).start()


def draw_boxes(frame, results, names):
    detected = []
    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls  = names[int(box.cls[0])]
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_BOX, 2)
        label = f"{cls.split('/')[-1]}  {conf:.2f}"
        (tw,th),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), COLOR_BG, -1)
        cv2.putText(frame, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1)
        detected.append(cls)
    return frame, detected


def run_webcam(model, names):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("웹캠을 열 수 없습니다. 카메라가 연결됐는지 확인하세요.")
        return
    print("웹캠 시작! 'q' 누르면 종료")
    cooldown = {}
    while True:
        ret, frame = cap.read()
        if not ret: break
        results = model(frame, verbose=False, conf=CONF)
        frame, detected = draw_boxes(frame, results, names)
        now = time.time()
        for cls in set(detected):
            if now - cooldown.get(cls, 0) > 3.0:
                speak_async(cls.split("/")[-1])
                cooldown[cls] = now
        cv2.putText(frame, f"표지판: {len(detected)}개  |  q: 종료", (10,25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BOX, 2)
        cv2.imshow("교통 표지판 실시간 인식", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


def run_image(model, names, path):
    frame = cv2.imread(path)
    if frame is None:
        print(f"이미지를 열 수 없습니다: {path}")
        return
    results = model(frame, verbose=False, conf=CONF)
    frame, detected = draw_boxes(frame, results, names)

    print(f"\n탐지 결과: {len(detected)}개")
    for i, cls in enumerate(detected, 1):
        short = cls.split("/")[-1]
        print(f"  {i}. {cls}")
        speak_async(short)

    # 결과 이미지 저장 (창 없이)
    stem = Path(path).stem
    out  = Path(path).parent / f"{stem}_result.jpg"
    cv2.imwrite(str(out), frame)
    print(f"\n결과 저장: {out}")

    # 창 표시 (가능한 경우)
    try:
        cv2.imshow("탐지 결과 (아무 키나 누르면 종료)", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["webcam","image"], default="webcam")
    parser.add_argument("--file", default="")
    parser.add_argument("--model", default=str(CKPT))
    args = parser.parse_args()

    from ultralytics import YOLO
    print(f"모델 로드: {args.model}")
    model = YOLO(args.model)
    names = [model.names[i] for i in sorted(model.names)]

    if args.mode == "webcam":
        run_webcam(model, names)
    else:
        if not args.file:
            print("--file 경로를 지정하세요.")
        else:
            run_image(model, names, args.file)