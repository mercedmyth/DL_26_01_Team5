import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from ultralytics import YOLO

if __name__ == "__main__":
    BASE      = Path(r"C:\Users\lkkmj\딥러닝 기초\프로젝트\yolo")
    YAML_PATH = BASE / "dataset.yaml"
    CKPT_DIR  = BASE / "checkpoints"
    CKPT_DIR.mkdir(exist_ok=True)

    print("YOLOv8n 로드...")
    model = YOLO("yolov8n.pt")

    print("학습 시작 — epochs=30, imgsz=640, batch=16")
    model.train(
        data=str(YAML_PATH),
        epochs=30,
        imgsz=640,
        batch=16,
        workers=0,
        project=str(CKPT_DIR),
        name="traffic_sign",
        exist_ok=True,
        patience=10,
        verbose=True,
    )

    # 학습 기록 저장
    result_dir = CKPT_DIR / "traffic_sign"
    csv_path   = result_dir / "results.csv"
    history    = {"epochs": [], "map50": [], "precision": [], "recall": [], "box_loss": []}

    if csv_path.exists():
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    vals = {k.strip(): v.strip() for k, v in row.items()}
                    history["epochs"].append(int(float(vals.get("epoch", 0))))
                    history["map50"].append(float(vals.get("metrics/mAP50(B)", 0)))
                    history["precision"].append(float(vals.get("metrics/precision(B)", 0)))
                    history["recall"].append(float(vals.get("metrics/recall(B)", 0)))
                    history["box_loss"].append(float(vals.get("train/box_loss", 0)))
                except:
                    pass
        with open(CKPT_DIR / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"학습 완료! 베스트 모델: {result_dir / 'weights' / 'best.pt'}")