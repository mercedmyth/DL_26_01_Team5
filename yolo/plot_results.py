import os, json, csv
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

# 한글 폰트 설정 (맑은 고딕)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE     = Path(r"C:\Users\lkkmj\딥러닝 기초\프로젝트\yolo")
CKPT_DIR = BASE / "checkpoints"

def load_history():
    csv_path = CKPT_DIR / "traffic_sign" / "results.csv"
    h = {"epochs": [], "map50": [], "precision": [], "recall": [], "box_loss": []}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vals = {k.strip(): v.strip() for k, v in row.items()}
            try:
                h["epochs"].append(int(float(vals.get("epoch", 0))))
                h["map50"].append(float(vals.get("metrics/mAP50(B)", 0)))
                h["precision"].append(float(vals.get("metrics/precision(B)", 0)))
                h["recall"].append(float(vals.get("metrics/recall(B)", 0)))
                h["box_loss"].append(float(vals.get("train/box_loss", 0)))
            except:
                pass
    return h

h = load_history()
epochs = h["epochs"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("YOLOv8 학습 곡선 — 한국 교통 표지판", fontsize=14, fontweight="bold")

# mAP50
ax = axes[0]
ax.plot(epochs, h["map50"], color="#00B4D8", linewidth=2, marker="o", markersize=3)
ax.axhline(y=h["map50"][0], color="gray", linestyle="--", alpha=0.6, label=f"초기: {h['map50'][0]:.3f}")
ax.axhline(y=max(h["map50"]), color="#22C55E", linestyle="--", alpha=0.6, label=f"최고: {max(h['map50']):.3f}")
ax.set_title("mAP@50 (학습 전→후)")
ax.set_xlabel("Epoch")
ax.set_ylabel("mAP@50")
ax.legend()
ax.grid(True, alpha=0.3)

# Precision
ax = axes[1]
ax.plot(epochs, h["precision"], color="#1E2761", linewidth=2, marker="o", markersize=3)
ax.axhline(y=max(h["precision"]), color="#22C55E", linestyle="--", alpha=0.6, label=f"최고: {max(h['precision']):.3f}")
ax.set_title("Precision")
ax.set_xlabel("Epoch")
ax.set_ylabel("Precision")
ax.legend()
ax.grid(True, alpha=0.3)

# Recall
ax = axes[2]
ax.plot(epochs, h["recall"], color="#F59E0B", linewidth=2, marker="o", markersize=3)
ax.axhline(y=max(h["recall"]), color="#22C55E", linestyle="--", alpha=0.6, label=f"최고: {max(h['recall']):.3f}")
ax.set_title("Recall")
ax.set_xlabel("Epoch")
ax.set_ylabel("Recall")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out = CKPT_DIR / "training_curves.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"그래프 저장: {out}")

# 최종 성능 출력
print(f"\n최종 성능:")
print(f"  mAP@50    : {max(h['map50']):.3f}  ({max(h['map50'])*100:.1f}%)")
print(f"  Precision : {max(h['precision']):.3f}")
print(f"  Recall    : {max(h['recall']):.3f}")
print(f"  학습 전(1에폭) mAP: {h['map50'][0]:.3f}")
print(f"  학습 후(최고) mAP: {max(h['map50']):.3f}")