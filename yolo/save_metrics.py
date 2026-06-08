import os, csv, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

CKPT_DIR = Path(r"C:\Users\lkkmj\딥러닝 기초\프로젝트\yolo\checkpoints")
csv_path = CKPT_DIR / "traffic_sign" / "results.csv"

# CSV 파싱
h = {"epochs":[], "map50":[], "map50_95":[], "precision":[], "recall":[], "box_loss":[], "cls_loss":[]}
with open(csv_path, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        vals = {k.strip(): v.strip() for k,v in row.items()}
        try:
            h["epochs"].append(int(float(vals.get("epoch",0))))
            h["map50"].append(float(vals.get("metrics/mAP50(B)",0)))
            h["map50_95"].append(float(vals.get("metrics/mAP50-95(B)",0)))
            h["precision"].append(float(vals.get("metrics/precision(B)",0)))
            h["recall"].append(float(vals.get("metrics/recall(B)",0)))
            h["box_loss"].append(float(vals.get("train/box_loss",0)))
            h["cls_loss"].append(float(vals.get("train/cls_loss",0)))
        except: pass

NAVY  = "#1E2761"
TEAL  = "#00B4D8"
GREEN = "#22C55E"
AMBER = "#F59E0B"
RED   = "#EF4444"
ICE   = "#CADCFс"
LGRAY = "#F1F5F9"

fig = plt.figure(figsize=(18, 12))
fig.patch.set_facecolor(NAVY)
gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.35)

# ── 타이틀 ──
fig.text(0.5, 0.97, "YOLOv8 학습 성능 지표 요약",
         ha="center", va="top", fontsize=20, fontweight="bold", color="white")
fig.text(0.5, 0.93, "한국 교통 표지판 탐지 모델 (AI Hub 도로환경 파노라마 데이터셋 | 57클래스 | 2,276장)",
         ha="center", va="top", fontsize=11, color=TEAL)

# ── 상단 카드 4개 ──
cards = [
    ("mAP@50",       f"{max(h['map50']):.3f}",   f"({max(h['map50'])*100:.1f}%)", "탐지 정확도 핵심 지표", TEAL),
    ("Precision",    f"{max(h['precision']):.3f}",f"({max(h['precision'])*100:.1f}%)", "탐지 시 정답 비율", GREEN),
    ("Recall",       f"{max(h['recall']):.3f}",   f"({max(h['recall'])*100:.1f}%)", "전체 표지판 중 탐지 비율", AMBER),
    ("mAP@50-95",    f"{max(h['map50_95']):.3f}", f"({max(h['map50_95'])*100:.1f}%)", "엄격한 기준의 정확도", "#A78BFA"),
]
for i, (title, val, pct, desc, color) in enumerate(cards):
    ax = fig.add_subplot(gs[0, i])
    ax.set_facecolor("#0F172A")
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis("off")
    # 색 강조선
    ax.axhline(y=0.92, xmin=0.05, xmax=0.95, color=color, linewidth=3)
    ax.text(0.5, 0.73, title, ha="center", va="center", fontsize=13, fontweight="bold", color="white")
    ax.text(0.5, 0.45, val,   ha="center", va="center", fontsize=26, fontweight="bold", color=color)
    ax.text(0.5, 0.22, pct,   ha="center", va="center", fontsize=13, color=color, alpha=0.8)
    ax.text(0.5, 0.06, desc,  ha="center", va="center", fontsize=9,  color="#94A3B8")

# ── 학습 전후 비교 카드 ──
ax_cmp = fig.add_subplot(gs[1, 0])
ax_cmp.set_facecolor("#0F172A")
ax_cmp.axis("off")
ax_cmp.text(0.5, 0.93, "학습 전 → 후 비교", ha="center", fontsize=12,
            fontweight="bold", color="white", transform=ax_cmp.transAxes)
rows = [
    ("mAP@50",    f"{h['map50'][0]:.3f}",    f"{max(h['map50']):.3f}",    f"+{(max(h['map50'])-h['map50'][0]):.3f}"),
    ("Precision", f"{h['precision'][0]:.3f}", f"{max(h['precision']):.3f}", f"+{(max(h['precision'])-h['precision'][0]):.3f}"),
    ("Recall",    f"{h['recall'][0]:.3f}",    f"{max(h['recall']):.3f}",    f"+{(max(h['recall'])-h['recall'][0]):.3f}"),
]
y = 0.76
for name, before, after, diff in rows:
    ax_cmp.text(0.05, y, name,   ha="left",   fontsize=9,  color="#94A3B8", transform=ax_cmp.transAxes)
    ax_cmp.text(0.38, y, before, ha="center", fontsize=9,  color="#EF4444", transform=ax_cmp.transAxes)
    ax_cmp.text(0.55, y, "→",   ha="center", fontsize=9,  color="white",   transform=ax_cmp.transAxes)
    ax_cmp.text(0.70, y, after,  ha="center", fontsize=9,  color=GREEN,     transform=ax_cmp.transAxes)
    ax_cmp.text(0.92, y, diff,   ha="right",  fontsize=9,  color=TEAL,      transform=ax_cmp.transAxes)
    y -= 0.16
ax_cmp.text(0.05, 0.08, "* 1에폭 → 최고 성능 기준", ha="left", fontsize=8, color="#64748B", transform=ax_cmp.transAxes)

# ── 학습 설정 카드 ──
ax_cfg = fig.add_subplot(gs[1, 1])
ax_cfg.set_facecolor("#0F172A")
ax_cfg.axis("off")
ax_cfg.text(0.5, 0.93, "학습 설정", ha="center", fontsize=12,
            fontweight="bold", color="white", transform=ax_cfg.transAxes)
settings = [
    ("모델",     "YOLOv8n (Nano)"),
    ("Epochs",  "30"),
    ("Batch",   "16"),
    ("Img size","640×640"),
    ("Classes", "57개"),
    ("Train",   "6,200장"),
    ("Val",     "1,552장"),
]
y = 0.78
for k, v in settings:
    ax_cfg.text(0.08, y, k+":", ha="left", fontsize=9, color="#94A3B8", transform=ax_cfg.transAxes)
    ax_cfg.text(0.92, y, v,     ha="right",fontsize=9, color=TEAL,      transform=ax_cfg.transAxes)
    y -= 0.10

# ── mAP50 곡선 ──
ax1 = fig.add_subplot(gs[1, 2:])
ax1.set_facecolor("#0F172A")
ax1.plot(h["epochs"], h["map50"], color=TEAL, linewidth=2.5, label="mAP@50")
ax1.plot(h["epochs"], h["map50_95"], color="#A78BFA", linewidth=2, linestyle="--", label="mAP@50-95")
ax1.axhline(y=max(h["map50"]), color=GREEN, linewidth=1, linestyle=":", alpha=0.7)
ax1.set_title("mAP 학습 곡선", color="white", fontsize=12, fontweight="bold", pad=8)
ax1.set_xlabel("Epoch", color="#94A3B8")
ax1.set_ylabel("mAP", color="#94A3B8")
ax1.tick_params(colors="#94A3B8")
ax1.spines[:].set_color("#334155")
ax1.legend(facecolor="#0F172A", labelcolor="white", fontsize=9)
ax1.grid(True, alpha=0.15, color="white")
ax1.set_facecolor("#0F172A")
ax1.annotate(f"최고 {max(h['map50']):.3f}", xy=(h["epochs"][h["map50"].index(max(h["map50"]))], max(h["map50"])),
             xytext=(5, 8), textcoords="offset points", color=GREEN, fontsize=9)

# ── Loss 곡선 ──
ax2 = fig.add_subplot(gs[2, :2])
ax2.set_facecolor("#0F172A")
ax2.plot(h["epochs"], h["box_loss"], color=AMBER, linewidth=2, label="Box Loss")
ax2.plot(h["epochs"], h["cls_loss"], color=RED, linewidth=2, label="Cls Loss")
ax2.set_title("Loss 학습 곡선 (낮을수록 좋음)", color="white", fontsize=12, fontweight="bold", pad=8)
ax2.set_xlabel("Epoch", color="#94A3B8")
ax2.set_ylabel("Loss", color="#94A3B8")
ax2.tick_params(colors="#94A3B8")
ax2.spines[:].set_color("#334155")
ax2.legend(facecolor="#0F172A", labelcolor="white", fontsize=9)
ax2.grid(True, alpha=0.15, color="white")

# ── Precision / Recall 곡선 ──
ax3 = fig.add_subplot(gs[2, 2:])
ax3.set_facecolor("#0F172A")
ax3.plot(h["epochs"], h["precision"], color=GREEN, linewidth=2, label="Precision")
ax3.plot(h["epochs"], h["recall"],    color=AMBER, linewidth=2, label="Recall")
ax3.set_title("Precision / Recall 곡선", color="white", fontsize=12, fontweight="bold", pad=8)
ax3.set_xlabel("Epoch", color="#94A3B8")
ax3.set_ylabel("Score", color="#94A3B8")
ax3.tick_params(colors="#94A3B8")
ax3.spines[:].set_color("#334155")
ax3.legend(facecolor="#0F172A", labelcolor="white", fontsize=9)
ax3.grid(True, alpha=0.15, color="white")

out = CKPT_DIR / "metrics_summary.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=NAVY)
print(f"저장: {out}")

# JSON 별도 저장
summary = {
    "모델": "YOLOv8n",
    "데이터셋": "AI Hub 도로환경 파노라마 이미지",
    "클래스 수": 57,
    "학습 이미지": 6200,
    "검증 이미지": 1552,
    "에폭": 30,
    "배치": 16,
    "이미지 크기": 640,
    "최고_mAP50": round(max(h["map50"]), 4),
    "최고_mAP50_95": round(max(h["map50_95"]), 4),
    "최고_Precision": round(max(h["precision"]), 4),
    "최고_Recall": round(max(h["recall"]), 4),
    "초기_mAP50 (1에폭)": round(h["map50"][0], 4),
    "향상_mAP50": round(max(h["map50"]) - h["map50"][0], 4),
}
with open(CKPT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("JSON 저장:", CKPT_DIR / "metrics_summary.json")
print("\n=== 지표 요약 ===")
for k, v in summary.items():
    print(f"  {k}: {v}")