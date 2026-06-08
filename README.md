# 🚦 한국 교통표지판 실시간 인식 시스템

**딥러닝 기초 · 최종 프로젝트 (5조)**

YOLOv8 기반으로 실제 도로 이미지에서 한국 교통표지판을 **탐지 + 분류 + 음성 안내**하는 시스템입니다.
앞선 중간 발표의 GTSRB CNN **분류** 모델을, 실도로에서 동작하는 **객체 탐지** 시스템으로 확장했습니다.

| 팀원 | 학번 |
|------|------|
| 김기윤 | 2022100835 |
| 이상혁 | 2022100890 |

---

## 📌 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **모델** | YOLOv8n (3.2M params, Anchor-free) |
| **학습 방식** | COCO 사전학습 → 한국 표지판 Fine-tuning (Transfer Learning) |
| **데이터셋** | AI Hub 도로환경 파노라마, 57클래스, 17,212장 |
| **입력 해상도** | 640 × 640 |
| **학습 설정** | batch=16, 30 epochs, imgsz 640, AMP(FP16) |
| **데모** | Gradio (이미지 업로드 + 웹캠) + gTTS 한국어 음성 |
| **배포** | Hugging Face Spaces |

---

## 📊 성능 결과 (데이터 단계별)

| 지표 | Round 1 (2,276장) | Round 2 (6,200장) | Round 3 (17,212장) |
|------|:---:|:---:|:---:|
| **mAP@50** | 24.3% | 26.3% | **31.3%** |
| **Precision** | 41.5% | 60.3% | **61.1%** |
| **Recall** | 23.7% | 26.5% | **30.4%** |

> 학습 데이터를 7.6배 늘리는 동안 Precision +19.6%p, mAP +7.0%p 향상.
> 30 epoch 시점에도 mAP 상승 추세 → 추가 학습 시 50~60% 도달 가능.

학습 곡선: [`yolo/checkpoints/traffic_sign/results.png`](yolo/checkpoints/traffic_sign/results.png)

---

## 📁 폴더 구조

```
.
├── yolo/                          # ⭐ 최종 프로젝트 (YOLOv8 객체 탐지)
│   ├── prepare_dataset.py         #   AI Hub zip → YOLO 형식 변환
│   ├── add_dataset.py             #   데이터셋 단계별 추가 (중복 제외)
│   ├── train.py                   #   YOLOv8 학습 스크립트
│   ├── dataset.yaml               #   57클래스 정의
│   ├── web_demo.py                #   로컬 Gradio 데모 (웹캠 + 업로드 + TTS)
│   ├── save_metrics.py            #   성능 지표 시각화
│   └── checkpoints/traffic_sign/  #   학습 결과 (best.pt, 그래프, 혼동행렬)
│
├── hf_space/                      # Hugging Face Spaces 배포 코드
│   ├── app.py                     #   Gradio 앱 (gTTS 음성)
│   ├── requirements.txt
│   └── packages.txt
│
├── src/                           # 중간 발표 (GTSRB CNN 분류)
│   ├── model.py / dataset.py / train.py / evaluate.py / predict.py
├── main.py                        #   CNN 학습 진입점
│
├── requirements.txt
├── .gitignore
└── README.md
```

> **참고:** 학습 데이터셋(`yolo/data/` 38GB, `data/` 415MB)은 용량 문제로 깃허브에서 제외했습니다.
> AI Hub [도로환경 파노라마 이미지](https://www.aihub.or.kr) 에서 다운로드 후 `prepare_dataset.py`로 변환하면 재현 가능합니다.

---

## 🚀 실행 방법

### 1. 환경 설치
```bash
pip install -r requirements.txt
```

### 2. 데이터셋 준비 (AI Hub zip 필요)
```bash
python yolo/prepare_dataset.py     # zip → YOLO 형식 변환
python yolo/add_dataset.py         # 데이터 추가 확장
```

### 3. 학습
```bash
python yolo/train.py               # YOLOv8n 학습
```

### 4. 로컬 데모 실행
```bash
python yolo/web_demo.py            # 브라우저에서 웹캠/이미지 탐지
```

---

## 🔍 주요 기법

- **Transfer Learning** : COCO 사전학습 가중치 → 한국 표지판 미세조정
- **AMP (Automatic Mixed Precision)** : FP16 혼합 연산으로 학습 가속
- **Anchor-free Detection** : YOLOv8 Decoupled Head
- **JSON 폴리곤 → YOLO 바운딩박스** 변환 파이프라인
- **gTTS** 한국어 음성 안내

---

## ⚠️ 한계 및 향후 개선

| 원인 | 개선 방안 |
|------|-----------|
| 클래스 불균형 (평균 302장/클래스) | 희귀 클래스 오버샘플링, Focal Loss |
| 소형 표지판 탐지 한계 | 입력 해상도 640 → 1280 |
| 모델 용량 부족 (Nano) | YOLOv8s/m 업그레이드 |
| 학습 epoch 부족 (미수렴) | 30 → 100+ epochs |

목표 성능: **mAP 50~60%** (위 보완책 적용 시)

---

*딥러닝 기초 · 2026 · 5조*
