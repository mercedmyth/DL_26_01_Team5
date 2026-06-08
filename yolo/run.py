"""
YOLO 프로젝트 실행 스크립트
사용법:
  python yolo/run.py prepare   # 데이터셋 준비
  python yolo/run.py train     # 학습
  python yolo/run.py plot      # 학습 곡선 그래프
  python yolo/run.py webcam    # 웹캠 데모
  python yolo/run.py image <파일경로>  # 이미지 데모
"""
import os, sys
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == "prepare":
        from yolo.prepare_dataset import main as prep
        prep()

    elif cmd == "train":
        from yolo.train import train
        train()

    elif cmd == "plot":
        from yolo.plot_results import plot_training_curves
        plot_training_curves()

    elif cmd == "webcam":
        os.system(f"{sys.executable} yolo/demo.py --mode webcam")

    elif cmd == "image":
        if len(sys.argv) < 3:
            print("사용법: python yolo/run.py image <이미지경로>")
            return
        os.system(f'{sys.executable} yolo/demo.py --mode image --file "{sys.argv[2]}"')

    else:
        print(f"알 수 없는 명령어: {cmd}")
        print(__doc__)

if __name__ == "__main__":
    main()