import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
BEST_MODEL = os.path.join(CHECKPOINT_DIR, "best_model.pth")


def cmd_train(args):
    from src.train import train
    import json

    history = train(
        data_root=DATA_ROOT,
        checkpoint_dir=CHECKPOINT_DIR,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.workers,
    )

    history_path = os.path.join(CHECKPOINT_DIR, "history.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"학습 기록 저장: {history_path}")


def cmd_evaluate(args):
    from src.evaluate import evaluate_model, plot_training_history
    import json

    evaluate_model(
        data_root=DATA_ROOT,
        checkpoint_path=BEST_MODEL,
        batch_size=args.batch_size,
        num_workers=args.workers,
        output_dir=CHECKPOINT_DIR,
    )

    history_path = os.path.join(CHECKPOINT_DIR, "history.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
        plot_training_history(history, os.path.join(CHECKPOINT_DIR, "training_history.png"))


def cmd_predict(args):
    from src.predict import predict_image

    predict_image(
        image_path=args.image,
        checkpoint_path=BEST_MODEL,
        speak=not args.no_speak,
    )


def main():
    parser = argparse.ArgumentParser(description="GTSRB 교통 표지판 분류")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="모델 학습")
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--workers", type=int, default=2)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="모델 평가")
    p_eval.add_argument("--batch-size", type=int, default=64)
    p_eval.add_argument("--workers", type=int, default=2)

    # predict
    p_pred = sub.add_parser("predict", help="이미지 예측 + 음성 안내")
    p_pred.add_argument("image", help="예측할 이미지 경로")
    p_pred.add_argument("--no-speak", action="store_true", help="음성 안내 비활성화")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)


if __name__ == "__main__":
    main()
