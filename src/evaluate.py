import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from .model import TrafficSignCNN
from .dataset import get_dataloaders, CLASS_NAMES


@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(loader, desc="예측 중"):
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = outputs.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, save_path: str = "confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("예측", fontsize=12)
    ax.set_ylabel("실제", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix 저장: {save_path}")


def plot_training_history(history: dict, save_path: str = "training_history.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"], label="Val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_acc"], label="Train")
    ax2.plot(history["val_acc"], label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"학습 곡선 저장: {save_path}")


def evaluate_model(
    data_root: str,
    checkpoint_path: str,
    batch_size: int = 64,
    num_workers: int = 2,
    output_dir: str = ".",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, test_loader = get_dataloaders(data_root, batch_size, num_workers)

    model = TrafficSignCNN(num_classes=43).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"체크포인트 로드: {checkpoint_path} (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    preds, labels, probs = get_predictions(model, test_loader, device)

    accuracy = (preds == labels).mean()
    print(f"\n전체 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")

    print("\n클래스별 분류 리포트:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES, digits=4))

    os.makedirs(output_dir, exist_ok=True)
    plot_confusion_matrix(labels, preds, os.path.join(output_dir, "confusion_matrix.png"))

    return accuracy, preds, labels
