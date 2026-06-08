import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


CLASS_NAMES = [
    "속도 제한 20km/h", "속도 제한 30km/h", "속도 제한 50km/h", "속도 제한 60km/h",
    "속도 제한 70km/h", "속도 제한 80km/h", "속도 제한 80km/h 해제", "속도 제한 100km/h",
    "속도 제한 120km/h", "추월 금지", "3.5톤 초과 차량 추월 금지", "교차로 우선",
    "우선도로", "양보", "정지", "차량 진입 금지",
    "3.5톤 초과 차량 진입 금지", "진입 금지", "일반 위험", "왼쪽 커브 위험",
    "오른쪽 커브 위험", "이중 커브 위험", "노면 울퉁불퉁", "미끄러운 노면",
    "도로 우측 협소", "도로 공사 중", "신호등", "보행자",
    "어린이 보호구역", "자전거 주의", "빙결/눈 주의", "야생동물 주의",
    "모든 제한 해제", "우회전", "좌회전", "직진",
    "직진 또는 우회전", "직진 또는 좌회전", "우측 통행", "좌측 통행",
    "로터리", "추월 금지 해제", "3.5톤 초과 차량 추월 금지 해제",
]

IMG_SIZE = 32

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]),
])

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3337, 0.3064, 0.3171], std=[0.2672, 0.2564, 0.2629]),
])


class GTSRBDataset(Dataset):
    def __init__(self, csv_path: str, data_root: str, transform=None):
        self.data_root = data_root
        self.transform = transform
        df = pd.read_csv(csv_path)
        # Train.csv: Path, ClassId / Test.csv: Path, ClassId
        self.paths = df["Path"].tolist()
        self.labels = df["ClassId"].tolist()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_root, self.paths[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def get_dataloaders(data_root: str, batch_size: int = 64, num_workers: int = 2):
    import torch
    pin = torch.cuda.is_available()

    train_csv = os.path.join(data_root, "Train.csv")
    test_csv = os.path.join(data_root, "Test.csv")

    train_dataset = GTSRBDataset(train_csv, data_root, transform=TRAIN_TRANSFORM)
    test_dataset = GTSRBDataset(test_csv, data_root, transform=VAL_TRANSFORM)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin,
    )
    return train_loader, test_loader
