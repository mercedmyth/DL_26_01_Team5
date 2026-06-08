"""
JSON 라벨 → YOLO 형식 변환 + 이미지 선별 추출 (수정판)
zip에서 직접 읽기, 이미지 zip에 실제 존재하는 파일만 처리
"""
import os, sys, json, zipfile, random
from pathlib import Path
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE      = Path(r"C:\Users\lkkmj\딥러닝 기초\프로젝트\yolo")
LABEL_ZIP = Path(r"C:\Users\lkkmj\Downloads\도로환경 파노라마 이미지\Training\라벨링데이터.zip")
IMG_ZIP   = Path(r"C:\Users\lkkmj\Downloads\도로환경 파노라마 이미지\Training\00-01.zip")
OUT_DIR   = BASE / "data"

SIGN_PREFIX = "교통안전표지판"
MAX_IMAGES  = 10000
IMG_SIZE    = 1000

def polygon_to_bbox(coord_xy):
    xs, ys = coord_xy[0], coord_xy[1]
    return min(xs), min(ys), max(xs), max(ys)

def bbox_to_yolo(x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh

def collect_annotations(available_imgs):
    """이미지 zip에 실제 있는 파일만 처리"""
    print(f"라벨링 스캔 중... (이미지 {len(available_imgs):,}개 기준)")
    class_set = set()
    records = []

    with zipfile.ZipFile(LABEL_ZIP, 'r') as z:
        json_names = [n for n in z.namelist() if n.endswith('.json')]
        print(f"총 JSON: {len(json_names):,}개")

        for name in tqdm(json_names, desc="스캔"):
            try:
                with z.open(name) as f:
                    data = json.load(f)
            except:
                continue

            iface    = data["interface"]
            img_fname = iface["filename"]

            # 이미지 zip에 실제 존재하는 파일만
            if img_fname not in available_imgs:
                continue

            anns = []
            for ann in iface.get("annotations", []):
                cls = ann["class_name"]
                if not cls.startswith(SIGN_PREFIX):
                    continue
                try:
                    x1, y1, x2, y2 = polygon_to_bbox(ann["coord_xy"])
                    anns.append((cls, x1, y1, x2, y2))
                    class_set.add(cls)
                except:
                    pass

            if anns:
                records.append((img_fname, anns))

            if len(records) >= MAX_IMAGES * 2:
                break

    print(f"교통 표지판 포함 이미지: {len(records):,}개 / 클래스: {len(class_set)}개")
    return records, sorted(class_set)

def build_dataset(records, classes, img_map):
    class_to_id = {c: i for i, c in enumerate(classes)}

    if len(records) > MAX_IMAGES:
        random.seed(42)
        records = random.sample(records, MAX_IMAGES)

    random.seed(42)
    random.shuffle(records)
    split    = int(len(records) * 0.8)
    splits   = {"train": records[:split], "val": records[split:]}

    with zipfile.ZipFile(IMG_ZIP, 'r') as zf:
        for split_name, recs in splits.items():
            img_dir = OUT_DIR / "images" / split_name
            lbl_dir = OUT_DIR / "labels" / split_name
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n{split_name} 처리 중 ({len(recs)}개)...")
            extracted = 0
            for img_fname, anns in tqdm(recs, desc=split_name):
                # 이미지 추출
                zip_path = img_map.get(img_fname)
                if zip_path:
                    with zf.open(zip_path) as src:
                        (img_dir / img_fname).write_bytes(src.read())
                    extracted += 1

                # YOLO 라벨 생성
                stem  = Path(img_fname).stem
                lines = []
                for cls, x1, y1, x2, y2 in anns:
                    cid = class_to_id[cls]
                    cx, cy, bw, bh = bbox_to_yolo(x1, y1, x2, y2, IMG_SIZE, IMG_SIZE)
                    cx  = max(0.0, min(1.0, cx))
                    cy  = max(0.0, min(1.0, cy))
                    bw  = max(0.001, min(1.0, bw))
                    bh  = max(0.001, min(1.0, bh))
                    lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")

            print(f"  이미지 추출: {extracted}/{len(recs)}개")

def save_yaml(classes):
    lines = [
        f"path: {OUT_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(classes)}",
        "names:",
    ]
    for cls in classes:
        lines.append(f"  - '{cls}'")
    (BASE / "dataset.yaml").write_text("\n".join(lines), encoding="utf-8")
    print(f"dataset.yaml 저장 완료")

def main():
    random.seed(42)

    # 이미지 zip에 있는 파일 목록 수집
    print("이미지 zip 인덱싱 중...")
    with zipfile.ZipFile(IMG_ZIP, 'r') as zf:
        img_map = {Path(n).name: n for n in zf.namelist() if n.endswith('.jpg')}
    available_imgs = set(img_map.keys())
    print(f"이미지 zip 내 파일: {len(available_imgs):,}개")

    records, classes = collect_annotations(available_imgs)

    if not records:
        print("매칭되는 교통 표지판 데이터가 없습니다!")
        return

    build_dataset(records, classes, img_map)
    save_yaml(classes)

    print("\n데이터셋 준비 완료!")
    print(f"클래스 {len(classes)}개:")
    for i, c in enumerate(classes):
        print(f"  {i:3d}: {c}")

if __name__ == "__main__":
    main()