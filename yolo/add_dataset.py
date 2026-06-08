import os, sys, json, zipfile, random
from pathlib import Path
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.stdout.reconfigure(encoding="utf-8")

BASE      = Path(r"C:\Users\lkkmj\딥러닝 기초\프로젝트\yolo")
LABEL_ZIP = Path(r"C:\Users\lkkmj\Downloads\도로환경 파노라마 이미지\Training\라벨링데이터.zip")
IMG_ZIPS  = [
    Path(r"C:\Users\lkkmj\Downloads\도로환경 파노라마 이미지\Training\01-01.zip"),
    Path(r"C:\Users\lkkmj\Downloads\도로환경 파노라마 이미지\Training\01-02.zip"),
]
OUT_DIR     = BASE / "data"
SIGN_PREFIX = "교통안전표지판"
MAX_ADD     = 18000
IMG_SIZE    = 1000

def polygon_to_bbox(coord_xy):
    xs, ys = coord_xy[0], coord_xy[1]
    return min(xs), min(ys), max(xs), max(ys)

def bbox_to_yolo(x1, y1, x2, y2, w, h):
    cx = ((x1+x2)/2)/w; cy = ((y1+y2)/2)/h
    bw = (x2-x1)/w;     bh = (y2-y1)/h
    return cx, cy, bw, bh

def main():
    random.seed(42)

    # 두 zip의 이미지 인덱스 합산
    print("이미지 zip 인덱싱 중...")
    img_map = {}
    for zip_path in IMG_ZIPS:
        with zipfile.ZipFile(zip_path) as zf:
            for n in zf.namelist():
                if n.endswith('.jpg'):
                    img_map[Path(n).name] = (zip_path, n)
    available = set(img_map.keys())
    print(f"새 이미지: {len(available):,}개")

    # 기존 이미지 (중복 방지)
    existing = set()
    for split in ["train","val"]:
        for f in (OUT_DIR/"images"/split).glob("*.jpg"):
            existing.add(f.name)
    print(f"기존 이미지: {len(existing):,}개")

    # 라벨링 스캔
    print("라벨링 스캔 중...")
    records, class_set = [], set()
    with zipfile.ZipFile(LABEL_ZIP) as z:
        json_names = [n for n in z.namelist() if n.endswith('.json')]
        for name in tqdm(json_names, desc="스캔"):
            try:
                with z.open(name) as f:
                    data = json.load(f)
            except: continue
            img_fname = data["interface"]["filename"]
            if img_fname not in available or img_fname in existing:
                continue
            anns = []
            for ann in data["interface"].get("annotations",[]):
                cls = ann["class_name"]
                if not cls.startswith(SIGN_PREFIX): continue
                try:
                    x1,y1,x2,y2 = polygon_to_bbox(ann["coord_xy"])
                    anns.append((cls,x1,y1,x2,y2))
                    class_set.add(cls)
                except: pass
            if anns:
                records.append((img_fname, anns))
            if len(records) >= MAX_ADD * 2:
                break

    print(f"추가 가능: {len(records):,}개")
    if len(records) > MAX_ADD:
        records = random.sample(records, MAX_ADD)

    random.shuffle(records)
    split_idx = int(len(records)*0.8)
    splits = {"train": records[:split_idx], "val": records[split_idx:]}

    # 기존 클래스 로드
    yaml_path = BASE / "dataset.yaml"
    classes = []
    with open(yaml_path, encoding="utf-8") as f:
        in_names = False
        for line in f:
            if line.strip() == "names:": in_names=True; continue
            if in_names and line.startswith("  - "):
                classes.append(line.strip()[3:].strip("'"))
    class_to_id = {c:i for i,c in enumerate(classes)}

    # 이미지/라벨 추출
    for split_name, recs in splits.items():
        img_dir = OUT_DIR/"images"/split_name
        lbl_dir = OUT_DIR/"labels"/split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{split_name} 처리 ({len(recs)}개)...")
        extracted = 0
        # zip별로 묶어서 처리 (효율)
        from collections import defaultdict
        by_zip = defaultdict(list)
        for img_fname, anns in recs:
            zip_path, zip_inner = img_map[img_fname]
            by_zip[zip_path].append((img_fname, zip_inner, anns))

        for zip_path, items in by_zip.items():
            with zipfile.ZipFile(zip_path) as zf:
                for img_fname, zip_inner, anns in tqdm(items, desc=f"{split_name}/{zip_path.name}"):
                    with zf.open(zip_inner) as src:
                        (img_dir/img_fname).write_bytes(src.read())
                    extracted += 1
                    stem = Path(img_fname).stem
                    lines = []
                    for cls,x1,y1,x2,y2 in anns:
                        cid = class_to_id.get(cls)
                        if cid is None: continue
                        cx,cy,bw,bh = bbox_to_yolo(x1,y1,x2,y2,IMG_SIZE,IMG_SIZE)
                        cx=max(0.,min(1.,cx)); cy=max(0.,min(1.,cy))
                        bw=max(0.001,min(1.,bw)); bh=max(0.001,min(1.,bh))
                        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    if lines:
                        (lbl_dir/f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"  추출: {extracted}개")

    for split in ["train","val"]:
        cnt = len(list((OUT_DIR/"images"/split).glob("*.jpg")))
        print(f"{split}: {cnt:,}장")
    print("\n추가 완료!")

if __name__ == "__main__":
    main()