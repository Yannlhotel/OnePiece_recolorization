import os
import shutil
import random
from collections import defaultdict

ROOT = "dataset"
IMG_DIR = os.path.join(ROOT, "images")
MASK_DIR = os.path.join(ROOT, "bw")

OUT_TRAIN = os.path.join(ROOT, "train")
OUT_VAL = os.path.join(ROOT, "val")

for d in [
    os.path.join(OUT_TRAIN, "images"),
    os.path.join(OUT_TRAIN, "bw"),
    os.path.join(OUT_VAL, "images"),
    os.path.join(OUT_VAL, "bw"),
]:
    os.makedirs(d, exist_ok=True)

# --- regrouper par chapitre ---
files = sorted(os.listdir(IMG_DIR))
chapters = defaultdict(list)

for f in files:
    # chapitre_001_page_001.tif → 001
    chap = f.split("_")[1]
    chapters[chap].append(f)

# --- split 80/20 ---
chap_list = sorted(chapters.keys())
random.seed(42)
random.shuffle(chap_list)

split_idx = int(0.8 * len(chap_list))
train_chaps = set(chap_list[:split_idx])
val_chaps = set(chap_list[split_idx:])

# --- déplacement des fichiers ---
for chap, files in chapters.items():
    target = OUT_TRAIN if chap in train_chaps else OUT_VAL

    for f in files:
        shutil.move(
            os.path.join(IMG_DIR, f),
            os.path.join(target, "images", f)
        )
        shutil.move(
            os.path.join(MASK_DIR, f),
            os.path.join(target, "bw", f)
        )

print(f"Train: {len(train_chaps)} chapitres")
print(f"Val: {len(val_chaps)} chapitres")
