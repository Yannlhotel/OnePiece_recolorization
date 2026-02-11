import os
from PIL import Image

ROOT_DIR = "one_piece"

OUT_ROOT = "one_piece_tiff"
OUT_COLOR = os.path.join(OUT_ROOT, "images")
OUT_BW = os.path.join(OUT_ROOT, "bw")

os.makedirs(OUT_COLOR, exist_ok=True)
os.makedirs(OUT_BW, exist_ok=True)

for chapter_name in sorted(os.listdir(ROOT_DIR)):
    chapter_path = os.path.join(ROOT_DIR, chapter_name)

    if not os.path.isdir(chapter_path):
        continue

    if not chapter_name.startswith("chapitre_"):
        continue

    chapter_number = chapter_name.replace("chapitre_", "").zfill(3)
    print(f"Traitement {chapter_name}")

    for file in sorted(os.listdir(chapter_path)):
        if not file.lower().endswith(".webp"):
            continue

        webp_path = os.path.join(chapter_path, file)
        page_number = os.path.splitext(file)[0]

        base_name = f"chapitre_{chapter_number}_{page_number}.tif"

        color_path = os.path.join(OUT_COLOR, base_name)
        bw_path = os.path.join(OUT_BW, base_name)

        try:
            with Image.open(webp_path) as img:
                # --- COULEUR LAB ---
                img_lab = img.convert("LAB")
                img_lab.save(
                    color_path,
                    format="TIFF",
                    compression="tiff_deflate"
                )

                # --- NOIR & BLANC ---
                img_bw = img.convert("L")
                img_bw.save(
                    bw_path,
                    format="TIFF",
                    compression="tiff_deflate"
                )

        except Exception as e:
            print(f"Erreur sur {webp_path} : {e}")
