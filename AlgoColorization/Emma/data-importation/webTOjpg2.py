import os
from PIL import Image

# dossier racine contenant chapitre_1, chapitre_2, ..., chapitre_400
ROOT_DIR = "one_piece"

# dossier de sortie unique
OUTPUT_DIR = "one_piece_jpg"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for chapter_name in sorted(os.listdir(ROOT_DIR)):
    chapter_path = os.path.join(ROOT_DIR, chapter_name)

    if not os.path.isdir(chapter_path):
        continue

    # on ne traite que les dossiers chapitre_X
    if not chapter_name.startswith("chapitre_"):
        continue

    chapter_number = chapter_name.replace("chapitre_", "").zfill(3)

    print(f"Traitement {chapter_name}")

    for file in sorted(os.listdir(chapter_path)):
        if not file.lower().endswith(".webp"):
            continue

        webp_path = os.path.join(chapter_path, file)

        page_number = os.path.splitext(file)[0]  # page_001
        jpg_name = f"chapitre_{chapter_number}_{page_number}_bw.jpg"
        jpg_path = os.path.join(OUTPUT_DIR, jpg_name)

        try:
            with Image.open(webp_path) as img:
                img = img.convert("RGB")
                img.save(jpg_path, "JPEG", quality=95)

        except Exception as e:
            print(f"Erreur sur {webp_path} : {e}")
