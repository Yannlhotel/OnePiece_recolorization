import os
from PIL import Image

def main():
    # SOURCE : Dossier temporaire
    ROOT_DIR = "data/temp_scraping"
    
    # DESTINATION INITIALE : Tout va dans TRAIN d'abord
    OUT_ROOT = "data/train"
    OUT_COLOR = os.path.join(OUT_ROOT, "color")
    OUT_GRAY = os.path.join(OUT_ROOT, "gray")

    os.makedirs(OUT_COLOR, exist_ok=True)
    os.makedirs(OUT_GRAY, exist_ok=True)

    if not os.path.exists(ROOT_DIR):
        print(f"Dossier source '{ROOT_DIR}' introuvable. Lance step0 d'abord.")
        return

    print(f"Conversion et stockage initial dans {OUT_ROOT}...")

    for chapter_name in sorted(os.listdir(ROOT_DIR)):
        chapter_path = os.path.join(ROOT_DIR, chapter_name)
        if not os.path.isdir(chapter_path): continue
        if not chapter_name.startswith("chapitre_"): continue

        chapter_number = chapter_name.replace("chapitre_", "").zfill(3)

        for file in sorted(os.listdir(chapter_path)):
            if not file.lower().endswith(".webp"): continue

            webp_path = os.path.join(chapter_path, file)
            page_number = os.path.splitext(file)[0]
            base_name = f"chapitre_{chapter_number}_{page_number}.tif"

            color_path = os.path.join(OUT_COLOR, base_name)
            gray_path = os.path.join(OUT_GRAY, base_name)

            try:
                with Image.open(webp_path) as img:
                    # Conversion LAB (Color)
                    img_lab = img.convert("LAB")
                    img_lab.save(color_path, format="TIFF", compression="tiff_deflate")

                    # Conversion L (Gray)
                    img_bw = img.convert("L")
                    img_bw.save(gray_path, format="TIFF", compression="tiff_deflate")
            except Exception as e:
                print(f"Erreur sur {webp_path} : {e}")

if __name__ == "__main__":
    main()