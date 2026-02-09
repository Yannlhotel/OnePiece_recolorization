import os
import cv2  # On passe en full OpenCV
import tifffile as tiff

def main():
    # SOURCE: dossier temporaire de scraping
    ROOT_DIR = "data/temp_scraping"

    # OUTPUT
    OUT_ROOT = "data/train"
    OUT_COLOR = os.path.join(OUT_ROOT, "color")
    OUT_GRAY = os.path.join(OUT_ROOT, "gray")

    os.makedirs(OUT_COLOR, exist_ok=True)
    os.makedirs(OUT_GRAY, exist_ok=True)

    if not os.path.exists(ROOT_DIR):
        print(f"Source folder '{ROOT_DIR}' not found. Run step0 first.")
        return

    print(f"Converting using OpenCV and saving to {OUT_ROOT}...")

    for chapter_name in sorted(os.listdir(ROOT_DIR)):
        chapter_path = os.path.join(ROOT_DIR, chapter_name)
        if not os.path.isdir(chapter_path):
            continue
        
        # Extraction numéro chapitre pour le nommage
        if chapter_name.startswith("chapitre_"):
            chapter_number = chapter_name.replace("chapitre_", "").zfill(3)
        else:
            continue

        for file in sorted(os.listdir(chapter_path)):
            if not file.lower().endswith(".webp"):
                continue

            webp_path = os.path.join(chapter_path, file)
            page_number = os.path.splitext(file)[0] # ex: page_001
            base_name = f"chapitre_{chapter_number}_{page_number}.tif"

            color_path = os.path.join(OUT_COLOR, base_name)
            gray_path = os.path.join(OUT_GRAY, base_name)

            try:
                # 1. LECTURE AVEC OPENCV (Lit en BGR par défaut)
                img_bgr = cv2.imread(webp_path)
                if img_bgr is None:
                    print(f"Warning: Impossible de lire {webp_path}")
                    continue

                # 2. CONVERSION BGR -> LAB (Standard OpenCV uint8)
                # L: 0-255, A: 0-255 (128=neutre), B: 0-255 (128=neutre)
                img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2Lab)

                # 3. CONVERSION BGR -> GRAY (Pour le target N&B)
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                # 4. SAUVEGARDE PROPRE (Avec Tifffile pour la compression)
                # On spécifie 'CIELAB' pour que les logiciels sachent, 
                # même si c'est l'encodage OpenCV.
                tiff.imwrite(color_path, img_lab, compression="deflate", photometric='CIELAB')
                tiff.imwrite(gray_path, img_gray, compression="deflate")

            except Exception as e:
                print(f"Error on {webp_path}: {e}")

if __name__ == "__main__":
    main()