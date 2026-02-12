import os
import cv2
import shutil

def main():
    RAW_DIR = "data/temp_scraping"
    OUT_DIR = "data/all_rgb"
    os.makedirs(OUT_DIR, exist_ok=True)

    for root, _, files in os.walk(RAW_DIR):
        for f in files:
            if f.lower().endswith(".webp"):
                # On utilise OpenCV pour garantir un encodage propre au départ
                img = cv2.imread(os.path.join(root, f))
                if img is not None:
                    # Nom unique : chapitre_page.png (PNG est sans perte)
                    chap_name = os.path.basename(root)
                    out_name = f"{chap_name}_{f.replace('.webp', '.png')}"
                    cv2.imwrite(os.path.join(OUT_DIR, out_name), img)
    print("Chargement terminé : images converties en PNG sans perte.")

if __name__ == "__main__":
    main()