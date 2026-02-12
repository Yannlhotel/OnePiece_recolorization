import os
import cv2
import numpy as np
import tifffile as tiff
from PIL import Image
import config


def run():
    print("\n=== STEP 1: CONVERSION (LAB TIFF & BW) [OPENCV STANDARD] ===")

    # Vérification des dépendances
    try:
        import cv2
        import tifffile
    except ImportError as e:
        print("❌ ERREUR : Manque de librairies.")
        print(f"   Détail : {e}")
        print("   Installez-les via : pip install opencv-python tifffile")
        return

    chapters = sorted(os.listdir(config.RAW_DIR))

    for chapter_folder in chapters:
        input_dir = os.path.join(config.RAW_DIR, chapter_folder)
        if not os.path.isdir(input_dir):
            continue

        # Extraction du numéro de chapitre
        try:
            chap_num = chapter_folder.split("_")[-1]
        except:
            continue

        print(f"Converting folder: {chapter_folder}")

        files = sorted(os.listdir(input_dir))
        for file in files:
            if not file.endswith(".webp"):
                continue

            # Input: page_001.webp -> Output: chapitre_X_page_Y.tif
            page_num = os.path.splitext(file)[0].split("_")[-1]
            base_name = f"chapitre_{chap_num}_page_{page_num}.tif"

            path_src = os.path.join(input_dir, file)
            path_dst_color = os.path.join(config.PROC_IMG_DIR, base_name)
            path_dst_bw = os.path.join(config.PROC_BW_DIR, base_name)

            # On skip si déjà fait
            if os.path.exists(path_dst_color) and os.path.exists(path_dst_bw):
                continue

            try:
                # 1. Lecture robuste avec PIL (gère mieux le WebP que OpenCV parfois)
                # .convert("RGB") est crucial pour éviter les modes palette (P) ou RGBA
                img_pil = Image.open(path_src).convert("RGB")
                img_np = np.array(img_pil)

                # 2. Conversion RGB -> LAB via OpenCV
                # OpenCV en uint8 : L[0..255], a[0..255] (128=neutre), b[0..255] (128=neutre)
                # C'est le standard attendu par les réseaux de neurones
                img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)

                # 3. Sauvegarde LAB (Couleur)
                # Compression 'zlib' est efficace et standard pour le TIFF
                tiff.imwrite(path_dst_color, img_lab, compression="zlib")

                # 4. Extraction et Sauvegarde L (Noir & Blanc)
                # Le canal 0 du LAB correspond exactement à la luminance
                img_l = img_lab[:, :, 0]
                tiff.imwrite(path_dst_bw, img_l, compression="zlib")

            except Exception as e:
                print(f"  ⚠️ Error on {file}: {e}")


if __name__ == "__main__":
    run()