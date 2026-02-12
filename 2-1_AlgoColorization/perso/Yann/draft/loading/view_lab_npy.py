import cv2
import numpy as np
import os

def show_lab_npy(npy_path):
    """
    Charge un fichier .npy (LAB float32), le convertit en BGR 
    et l'affiche via OpenCV.
    """
    if not os.path.exists(npy_path):
        print(f"Erreur : Le fichier {npy_path} n'existe pas.")
        return

    # 1. Chargement du fichier numpy
    # Les valeurs sont L[0-100], a[-127, 127], b[-127, 127]
    img_lab = np.load(npy_path)

    # 2. Conversion LAB (float32) -> BGR (float32)
    # OpenCV s'attend à ce que le float32 soit dans des plages spécifiques 
    # pour la conversion, ou bien on repasse en BGR directement.
    img_bgr_float = cv2.cvtColor(img_lab, cv2.COLOR_Lab2BGR)

    # 3. Conversion en uint8 [0-255] pour l'affichage
    # On multiplie par 255 et on clip pour éviter les débordements
    img_bgr = np.clip(img_bgr_float * 255.0, 0, 255).astype(np.uint8)

    # 4. Affichage
    cv2.imshow("Visualisation LAB -> BGR", img_bgr)
    print(f"Affichage de : {os.path.basename(npy_path)}")
    print("Appuyez sur une touche pour fermer la fenêtre...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Exemple d'utilisation (ajuste le chemin selon tes fichiers générés)
    sample_path = "data/train/color_lab/chapitre_1_page_003_chunk0.npy"
    show_lab_npy(sample_path)