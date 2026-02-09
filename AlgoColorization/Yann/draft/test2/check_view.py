import cv2
import matplotlib.pyplot as plt

def view_lab_image_opencv(path):
    # Lecture avec OpenCV
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # Lire sans conversion automatique

    if img is None:
        print(f"Erreur : impossible de lire {path}")
        return

    img_bgr = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    cv2.imshow("Image RGB", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Exemple
view_lab_image_opencv("data/train/color/chapitre_001_page_001_chunk00.tif")
