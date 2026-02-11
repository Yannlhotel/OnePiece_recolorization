import matplotlib.pyplot as plt
import tifffile as tiff
import cv2
import os

def show(path):
    if not os.path.exists(path):
        print(f"❌ Fichier introuvable : {path}")
        return

    # 1. Lecture
    img = tiff.imread(path)

    plt.figure(figsize=(6, 6))

    # 2. Cas Noir & Blanc (2 dimensions)
    if img.ndim == 2:
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title("Niveau de Gris (L)")

    # 3. Cas Couleur LAB (3 dimensions)
    elif img.ndim == 3:
        # Conversion obligatoire LAB -> RGB pour l'affichage écran
        # Sans ça, l'image apparaîtrait rose/saumon
        img_rgb = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        plt.imshow(img_rgb)
        plt.title("Couleur (LAB dé-codé)")

    plt.axis('off')
    plt.show()

# --- Exemple d'utilisation ---
# Remplace par le chemin de ton image
mon_image = 'ColorizationOnCluster/data/train/images/chapitre_1_page_001_chunk000.tif'
show(mon_image)