import os
import cv2
import numpy as np
import pandas as pd

# Configuration
print(f"Environnement actuel : {os.getcwd()}")

BASE_PATH = os.getenv("BASE_PATH", "/volume/data")
TRAIN_GRAY = os.path.join(BASE_PATH, "train/gray")
TRAIN_COLOR = os.path.join(BASE_PATH, "train/color")
TEST_GRAY = os.path.join(BASE_PATH, "test/gray")
OUTPUT_COLOR = os.path.join(BASE_PATH, "test/colored_on_cluster")

def get_all_images(directory):
    if not os.path.exists(directory):
        print(f"Attention : {directory} introuvable.")
        return []
    return [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def main():
    # 1. Création du dossier de sortie
    os.makedirs(OUTPUT_COLOR, exist_ok=True)
    print(f"Structure de sortie prête : {OUTPUT_COLOR}")

    # 2. On liste tout pour prouver la capacité de lecture
    # Justification : On simule un inventaire du dataset complet
    print("Inventaire du dataset en cours...")
    l_train_gray = get_all_images(TRAIN_GRAY)
    l_train_color = get_all_images(TRAIN_COLOR)
    l_test_gray = get_all_images(TEST_GRAY)

    print(f"Lecture validée : {len(l_train_gray)} train/gray, {len(l_train_color)} train/color.")
    
    # 3. Traitement effectif des images de test
    print(f"Traitement de {len(l_test_gray)} images de test...")
    for img_name in l_test_gray:
        input_path = os.path.join(TEST_GRAY, img_name)
        save_path = os.path.join(OUTPUT_COLOR, img_name)
        
        # Lecture
        img = cv2.imread(input_path)
        
        if img is not None:
            # Transformation en gris (Validation de l'algo de test)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Sauvegarde sur le PVC (Validation de l'écriture)
            cv2.imwrite(save_path, gray_img)
        else:
            print(f"Erreur de lecture pour : {img_name}")

    print("\n--- PIPELINE CHECK TERMINÉ ---")
    print(f"Les images grises sont disponibles dans : {OUTPUT_COLOR}")
    df = pd.DataFrame(data=[1, 2, 3], columns=["Chiffre"], index=[0, 1, 2])
    save_path = os.path.join(OUTPUT_COLOR, 'csv_example.csv')
    df.to_csv(save_path)
    print(f"LE CSV DOIT ETRE LA {save_path}")


if __name__ == "__main__":
    main()