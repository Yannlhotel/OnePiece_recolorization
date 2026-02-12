import os
import cv2
import numpy as np

def convert_to_lab(src_dir, dest_color, dest_gray):
    os.makedirs(dest_color, exist_ok=True)
    os.makedirs(dest_gray, exist_ok=True)
    
    for f in os.listdir(src_dir):
        img_bgr = cv2.imread(os.path.join(src_dir, f))
        
        # 1. Conversion LAB en float32 pour éviter les arrondis uint8
        # L: [0, 100], a: [-127, 127], b: [-127, 127]
        img_lab = cv2.cvtColor(img_bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
        
        # 2. Séparation N&B (Luminance seule)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Sauvegarde en .npy pour préserver les floats sans perte de fichier image
        base_name = f.replace(".png", "")
        np.save(os.path.join(dest_color, f"{base_name}.npy"), img_lab)
        cv2.imwrite(os.path.join(dest_gray, f"{base_name}.png"), img_gray)

def main():
    convert_to_lab("data/train_rgb", "data/train/color_lab", "data/train/gray")
    convert_to_lab("data/test_rgb", "data/test/color_lab", "data/test/gray")
    print("Conversion LAB float32 terminée.")

if __name__ == "__main__":
    main()