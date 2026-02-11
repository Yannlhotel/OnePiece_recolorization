import cv2
import numpy as np
import torch
from model import UNet

# 1. Charger le modèle et les poids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=1, out_channels=2).to(device)
model.load_state_dict(torch.load("./final_colorization_model.pth", map_location=device))
model.eval()

# 2. Préparer l'image N&B (Canal L)
path_image = "./data/train/bw/chapitre_1_page_001_chunk001.tif"
img_bgr = cv2.imread(path_image)
h, w = img_bgr.shape[:2] # On garde la taille d'origine pour la fin

# Conversion en Gris puis resize pour le modèle
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_resized = cv2.resize(img_gray, (256, 256))

# Normalisation [-1, 1]
l_input = (img_resized.astype(np.float32) / 127.5) - 1.0
l_tensor = torch.from_numpy(l_input).unsqueeze(0).unsqueeze(0).to(device) # Shape (1, 1, 256, 256)

# 3. Inférence (Prédire ab)
with torch.no_grad():
    prediction_ab = model(l_tensor) # Sortie entre -1 et 1

# 4. Reconstruction
# On repasse en [0, 255] pour OpenCV
pred_ab = ((prediction_ab.squeeze().permute(1, 2, 0).cpu().numpy() + 1) * 127.5).astype(np.uint8)

# On crée une image LAB vide
res_lab = np.zeros((256, 256, 3), dtype=np.uint8)
res_lab[:, :, 0] = img_resized # Le canal L d'origine
res_lab[:, :, 1:3] = pred_ab   # Les canaux ab prédits

# Conversion LAB -> BGR pour affichage
res_bgr = cv2.cvtColor(res_lab, cv2.COLOR_Lab2BGR)
res_final = cv2.resize(res_bgr, (w, h)) # Remise à la taille d'origine

cv2.imshow("Original N&B", img_gray)
cv2.imshow("Colorisation IA", res_final)
cv2.waitKey(0)