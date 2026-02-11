import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from model import UNet
import tifffile as tiff

# --- CONFIGURATION ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
NUM_EPOCHS = 20
IMAGE_SIZE = 256
IMG_SIZE = 512
LOAD_MODEL_PATH = "final_colorization_model.pth"
SAVE_MODEL_PATH = "finetuned_colorization.pth"

# --- DATASET ---
import cv2
import numpy as np
from torch.utils.data import Dataset

# --- 3. DATASET MANAGER ---
class ClusterDataset(Dataset):
    def __init__(self, bw_dir, color_dir, mode='train'):
        self.bw_dir, self.color_dir, self.mode = bw_dir, color_dir, mode
        self.root_dir = color_dir if mode == 'val' else bw_dir
        if os.path.exists(self.root_dir):
            self.file_names = [f for f in os.listdir(self.root_dir) if f.lower().endswith(('.tif', '.tiff'))]
        else:
            self.file_names = []

    def __len__(self): return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        # Initialisation sécurisée
        l_tensor = torch.zeros((1, IMG_SIZE, IMG_SIZE)).float()
        ab_tensor = torch.zeros((2, IMG_SIZE, IMG_SIZE)).float()

        try:
            path = os.path.join(self.bw_dir if self.mode != 'val' else self.color_dir, fname)
            img = tiff.imread(path).astype(np.float32)

            # Si c'est du TIFF 16-bit, on ramène en 8-bit (0-255)
            if img.max() > 255:
                img = (img / 65535.0) * 255.0

            if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

            # Normalisation L (Gris)
            # On s'assure d'être dans [0, 255] avant le calcul
            img_clipped = np.clip(img, 0, 255)
            
            if self.mode == 'val' or (img.ndim == 3 and img.shape[2] >= 3):
                # On traite une image couleur
                l_norm = img_clipped[:, :, 0] / 127.5 - 1.0
                ab_norm = (img_clipped[:, :, 1:3] - 128.0) / 128.0
                l_tensor = torch.from_numpy(l_norm).unsqueeze(0)
                ab_tensor = torch.from_numpy(ab_norm.transpose(2, 0, 1))
            else:
                # On traite une image N&B
                l_norm = img_clipped / 127.5 - 1.0
                l_tensor = torch.from_numpy(l_norm).unsqueeze(0)
                
                # Charger les couleurs si on n'est pas en val
                color_path = os.path.join(self.color_dir, fname)
                if os.path.exists(color_path):
                    c_img = tiff.imread(color_path).astype(np.float32)
                    if c_img.max() > 255: c_img = (c_img / 65535.0) * 255.0
                    c_img = cv2.resize(c_img, (IMG_SIZE, IMG_SIZE))
                    ab_norm = (np.clip(c_img[:, :, 1:3], 0, 255) - 128.0) / 128.0
                    ab_tensor = torch.from_numpy(ab_norm.transpose(2, 0, 1))

        except Exception as e:
            print(f"Error {fname}: {e}")
            
        return l_tensor.float(), ab_tensor.float() # On retire fname ici


# --- INITIALISATION ---
model = UNet(in_channels=1, out_channels=2).to(DEVICE)

# Chargement sécurisé
try:
    state_dict = torch.load(LOAD_MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print("Poids chargés avec succès.")
except Exception as e:
    print(f"Erreur chargement : {e}")

# STRATÉGIE DE FINETUNING : Geler l'Encoder (Downsampling)
# On ne réentraîne que le décodeur pour stabiliser la loss
for name, param in model.named_parameters():
    if "down" in name:
        param.requires_grad = False

# Dataset & Loader
dataset = ClusterDataset(bw_dir='./data/train/bw', color_dir='./data/train/images')
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Optimiseur Adam avec epsilon plus élevé pour éviter les divisions par zéro
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                       lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-4)
criterion = nn.L1Loss()

# --- BOUCLE D'ENTRAÎNEMENT ---
model.train()
print("Début du finetuning (Encoder gelé)...")

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    for i, (l_input, ab_target) in enumerate(train_loader):
        l_input, ab_target = l_input.to(DEVICE), ab_target.to(DEVICE)

        # Forward
        prediction = model(l_input)
        loss = criterion(prediction, ab_target)

        if torch.isnan(loss):
            print(f"Loss NaN au batch {i}, on passe.")
            continue

        # Backward avec Gradient Clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1) # Clipping très serré
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)
    print(f"Époque [{epoch+1}/{NUM_EPOCHS}] - Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), SAVE_MODEL_PATH)