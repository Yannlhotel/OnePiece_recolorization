import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import tifffile as tiff  # INDISPENSABLE pour lire les données LAB brutes

# --- 1. CONFIGURATION CLUSTER ---
BASE_PATH = os.getenv("BASE_PATH", "/volume/data")

# Chemins (correspondant à votre pipeline step2/step3)
TRAIN_BW = os.path.join(BASE_PATH, "train/bw")
TRAIN_COLOR = os.path.join(BASE_PATH, "train/images")
VAL_COLOR = os.path.join(BASE_PATH, "val/images")   
OUTPUT_COLOR = os.path.join(BASE_PATH, "val/colored_on_cluster")

# Hyperparamètres
BATCH_SIZE = 8       # Réduire à 4 si erreur OOM
EPOCHS = 10          # Suffisant pour un bon résultat initial
LEARNING_RATE = 2e-4
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"--- CONFIGURATION ---")
print(f"Device: {DEVICE}")
print(f"Train BW: {TRAIN_BW}")
print(f"Train Color: {TRAIN_COLOR}")
print(f"Output: {OUTPUT_COLOR}")

os.makedirs(OUTPUT_COLOR, exist_ok=True)

# --- 2. ARCHITECTURE CNN (DeepColor512) ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.conv(x)

class DeepColor512(nn.Module):
    def __init__(self):
        super(DeepColor512, self).__init__()
        # Encodeur
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), 
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Bottleneck
        self.res_blocks = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256)
        )
        # Décodeur
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1), 
            nn.Tanh() # Sortie normalisée entre -1 et 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

# --- 3. DATASET MANAGER (CORRIGÉ TIFF) ---
class ClusterDataset(Dataset):
    def __init__(self, bw_dir, color_dir, mode='train'):
        self.bw_dir = bw_dir
        self.color_dir = color_dir
        self.mode = mode
        
        # Choix du dossier racine pour lister les fichiers
        if self.mode == 'val':
            self.root_dir = color_dir
        else:
            self.root_dir = bw_dir

        if os.path.exists(self.root_dir):
            self.file_names = [f for f in os.listdir(self.root_dir) if f.lower().endswith(('.tif', '.tiff'))]
        else:
            print(f"ATTENTION: Dossier introuvable -> {self.root_dir}")
            self.file_names = []
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]
        
        # Initialisation par défaut (sécurité)
        l_tensor = torch.zeros((1, IMG_SIZE, IMG_SIZE)).float()
        ab_tensor = torch.zeros((2, IMG_SIZE, IMG_SIZE)).float()

        try:
            # --- CAS VALIDATION (Source = Image Couleur TIFF LAB) ---
            if self.mode == 'val':
                path = os.path.join(self.color_dir, fname)
                img_lab = tiff.imread(path) # Lecture BRUTE (H, W, 3)
                
                if img_lab is not None:
                    # Resize de sécurité
                    if img_lab.shape[0] != IMG_SIZE:
                         img_lab = cv2.resize(img_lab, (IMG_SIZE, IMG_SIZE))

                    # Normalisation L (0..255) -> Tensor (-1..1)
                    l_channel = img_lab[:, :, 0].astype(np.float32) / 255.0 * 2 - 1
                    l_tensor = torch.from_numpy(l_channel).unsqueeze(0).float()
                    
                    # Normalisation AB (0..255) -> Tensor (-1..1)
                    # Note : Pillow sauve AB avec offset +128
                    ab_channel = (img_lab[:, :, 1:].astype(np.float32) - 128) / 128.0
                    ab_tensor = torch.from_numpy(ab_channel.transpose(2, 0, 1)).float()

            # --- CAS TRAIN (Source = Fichier BW + Fichier Color) ---
            else:
                # 1. Input BW
                bw_path = os.path.join(self.bw_dir, fname)
                img_bw = tiff.imread(bw_path) # Lecture brute (H, W)
                
                if img_bw is not None:
                    if img_bw.shape[0] != IMG_SIZE:
                        img_bw = cv2.resize(img_bw, (IMG_SIZE, IMG_SIZE))
                        
                    l_norm = img_bw.astype(np.float32) / 255.0 * 2 - 1
                    l_tensor = torch.from_numpy(l_norm).unsqueeze(0).float()

                # 2. Target Color
                color_path = os.path.join(self.color_dir, fname)
                if os.path.exists(color_path):
                    img_color = tiff.imread(color_path) # Lecture brute (H, W, 3)
                    
                    if img_color is not None:
                        if img_color.shape[0] != IMG_SIZE:
                            img_color = cv2.resize(img_color, (IMG_SIZE, IMG_SIZE))
                            
                        # Extraction AB seulement
                        ab_channel = (img_color[:, :, 1:].astype(np.float32) - 128) / 128.0
                        ab_tensor = torch.from_numpy(ab_channel.transpose(2, 0, 1)).float()

        except Exception as e:
            print(f"Erreur lecture {fname}: {e}")

        return l_tensor, ab_tensor, fname

# --- 4. UTILITAIRES ---
def lab_to_bgr(l_t, ab_t):
    """
    Recombine L (input) et AB (pred) et convertit en BGR.
    Cette fonction prépare l'image pour être sauvegardée correctement en JPG/PNG via OpenCV.
    """
    # Dé-normalisation
    l_np = (l_t.cpu().detach().numpy().squeeze() + 1) / 2 * 255
    ab_np = ab_t.cpu().detach().numpy().squeeze() * 128 + 128
    
    # Création du conteneur LAB
    img_lab = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img_lab[:, :, 0] = np.clip(l_np, 0, 255)
    img_lab[:, :, 1:] = np.clip(ab_np.transpose(1, 2, 0), 0, 255)
    
    # Conversion finale pour affichage standard
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

# --- 5. MAIN ---
def main():
    # A. Initialisation
    model = DeepColor512().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # B. Chargement des données
    print("Chargement des datasets...")
    # Train : lit BW dans train/bw et Couleur dans train/images
    train_ds = ClusterDataset(TRAIN_BW, TRAIN_COLOR, mode='train')
    # Val : lit tout dans val/images (génère le BW à la volée)
    val_ds = ClusterDataset(None, VAL_COLOR, mode='val') 
    
    # IMPORTANT: num_workers=0 pour éviter le crash mémoire partagée sur le cluster
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    print(f"Images Train : {len(train_ds)}")
    print(f"Images Val : {len(val_ds)}")
    
    if len(train_ds) == 0:
        print("ERREUR FATALE: Pas d'images d'entraînement trouvées ! Vérifiez les dossiers.")
        return

    # C. Boucle d'Entraînement
    print(f"\n--- Démarrage de l'entraînement ({EPOCHS} époques) ---")
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # mininterval=5.0 évite de saturer les logs Kubernetes
        loop = tqdm(enumerate(train_loader), total=len(train_loader), 
                   desc=f"Epoch {epoch+1}/{EPOCHS}", mininterval=5.0)
        
        for i, (l_in, ab_target, _) in loop:
            l_in, ab_target = l_in.to(DEVICE), ab_target.to(DEVICE)
            
            optimizer.zero_grad()
            ab_pred = model(l_in)
            
            loss = criterion(ab_pred, ab_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            # Log texte explicite de sécurité pour le cluster (tous les 200 batchs)
            if i % 200 == 0:
                print(f" [Ep {epoch+1}] Batch {i}/{len(train_loader)} - Loss: {loss.item():.4f}")
        
        print(f"==> Fin Epoch {epoch+1} | Loss Moyenne: {train_loss / len(train_loader):.5f}")

    # D. Sauvegarde du Modèle
    model_path = os.path.join(OUTPUT_COLOR, "model_final.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Modèle sauvegardé : {model_path}")

    # E. Génération Finale et Calcul MSE
    print("\n--- Génération des images finales (Validation) ---")
    model.eval()
    mse_total = 0.0
    count = 0
    
    with torch.no_grad():
        for l_in, ab_target, fnames in tqdm(val_loader, mininterval=2.0):
            l_in = l_in.to(DEVICE)
            ab_target = ab_target.to(DEVICE)
            
            # Prédiction
            ab_pred = model(l_in)
            
            # Calcul erreur
            loss = criterion(ab_pred, ab_target)
            mse_total += loss.item()
            count += 1
            
            # Recombinaison en image visible (BGR)
            img_bgr = lab_to_bgr(l_in[0], ab_pred[0])
            
            # Sauvegarde en JPG pour corriger les couleurs (le TIFF/LAB est mal lu par les visionneuses)
            fname = fnames[0]
            fname_jpg = os.path.splitext(fname)[0] + ".jpg"
            save_path = os.path.join(OUTPUT_COLOR, fname_jpg)
            
            cv2.imwrite(save_path, img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    final_mse = mse_total / count if count > 0 else 0
    print(f"\n" + "="*30)
    print(f"RAPPORT FINAL CLUSTER")
    print(f"MSE Moyen (Validation) : {final_mse:.6f}")
    print(f"Images générées dans : {OUTPUT_COLOR}")
    print(f"="*30)
    
    # Petit fichier de metrics
    with open(os.path.join(OUTPUT_COLOR, "metrics.txt"), "w") as f:
        f.write(f"Model: DeepColor512\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Final MSE: {final_mse:.6f}\n")

if __name__ == "__main__":
    main()