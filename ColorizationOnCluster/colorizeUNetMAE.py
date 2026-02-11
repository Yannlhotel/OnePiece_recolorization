import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import tifffile as tiff
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# --- 1. CONFIGURATION CLUSTER ---
BASE_PATH = os.getenv("BASE_PATH", "/volume/data")
TRAIN_BW = os.path.join(BASE_PATH, "train/bw")
TRAIN_COLOR = os.path.join(BASE_PATH, "train/images")
VAL_COLOR = os.path.join(BASE_PATH, "val/images")
OUTPUT_COLOR = os.path.join(BASE_PATH, "val/colored_on_cluster") # Dossier spécifique

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 2e-4
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOSS_TYPE = "L1Loss"

print(f"--- CONFIGURATION ({LOSS_TYPE}) ---")
print(f"Device: {DEVICE}")
print(f"Output: {OUTPUT_COLOR}")
os.makedirs(OUTPUT_COLOR, exist_ok=True)

# --- 2. ARCHITECTURE CNN ---
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
    def forward(self, x): return x + self.conv(x)

class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up=False):
        super().__init__()
        if up:
            self.conv = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True)
            )

    def forward(self, x):
        return self.conv(x)

class OnePieceUNet(nn.Module):
    def __init__(self):
        super(OnePieceUNet, self).__init__()
        
        # Encodeur (Descente)
        self.e1 = nn.Conv2d(1, 64, 4, 2, 1) # 512 -> 256
        self.e2 = UNetBlock(64, 128)        # 256 -> 128
        self.e3 = UNetBlock(128, 256)       # 128 -> 64
        self.e4 = UNetBlock(256, 512)       # 64 -> 32
        
        # Bottleneck (Le bas du U)
        self.res = nn.Sequential(ResidualBlock(512), ResidualBlock(512))
        
        # Décodeur (Remontée avec Skip Connections)
        # On multiplie les canaux par 2 à l'entrée car on concatène avec l'encodeur
        self.d4 = UNetBlock(512, 256, up=True) 
        self.d3 = UNetBlock(256 + 256, 128, up=True) # +256 venant de e3
        self.d2 = UNetBlock(128 + 128, 64, up=True)  # +128 venant de e2
        self.d1 = UNetBlock(64 + 64, 32, up=True)    # +64 venant de e1
        
        # Couche finale pour retrouver la taille originale et les 2 canaux (a, b)
        self.final = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # Forward Encodeur
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        
        # Bottleneck
        out = self.res(s4)
        
        # Forward Décodeur avec Concaténation (skip connections)
        out = self.d4(out)
        out = torch.cat([out, s3], dim=1) # On recolle les détails de s3
        
        out = self.d3(out)
        out = torch.cat([out, s2], dim=1) # On recolle les détails de s2
        
        out = self.d2(out)
        out = torch.cat([out, s1], dim=1) # On recolle les détails de s1
        
        out = self.d1(out)
        return self.final(out)
    
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
        l_tensor = torch.zeros((1, IMG_SIZE, IMG_SIZE)).float()
        ab_tensor = torch.zeros((2, IMG_SIZE, IMG_SIZE)).float()

        try:
            if self.mode == 'val':
                path = os.path.join(self.color_dir, fname)
                img_lab = tiff.imread(path)
                if img_lab is not None:
                    if img_lab.shape[0] != IMG_SIZE: img_lab = cv2.resize(img_lab, (IMG_SIZE, IMG_SIZE))
                    l_channel = img_lab[:, :, 0].astype(np.float32) / 255.0 * 2 - 1
                    l_tensor = torch.from_numpy(l_channel).unsqueeze(0).float()
                    ab_channel = (img_lab[:, :, 1:].astype(np.float32) - 128) / 128.0
                    ab_tensor = torch.from_numpy(ab_channel.transpose(2, 0, 1)).float()
            else:
                bw_path = os.path.join(self.bw_dir, fname)
                img_bw = tiff.imread(bw_path)
                if img_bw is not None:
                    if img_bw.shape[0] != IMG_SIZE: img_bw = cv2.resize(img_bw, (IMG_SIZE, IMG_SIZE))
                    l_norm = img_bw.astype(np.float32) / 255.0 * 2 - 1
                    l_tensor = torch.from_numpy(l_norm).unsqueeze(0).float()
                
                color_path = os.path.join(self.color_dir, fname)
                if os.path.exists(color_path):
                    img_color = tiff.imread(color_path)
                    if img_color is not None:
                        if img_color.shape[0] != IMG_SIZE: img_color = cv2.resize(img_color, (IMG_SIZE, IMG_SIZE))
                        ab_channel = (img_color[:, :, 1:].astype(np.float32) - 128) / 128.0
                        ab_tensor = torch.from_numpy(ab_channel.transpose(2, 0, 1)).float()
        except Exception as e: print(f"Error {fname}: {e}")
        return l_tensor, ab_tensor, fname

# --- 4. UTILITAIRES ---
def lab_to_bgr(l_t, ab_t):
    l_np = (l_t.cpu().detach().numpy().squeeze() + 1) / 2 * 255
    ab_np = ab_t.cpu().detach().numpy().squeeze() * 128 + 128
    img_lab = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    img_lab[:, :, 0] = np.clip(l_np, 0, 255)
    img_lab[:, :, 1:] = np.clip(ab_np.transpose(1, 2, 0), 0, 255)
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

# --- 5. MAIN ---
def main():
    model = OnePieceUNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- CONFIGURATION LOSS ICI ---
    criterion = nn.L1Loss() 
    # ------------------------------

    train_ds = ClusterDataset(TRAIN_BW, TRAIN_COLOR, mode='train')
    val_ds = ClusterDataset(None, VAL_COLOR, mode='val')
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    print(f"\n--- Training with {LOSS_TYPE} ---")
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Ep {epoch+1}/{EPOCHS}")
        
        for i, (l_in, ab_target, _) in loop:
            l_in, ab_target = l_in.to(DEVICE), ab_target.to(DEVICE)
            optimizer.zero_grad()
            ab_pred = model(l_in)
            loss = criterion(ab_pred, ab_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    # --- VALIDATION & MÉTRIQUES ---
    print("\n--- Calcul des Métriques Finales ---")
    model.eval()
    
    # Accumulateurs
    total_mse = 0.0
    total_mae = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    count = 0
    
    # Critères pour l'évaluation (on calcule les deux peu importe la loss d'entraînement)
    eval_mse = nn.MSELoss()
    eval_mae = nn.L1Loss()

    with torch.no_grad():
        for l_in, ab_target, fnames in tqdm(val_loader):
            l_in, ab_target = l_in.to(DEVICE), ab_target.to(DEVICE)
            ab_pred = model(l_in)

            # 1. Métriques Mathématiques (Tensors)
            total_mse += eval_mse(ab_pred, ab_target).item()
            total_mae += eval_mae(ab_pred, ab_target).item()

            # 2. Métriques Visuelles (Images RGB)
            # On reconstruit l'image prédite ET l'image cible (Ground Truth)
            img_pred_bgr = lab_to_bgr(l_in[0], ab_pred[0])
            img_target_bgr = lab_to_bgr(l_in[0], ab_target[0]) # Pour comparer avec la vraie vérité terrain

            # PSNR
            total_psnr += psnr(img_target_bgr, img_pred_bgr, data_range=255)
            # SSIM (multichannel=True ou channel_axis selon version, ici on gère le standard)
            try:
                # Scikit-image > 0.19 uses channel_axis
                total_ssim += ssim(img_target_bgr, img_pred_bgr, channel_axis=2, data_range=255)
            except TypeError:
                # Older versions
                total_ssim += ssim(img_target_bgr, img_pred_bgr, multichannel=True, data_range=255)

            count += 1
            
            # Sauvegarde Image
            fname_jpg = os.path.splitext(fnames[0])[0] + ".jpg"
            cv2.imwrite(os.path.join(OUTPUT_COLOR, fname_jpg), img_pred_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    # Moyennes
    avg_mse = total_mse / count
    avg_mae = total_mae / count
    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    print(f"\n=== RÉSULTATS FINAUX ({LOSS_TYPE}) ===")
    print(f"MSE : {avg_mse:.6f}")
    print(f"MAE : {avg_mae:.6f}")
    print(f"PSNR: {avg_psnr:.2f} dB")
    print(f"SSIM: {avg_ssim:.4f}")

    # Sauvegarde Metrics
    with open(os.path.join(OUTPUT_COLOR, "metrics.txt"), "w") as f:
        f.write(f"Model: DeepColor512\n")
        f.write(f"Loss Function Used: {LOSS_TYPE}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n")
        f.write("-" * 20 + "\n")
        f.write(f"Final MSE (Tensor): {avg_mse:.6f}\n")
        f.write(f"Final MAE (Tensor): {avg_mae:.6f}\n")
        f.write(f"Final PSNR (RGB):   {avg_psnr:.4f} dB\n")
        f.write(f"Final SSIM (RGB):   {avg_ssim:.4f}\n")
        torch.save(model.state_dict(), os.path.join(OUTPUT_COLOR, "onepiece_unet_final.pth"))

if __name__ == "__main__":
    main()