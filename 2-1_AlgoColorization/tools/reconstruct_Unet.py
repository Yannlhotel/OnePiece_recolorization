import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import tifffile as tiff
from PIL import Image

# --- CONFIGURATION ---
INPUT_IMAGE = "results/2_Unet_MAE/img/original/chap1_page_001.webp" 
OUTPUT_IMAGE = "results/2_Unet_MAE/img/colored/chap1_page_001.webp"
MODEL_PATH = "results/2_Unet_MAE/model.pth"  # Ton nouveau fichier de poids
CHUNK_SIZE = 512
OVERLAP = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. ARCHITECTURE ONE PIECE U-NET (Strictement identique à colorize.py)
# ==========================================
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
        self.d4 = UNetBlock(512, 256, up=True) 
        self.d3 = UNetBlock(256 + 256, 128, up=True) 
        self.d2 = UNetBlock(128 + 128, 64, up=True)  
        self.d1 = UNetBlock(64 + 64, 32, up=True)    
        
        # Couche finale
        self.final = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(s1)
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        
        out = self.res(s4)
        
        out = self.d4(out)
        out = torch.cat([out, s3], dim=1) 
        
        out = self.d3(out)
        out = torch.cat([out, s2], dim=1) 
        
        out = self.d2(out)
        out = torch.cat([out, s1], dim=1) 
        
        out = self.d1(out)
        return self.final(out)

# ==========================================
# 2. LOGIQUE DE DÉCOUPAGE
# ==========================================
def get_axis_coords(total_size, chunk_size, overlap_ratio=0.1):
    if total_size < chunk_size: return [0]
    stride = int(chunk_size * (1 - overlap_ratio))
    coords = []
    current = 0
    while current + chunk_size <= total_size:
        coords.append(current)
        current += stride
    final_start = total_size - chunk_size
    if final_start not in coords: coords.append(final_start)
    return sorted(list(set(coords)))

# ==========================================
# 3. FONCTIONS UTILITAIRES
# ==========================================
def load_model(path, device):
    print(f"Loading OnePieceUNet from {path}...")
    model = OnePieceUNet().to(device)
    
    if os.path.exists(path):
        try:
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ Poids chargés avec succès.")
        except Exception as e:
            print(f"❌ Erreur de chargement (Architecture incompatible ?) : {e}")
            exit()
    else:
        print(f"⚠️ Fichier introuvable : {path}")
        exit()
    model.eval()
    return model

def process_patch(model, patch_l, device):
    # Normalisation Identique au Dataset ClusterDataset
    # L: [0..255] -> [-1..1]
    l_norm = patch_l.astype(np.float32) / 255.0 * 2 - 1
    
    input_tensor = torch.from_numpy(l_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    
    with torch.no_grad():
        output_ab = model(input_tensor)
        
    # Dénormalisation Identique au Dataset
    # AB: [-1..1] -> [0..255]
    ab_np = output_ab.squeeze(0).cpu().numpy()
    ab_np = np.transpose(ab_np, (1, 2, 0))
    ab_denorm = (ab_np * 128 + 128)
    
    return np.clip(ab_denorm, 0, 255).astype(np.uint8)

# ==========================================
# 4. MAIN
# ==========================================
def run():
    print(f"=== RECONSTRUCTION ONE PIECE ({DEVICE}) ===")
    
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ Input introuvable: {INPUT_IMAGE}")
        return

    model = load_model(MODEL_PATH, DEVICE)

    print(f"Processing: {INPUT_IMAGE}")
    try:
        img_pil = Image.open(INPUT_IMAGE).convert("RGB")
        img_np = np.array(img_pil)
    except Exception as e:
        print(f"Erreur lecture image: {e}")
        return

    # Conversion LAB pour récupérer le canal L
    img_lab_orig = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    H, W, _ = img_lab_orig.shape
    channel_l = img_lab_orig[:, :, 0]
    
    # Canvas vide pour AB
    full_ab = np.full((H, W, 2), 128, dtype=np.uint8)

    # Découpage
    y_coords = get_axis_coords(H, CHUNK_SIZE, OVERLAP)
    x_coords = get_axis_coords(W, CHUNK_SIZE, OVERLAP)
    print(f"Grid: {len(y_coords)}x{len(x_coords)} chunks")

    for y in y_coords:
        for x in x_coords:
            h_end = min(y + CHUNK_SIZE, H)
            w_end = min(x + CHUNK_SIZE, W)
            
            patch_l = channel_l[y:h_end, x:w_end]
            
            # Padding
            pad_h = CHUNK_SIZE - patch_l.shape[0]
            pad_w = CHUNK_SIZE - patch_l.shape[1]
            if pad_h > 0 or pad_w > 0:
                patch_l = np.pad(patch_l, ((0, pad_h), (0, pad_w)), mode='reflect')

            # Prédiction
            patch_ab = process_patch(model, patch_l, DEVICE)
            
            # Retrait Padding
            if pad_h > 0 or pad_w > 0:
                patch_ab = patch_ab[:CHUNK_SIZE-pad_h, :CHUNK_SIZE-pad_w]

            full_ab[y:h_end, x:w_end] = patch_ab

    # Fusion
    print("Fusion des canaux...")
    result_lab = np.zeros((H, W, 3), dtype=np.uint8)
    result_lab[:, :, 0] = channel_l
    result_lab[:, :, 1:] = full_ab

    # Sauvegarde JPG (LAB -> RGB -> BGR)
    result_bgr = cv2.cvtColor(cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB), cv2.COLOR_RGB2BGR)
    
    # Gestion nom de fichier
    final_name = OUTPUT_IMAGE if OUTPUT_IMAGE.endswith(".jpg") else OUTPUT_IMAGE.split(".")[0] + ".jpg"
    
    cv2.imwrite(final_name, result_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(f"✅ Image sauvée : {final_name}")

if __name__ == "__main__":
    run()