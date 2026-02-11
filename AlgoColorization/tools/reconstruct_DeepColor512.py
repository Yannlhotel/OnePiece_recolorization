import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import tifffile as tiff
from PIL import Image
from tqdm import tqdm  # Barre de progression (pip install tqdm)

# --- CONFIGURATION ---
INPUT_DIR = "AlgoColorization/Quentin/model_test"              # Dossier contenant tes images sources
OUTPUT_DIR = "AlgoColorization/Quentin/model_test_output"   # Dossier où seront sauvées les images colorisées
MODEL_PATH = "AlgoColorization/Quentin/other_colorization_model.pth"
CHUNK_SIZE = 512
OVERLAP = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. ARCHITECTURE DU MODÈLE (DeepColor512)
# ==========================================
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

class DeepColor512(nn.Module):
    def __init__(self):
        super(DeepColor512, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
        )
        self.res_blocks = nn.Sequential(ResidualBlock(256), ResidualBlock(256), ResidualBlock(256))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(32, 2, kernel_size=3, padding=1), nn.Tanh()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.res_blocks(x)
        x = self.decoder(x)
        return x

# ==========================================
# 2. FONCTIONS UTILITAIRES
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

def load_model(path, device):
    print(f"Loading model from {path}...")
    model = DeepColor512().to(device)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"⚠️ ERROR: Model file not found at {path}!")
        exit()
    model.eval()
    return model

def process_patch(model, patch_l, device):
    l_norm = patch_l.astype(np.float32) / 255.0 * 2 - 1
    input_tensor = torch.from_numpy(l_norm).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output_ab = model(input_tensor)
    ab_np = output_ab.squeeze(0).cpu().numpy()
    ab_np = np.transpose(ab_np, (1, 2, 0))
    ab_denorm = (ab_np * 128 + 128)
    return np.clip(ab_denorm, 0, 255).astype(np.uint8)

# ==========================================
# 3. FONCTION DE TRAITEMENT D'UNE IMAGE
# ==========================================
def reconstruct_image(model, img_path, output_path):
    try:
        # Lecture
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)
    except Exception as e:
        print(f"❌ Erreur lecture {os.path.basename(img_path)}: {e}")
        return

    # Conversion LAB
    img_lab_orig = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    H, W, _ = img_lab_orig.shape
    channel_l = img_lab_orig[:, :, 0]
    full_ab = np.full((H, W, 2), 128, dtype=np.uint8)

    # Découpage
    y_coords = get_axis_coords(H, CHUNK_SIZE, OVERLAP)
    x_coords = get_axis_coords(W, CHUNK_SIZE, OVERLAP)

    # Boucle sur les chunks
    for y in y_coords:
        for x in x_coords:
            h_end = min(y + CHUNK_SIZE, H)
            w_end = min(x + CHUNK_SIZE, W)
            
            patch_l = channel_l[y:h_end, x:w_end]
            pad_h = CHUNK_SIZE - patch_l.shape[0]
            pad_w = CHUNK_SIZE - patch_l.shape[1]
            
            if pad_h > 0 or pad_w > 0:
                patch_l = np.pad(patch_l, ((0, pad_h), (0, pad_w)), mode='reflect')

            patch_ab = process_patch(model, patch_l, DEVICE)
            
            if pad_h > 0 or pad_w > 0:
                patch_ab = patch_ab[:CHUNK_SIZE-pad_h, :CHUNK_SIZE-pad_w]

            full_ab[y:h_end, x:w_end] = patch_ab

    # Recombinaison
    result_lab = np.zeros((H, W, 3), dtype=np.uint8)
    result_lab[:, :, 0] = channel_l
    result_lab[:, :, 1:] = full_ab

    # Sauvegarde JPG
    result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

# ==========================================
# 4. MAIN LOOP
# ==========================================
def run():
    print(f"=== BATCH RECONSTRUCTION ({DEVICE}) ===")
    
    # Création du dossier de sortie
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Chargement Modèle
    model = load_model(MODEL_PATH, DEVICE)

    # Liste des images
    valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff")
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_exts)]
    
    if not files:
        print(f"⚠️ Aucun fichier image trouvé dans {INPUT_DIR}")
        return

    print(f"Found {len(files)} images. Starting processing...")

    # Boucle avec barre de progression
    for fname in tqdm(files):
        input_path = os.path.join(INPUT_DIR, fname)
        
        # Nom de sortie : cat.jpg -> cat_reco.jpg
        name_no_ext = os.path.splitext(fname)[0]
        output_name = f"{name_no_ext}_reco.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        reconstruct_image(model, input_path, output_path)

    print(f"\n✅ Terminé ! Résultats dans : {OUTPUT_DIR}")

if __name__ == "__main__":
    run()