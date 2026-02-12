import gradio as gr
from PIL import Image
import torch, numpy as np
import torch.nn as nn
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from model import UNet
import os
import cv2
import tifffile as tiff

# ======= MODEL ========
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


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 512
OVERLAP = 0.1

def load_model():
    model = DeepColor512().to(DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

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

def reconstruct_image(image):

    image = np.array(image)

    # Conversion LAB
    img_lab_orig = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
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

    result_rgb = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
    return result_rgb

# def colorize_image(image):
#     img_resized = resize(np.array(image), (IMAGE_SIZE, IMAGE_SIZE))
#     lab_img = rgb2lab(img_resized)
#     l_channel = lab_img[:, :, 0]
#     l_tensor = torch.from_numpy((l_channel / 50.0) - 1.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

#     with torch.no_grad():
#         predicted_ab = model(l_tensor)
#     predicted_ab = predicted_ab.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 128.0

#     lab_colorized = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
#     lab_colorized[:, :, 0] = l_channel
#     lab_colorized[:, :, 1:] = predicted_ab

#     rgb_colorized = lab2rgb(lab_colorized)
#     return (rgb_colorized * 255).astype(np.uint8)

demo = gr.Interface(
    fn=reconstruct_image,
    inputs=gr.Image(type="pil", label="Upload grayscale image"),
    outputs=gr.Image(label="Colorized image"),
    title="🎨 AI One Piece Image Colorizer",
)

demo.launch(share=True)
