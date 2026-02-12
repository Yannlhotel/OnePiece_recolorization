import gradio as gr
from PIL import Image
import torch, numpy as np
import torch.nn as nn
from skimage.color import rgb2lab, lab2rgb
from skimage.transform import resize
from model import UNet

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
IMAGE_SIZE = 512

def load_model():
    model = DeepColor512().to(DEVICE)
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.eval()
    return model

model = load_model()

def colorize_image(image):
    img_resized = resize(np.array(image), (IMAGE_SIZE, IMAGE_SIZE))
    lab_img = rgb2lab(img_resized)
    l_channel = lab_img[:, :, 0]
    l_tensor = torch.from_numpy((l_channel / 50.0) - 1.0).unsqueeze(0).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        predicted_ab = model(l_tensor)
    predicted_ab = predicted_ab.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 128.0

    lab_colorized = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3))
    lab_colorized[:, :, 0] = l_channel
    lab_colorized[:, :, 1:] = predicted_ab

    rgb_colorized = lab2rgb(lab_colorized)
    return (rgb_colorized * 255).astype(np.uint8)

demo = gr.Interface(
    fn=colorize_image,
    inputs=gr.Image(type="pil", label="Upload grayscale image"),
    outputs=gr.Image(label="Colorized image"),
    title="🎨 AI Image Colorizer",
)

demo.launch(share=True)
