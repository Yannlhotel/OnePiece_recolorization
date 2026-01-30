import os
from PIL import Image

# Dossier contenant tes .webp
input_dir = "one_piece_chap3"

# Dossier de sortie des .jpg
output_dir = "one_piece_chap3_jpg"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if not filename.lower().endswith(".webp"):
        continue

    webp_path = os.path.join(input_dir, filename)
    jpg_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.jpg")

    with Image.open(webp_path) as img:
        rgb = img.convert("RGB")  
        rgb.save(jpg_path, "JPEG", quality=95)

    print(f"Converti {filename} → {os.path.basename(jpg_path)}")