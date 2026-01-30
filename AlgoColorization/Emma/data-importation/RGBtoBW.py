import os
from PIL import Image

INPUT_DIR = "one_piece_jpg"
OUTPUT_DIR = "one_piece_jpg_bw"

os.makedirs(OUTPUT_DIR, exist_ok=True)

for filename in sorted(os.listdir(INPUT_DIR)):
    if not filename.lower().endswith(".jpg"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)

    name, ext = os.path.splitext(filename)
    output_filename = f"{name}_bw{ext}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        with Image.open(input_path) as img:
            gray = img.convert("L")  # conversion NB
            gray.save(output_path, "JPEG", quality=95)

        print(f"NB : {output_filename}")

    except Exception as e:
        print(f"Erreur sur {filename} : {e}")