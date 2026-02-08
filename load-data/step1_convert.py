import os
from PIL import Image
import config


def run():
    print("\n=== STEP 1: CONVERSION (LAB TIFF & BW) ===")

    chapters = sorted(os.listdir(config.RAW_DIR))

    for chapter_folder in chapters:
        input_dir = os.path.join(config.RAW_DIR, chapter_folder)
        if not os.path.isdir(input_dir):
            continue

        # Extract chapter number (e.g., "chapitre_1000" -> "1000")
        try:
            chap_num = chapter_folder.split("_")[-1]
        except:
            continue

        print(f"Converting folder: {chapter_folder}")

        files = sorted(os.listdir(input_dir))
        for file in files:
            if not file.endswith(".webp"):
                continue

            # Input: page_001.webp
            # Output: chapitre_1000_page_001.tif
            page_num = os.path.splitext(file)[0].split("_")[-1]
            base_name = f"chapitre_{chap_num}_page_{page_num}.tif"

            path_src = os.path.join(input_dir, file)
            path_dst_color = os.path.join(config.PROC_IMG_DIR, base_name)
            path_dst_bw = os.path.join(config.PROC_BW_DIR, base_name)

            if os.path.exists(path_dst_color):
                continue  # Already done

            try:
                with Image.open(path_src) as img:
                    # Color (LAB)
                    img.convert("LAB").save(path_dst_color, format="TIFF", compression="tiff_deflate")
                    # Black & White (L)
                    img.convert("L").save(path_dst_bw, format="TIFF", compression="tiff_deflate")
            except Exception as e:
                print(f"  ⚠️ Error on {file}: {e}")


if __name__ == "__main__":
    run()