import os
import numpy as np
import tifffile as tiff
import config

# Default chunk size
CHUNK_SIZE = (512, 512)  # (Height, Width)
OVERLAP_RATIO = 0.1  # 10% overlap to avoid edge artifacts


def get_axis_coords(total_size, chunk_size, overlap_ratio=0.1):
    """
    Generate starting coordinates to cover 100% of the axis.
    If the image does not divide perfectly, the last chunk is aligned to the end.
    """
    if total_size < chunk_size:
        return []

    stride = int(chunk_size * (1 - overlap_ratio))
    coords = []

    # 1. Normal advance
    current = 0
    while current + chunk_size <= total_size:
        coords.append(current)
        current += stride

    # 2. Border assurance: Always add a chunk aligned to the end
    # to ensure no pixel is lost from the right/bottom edge.
    final_start = total_size - chunk_size
    if final_start not in coords:
        coords.append(final_start)

    return sorted(list(set(coords)))

def chunkify(img_path, chunk_size=(512, 512), overlap_ratio=0.1):
    ch, cw = chunk_size

    try:
        # tifffile is often more robust for scientific/ML tiff
        img = tiff.imread(img_path)
    except Exception as e:
        print(f"Error reading {os.path.basename(img_path)}: {e}")
        return 0

    # Handle dimensions (H, W) or (H, W, C)
    if img.ndim == 2:
        H, W = img.shape
        C = None
    elif img.ndim == 3:
        H, W, C = img.shape
    else:
        return 0

    # If image is smaller than chunk, skip it (or we could pad)
    if H < ch or W < cw:
        return 0

    base_dir = os.path.dirname(img_path)
    file_name = os.path.basename(img_path)
    base_name = os.path.splitext(file_name)[0]

    # Calculate coordinates
    y_coords = get_axis_coords(H, ch, overlap_ratio)
    x_coords = get_axis_coords(W, cw, overlap_ratio)

    chunk_count = 0

    for y in y_coords:
        for x in x_coords:
            # Name: original_chunk_01.tif
            out_name = f"{base_name}_chunk{chunk_count:03d}.tif"
            out_path = os.path.join(base_dir, out_name)

            if C is None:
                # Black & White
                patch = img[y : y + ch, x : x + cw]
                tiff.imwrite(out_path, patch, compression="deflate")
            else:
                # Color (LAB)
                patch = img[y : y + ch, x : x + cw, :]
                tiff.imwrite(out_path, patch, compression="deflate", photometric="CIELAB")

            chunk_count += 1

    return chunk_count

def process_folder(folder_path, chunk_size, overlap_ratio):
    if not os.path.exists(folder_path):
        return 0, 0

    # Only take images that are NOT ALREADY chunks
    files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".tif") and "_chunk" not in f]
    )

    if not files:
        return 0, 0

    print(f" Processing {os.path.basename(folder_path)} ({len(files)} images)...")

    count_created = 0
    count_deleted = 0

    for f in files:
        full_path = os.path.join(folder_path, f)

        nb_chunks = chunkify(full_path, chunk_size, overlap_ratio)

        if nb_chunks > 0:
            # If chunking succeeded, delete the original large file
            # to keep only chunks in the final dataset
            os.remove(full_path)
            count_created += nb_chunks
            count_deleted += 1

    return count_deleted, count_created


def run():
    print(f"\n=== STEP 3: CHUNKING ({CHUNK_SIZE}) ===")

    # List of folders to process (Train + Val / Images + BW)
    target_dirs = [
        os.path.join(config.TRAIN_DIR, "images"),
        os.path.join(config.TRAIN_DIR, "bw"),
        os.path.join(config.VAL_DIR, "images"),
        os.path.join(config.VAL_DIR, "bw"),
    ]

    total_chunks = 0

    for d in target_dirs:
        deleted, created = process_folder(d, CHUNK_SIZE, OVERLAP_RATIO)
        if created > 0:
            print(f"   -> {deleted} originals deleted, {created} chunks created.")
            total_chunks += created

    if total_chunks == 0:
        print(" No chunks created. Did you run step 2?")
    else:
        print(f" Done. Total dataset chunks: {total_chunks}")

if __name__ == "__main__":
    run()