import os
import math
import numpy as np
import tifffile as tiff

def get_axis_coords(total_size, chunk_size, overlap_ratio=0.1):
    """
    Generates starting coordinates (x or y) using the following logic:
    - Move forward by (1 - overlap_ratio) * chunk_size
    - If the remaining space at the end is greater than chunk_size / 2,
      add a final chunk aligned with the image border.
    """
    coords = []
    stride = int(chunk_size * (1 - overlap_ratio))
    
    # 1. Standard loop
    current = 0
    while current + chunk_size <= total_size:
        coords.append(current)
        current += stride
    
    # 2. Handling the "remainder"
    if len(coords) > 0:
        last_end = coords[-1] + chunk_size
        remaining = total_size - last_end
        
        # Condition: if what remains is larger than half a chunk
        if remaining > (chunk_size / 2):
            # Force a chunk that ends exactly at the image boundary
            final_start = total_size - chunk_size
            # Check that it has not already been added
            # (case where the image size is an exact multiple)
            if final_start != coords[-1]:
                coords.append(final_start)
    
    return coords

def chunkify(img_path, chunk_size=(512, 512), overlap_ratio=0.1):
    ch, cw = chunk_size
    
    try:
        img = tiff.imread(img_path)
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return 0

    # Handle dimensions (H, W, C) or (H, W)
    if img.ndim == 2:
        H, W = img.shape
        C = None
    elif img.ndim == 3:
        H, W, C = img.shape
    else:
        return 0

    # If the image is smaller than the chunk, ignore it (or we could pad it)
    if H < ch or W < cw:
        return 0

    base_dir = os.path.dirname(img_path)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Get smart coordinates
    y_coords = get_axis_coords(H, ch, overlap_ratio)
    x_coords = get_axis_coords(W, cw, overlap_ratio)

    chunk_id = 0
    
    for y in y_coords:
        for x in x_coords:
            
            # Définition du nom de fichier de sortie
            out_name = f"{base_name}_chunk{chunk_id:02d}.tif"
            out_path = os.path.join(base_dir, out_name)

            if C is None:
                # Image en niveaux de gris (Channel L unique)
                patch = img[y:y+ch, x:x+cw]
                # Pour le noir et blanc, le défaut "MinIsBlack" de tifffile est correct
                tiff.imwrite(out_path, patch, compression="deflate")
            else:
                # Image couleur (LAB)
                patch = img[y:y+ch, x:x+cw, :]
                # On force l'interprétation LAB dans les métadonnées TIFF
                tiff.imwrite(out_path, patch, compression="deflate", photometric='CIELAB') 

            chunk_id += 1

    return chunk_id

def process_folder(folder_path, chunk_size, overlap_ratio=0.1):
    if not os.path.exists(folder_path):
        return

    files = sorted([
        f for f in os.listdir(folder_path)
        if f.endswith(".tif") and "_chunk" not in f
    ])
    print(
        f"Processing {folder_path}: {len(files)} images "
        f"-> Stride ~{int(chunk_size[0] * (1 - overlap_ratio))} px"
    )

    count_created = 0
    count_deleted = 0

    for f in files:
        full_path = os.path.join(folder_path, f)
        
        # Call with overlap
        nb_chunks = chunkify(
            full_path,
            chunk_size=chunk_size,
            overlap_ratio=overlap_ratio
        )
        
        if nb_chunks > 0:
            os.remove(full_path)
            count_created += nb_chunks
            count_deleted += 1

    print(
        f" -> {count_deleted} original images deleted, "
        f"{count_created} chunks created."
    )

def main(chunk_size=(512,512)):
    target_dirs = [
        "data/train/color",
        "data/train/gray",
        "data/test/color",
        "data/test/gray"
    ]
    
    # You can change the ratio here if needed
    OVERLAP_RATIO = 0.1 

    print(
        f"Starting chunking "
        f"(Size={chunk_size}, Overlap={OVERLAP_RATIO * 100}%)"
    )
    for d in target_dirs:
        process_folder(d, chunk_size=chunk_size, overlap_ratio=OVERLAP_RATIO)
    print("Chunking completed.")

if __name__ == "__main__":
    main()