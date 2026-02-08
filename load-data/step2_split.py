import os
import shutil
import random
import config


def run():
    print("\n=== STEP 2: RANDOM SPLIT (80% Train / 20% Val) ===")

    # 1. Cleanup and prepare output folders
    # Remove old train/val folders to start fresh
    for root in [config.TRAIN_DIR, config.VAL_DIR]:
        if os.path.exists(root):
            shutil.rmtree(root)

        os.makedirs(os.path.join(root, "images"), exist_ok=True)
        os.makedirs(os.path.join(root, "bw"), exist_ok=True)

    # 2. Get ALL image files
    # List only the images folder, deduce the bw path from it
    all_files = sorted([f for f in os.listdir(config.PROC_IMG_DIR) if f.endswith(".tif")])

    total_files = len(all_files)
    if total_files == 0:
        print("❌ No images found in processed folder. Run step 1 first.")
        return

    # 3. Shuffle the complete list
    random.seed(42)  # Fixed seed ensures same shuffle if re-run
    random.shuffle(all_files)

    # 4. Calculate split
    val_count = int(total_files * 0.2)  # 20%
    val_files = all_files[:val_count]
    train_files = all_files[val_count:]

    print(f"Total Images: {total_files}")
    print(f" -> Train: {len(train_files)} images")
    print(f" -> Val  : {len(val_files)} images")

    # 5. Helper function to copy files
    def copy_list(file_list, destination_root):
        count = 0
        for filename in file_list:
            # Source paths
            src_img = os.path.join(config.PROC_IMG_DIR, filename)
            src_bw = os.path.join(config.PROC_BW_DIR, filename)

            # Destination paths
            dst_img = os.path.join(destination_root, "images", filename)
            dst_bw = os.path.join(destination_root, "bw", filename)

            # Verify and copy
            if os.path.exists(src_img) and os.path.exists(src_bw):
                shutil.copy(src_img, dst_img)
                shutil.copy(src_bw, dst_bw)
                count += 1
            else:
                print(f"⚠️ Incomplete pair skipped: {filename}")
        return count

    # 6. Execute copies
    print("Copying train files...")
    c_train = copy_list(train_files, config.TRAIN_DIR)

    print("Copying val files...")
    c_val = copy_list(val_files, config.VAL_DIR)

    print(f"\n✅ Done! {c_train + c_val} image pairs distributed.")


if __name__ == "__main__":
    run()