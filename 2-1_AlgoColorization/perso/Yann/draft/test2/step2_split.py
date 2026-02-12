import os
import shutil
import random

def main():
    # SOURCE (current location)
    TRAIN_ROOT = "data/train"
    TRAIN_COLOR = os.path.join(TRAIN_ROOT, "color")
    TRAIN_GRAY = os.path.join(TRAIN_ROOT, "gray")

    # DESTINATION (20% will be moved here)
    TEST_ROOT = "data/test"
    TEST_COLOR = os.path.join(TEST_ROOT, "color")
    TEST_GRAY = os.path.join(TEST_ROOT, "gray")

    if not os.path.exists(TRAIN_COLOR):
        print("Train folder empty. Run step1 first.")
        return

    # Create test folders
    os.makedirs(TEST_COLOR, exist_ok=True)
    os.makedirs(TEST_GRAY, exist_ok=True)

    # List all available images in Train/Color
    all_files = sorted([f for f in os.listdir(TRAIN_COLOR) if f.endswith(".tif")])
    total_files = len(all_files)

    if total_files == 0:
        print("No images to split.")
        return

    # Number of images to move (20%)
    split_count = int(total_files * 0.2)

    # Random selection
    random.seed(42)  # deterministic split
    files_to_move = random.sample(all_files, split_count)

    print(f"Moving {split_count} images ({split_count/total_files:.0%}) to the test dataset...")

    for f in files_to_move:
        # Source paths
        src_c = os.path.join(TRAIN_COLOR, f)
        src_g = os.path.join(TRAIN_GRAY, f)

        # Destination paths
        dst_c = os.path.join(TEST_COLOR, f)
        dst_g = os.path.join(TEST_GRAY, f)

        # Move files to remove them from train
        if os.path.exists(src_c) and os.path.exists(src_g):
            shutil.move(src_c, dst_c)
            shutil.move(src_g, dst_g)
        else:
            print(f"Warning: incomplete pair for {f}, skipping.")

    print("Split completed.")
    print(f"Train : {len(os.listdir(TRAIN_COLOR))} images")
    print(f"Test  : {len(os.listdir(TEST_COLOR))} images")


if __name__ == "__main__":
    main()