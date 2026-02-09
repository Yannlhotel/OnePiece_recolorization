import os
import random
import shutil

def main():
    SRC = "data/all_rgb"
    TRAIN_DIR = "data/train_rgb"
    TEST_DIR = "data/test_rgb"
    
    files = [f for f in os.listdir(SRC) if f.endswith(".png")]
    random.seed(42)
    random.shuffle(files)
    
    split_idx = int(len(files) * 0.8)
    train_files = files[:split_idx]
    test_files = files[split_idx:]

    for f, target in [(train_files, TRAIN_DIR), (test_files, TEST_DIR)]:
        os.makedirs(target, exist_ok=True)
        for file in f:
            shutil.move(os.path.join(SRC, file), os.path.join(target, file))
    print(f"Split terminé : {len(train_files)} train, {len(test_files)} test.")

if __name__ == "__main__":
    main()