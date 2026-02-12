import sys
import os
import shutil

# Import local step scripts
import step0_scrap
import step1_convert
import step2_split
import step3_chunk


def make_data_tree(reset=False):
    """Create the data directory tree.

    If `reset` is True, remove only the data subfolders (preserve data/loading).
    """
    base_data = "data"
    data_subdirs = ["train", "test", "temp_scraping"]

    dirs_to_create = [
        "data/temp_scraping",
        "data/train/color",
        "data/train/gray",
        "data/test/color",
        "data/test/gray",
        "data/test/colored_by_cluster",
    ]

    if reset:
        print("[SETUP] Removing existing data folders (keeping data/loading)...")
        for sub in data_subdirs:
            path = os.path.join(base_data, sub)
            if os.path.exists(path):
                try:
                    shutil.rmtree(path)
                    print(f" - Removed: {path}")
                except Exception as e:
                    print(f" - Error removing {path}: {e}")

    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

    if reset:
        print("[SETUP] Data folders recreated.")
    else:
        print("[SETUP] Directory tree verified.")


def cleanup_temp():
    """Remove the temporary scraping folder if present."""
    temp_dir = "data/temp_scraping"
    if os.path.exists(temp_dir):
        print(f"Removing temporary folder {temp_dir}...")
        try:
            shutil.rmtree(temp_dir)
            print("Cleanup done.")
        except Exception as e:
            print(f"Error during cleanup: {e}")


def run_pipeline(new_data_tree=False):
    print("==========================================")
    print(f" 0. SETUP (Reset={new_data_tree})")
    print("==========================================")
    make_data_tree(reset=new_data_tree)

    print("\n==========================================")
    print(" 1. SCRAPING (to data/temp_scraping)")
    print("==========================================")
    step0_scrap.main()

    print("\n==========================================")
    print(" 2. CONVERSION (to data/train)")
    print("==========================================")
    step1_convert.main()

    print("\n==========================================")
    print(" 3. SPLIT (from data/train to data/test)")
    print("==========================================")
    step2_split.main()

    print("\n==========================================")
    print(" 4. CHUNKING (image tiling)")
    print("==========================================")
    step3_chunk.main()

    print("\n==========================================")
    print(" 5. CLEANUP")
    print("==========================================")
    cleanup_temp()

    print("\n>>> PIPELINE FINISHED <<<")
    print("Final chunked data is available in:")
    print(" - data/train/color & data/train/gray")
    print(" - data/test/color  & data/test/gray")


if __name__ == "__main__":
    # True = remove and recreate data folders
    run_pipeline(new_data_tree=True)