import os
import shutil
import step0_scrap
import step1_convert
import step2_split
import step3_chunk
import step4_sync
import config


def clean_raw_data():
    """Remove raw and intermediate image folders."""
    raw_path = config.RAW_DIR
    processed_path = config.PROCESSED_DIR

    print(f"\nDISK CLEANUP")
    print(f"   Targets: \n   - {raw_path}\n   - {processed_path}")

    if input("Confirm permanent deletion? (yes/no): ").strip().lower() in ["yes", "oui", "y"]:
        try:
            if os.path.exists(raw_path):
                shutil.rmtree(raw_path)
                os.makedirs(raw_path, exist_ok=True)
            if os.path.exists(processed_path):
                shutil.rmtree(processed_path)
                os.makedirs(processed_path, exist_ok=True)
            print("Cleanup completed.")
        except Exception as e:
            print(f"Cleanup error: {e}")
    else:
        print("Cancelled.")


def main():
    while True:
        print("\n=======================================")
        print("  ONE PIECE IMPORT PIPELINE")
        print("=======================================")
        print(f"   Dataset Source : {config.ROOT_DIR}")
        print(f"   Cluster Dest   : {config.CLUSTER_DATA_DIR}")
        print("---------------------------------------")
        print("1. RUN ALL (Steps 0 to 4)")
        print("---------------------------------------")
        print("2. Step 0: Scraping (Download)")
        print("3. Step 1: Conversion (WebP -> TIFF)")
        print("4. Step 2: Split (Train/Val)")
        print("5. Step 3: Chunking (512x512 tiling)")
        print("6. Step 4: Copy to Cluster (Sync)")
        print("---------------------------------------")
        print("7. Cleanup (Remove temporary files)")
        print("0. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            # Full workflow
            step0_scrap.run()
            step1_convert.run()
            step2_split.run()
            step3_chunk.run()
            step4_sync.run()

            print("\n Complete pipeline finished.")
            clean_ask = input("Remove temporary files (Raw/Processed) to save space? (y/n): ").lower()
            if clean_ask == "y":
                clean_raw_data()

        elif choice == "2":
            step0_scrap.run()
        elif choice == "3":
            step1_convert.run()
        elif choice == "4":
            step2_split.run()
        elif choice == "5":
            step3_chunk.run()
        elif choice == "6":
            step4_sync.run()
        elif choice == "7":
            clean_raw_data()
        elif choice == "0":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()