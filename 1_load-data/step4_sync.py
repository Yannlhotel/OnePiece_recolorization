import os
import shutil
import time
import config

def run():
    print(f"\n=== STEP 4: SYNC TO CLUSTER ===")

    # 1. Define absolute paths to avoid errors
    src = os.path.abspath(config.FINAL_DIR)
    dst = config.CLUSTER_DATA_DIR  # Already in abspath via config
    dst_parent = os.path.dirname(dst)  # The "ColorizationOnCluster" folder

    print(f"📂 Source      : {src}")
    print(f"📂 Destination : {dst}")

    # 2. Safety checks
    if not os.path.exists(src):
        print(f"❌ ERROR: Source folder does not exist: {src}")
        print("   -> First run steps 2 (Split) and 3 (Chunking).")
        return

    # Verify that cluster parent folder exists (ColorizationOnCluster)
    # Otherwise we risk copying 'data' anywhere.
    if not os.path.exists(dst_parent):
        print(f"⚠️  WARNING: Parent folder '{dst_parent}' does not exist.")
        create = input("   Create it? (y/n): ").lower()
        if create == "y":
            os.makedirs(dst_parent, exist_ok=True)
        else:
            print("❌ Cancelled.")
            return

    # 3. Cleanup (Remove old 'data' folder from cluster)
    if os.path.exists(dst):
        print("🧹 Cleaning up existing destination folder...")
        try:
            shutil.rmtree(dst)
            # Small pause to let filesystem release locks (Windows)
            time.sleep(0.5)
        except Exception as e:
            print(f"❌ Error removing {dst}: {e}")
            return

    # 4. Copy (Create new 'data' folder identical to source)
    print("🚀 Copying in progress (this may take a moment)...")
    try:
        shutil.copytree(src, dst)
        print(f"✅ SUCCESS: Dataset synced to {dst}")

        # Quick verification of content
        nb_files = sum([len(files) for r, d, files in os.walk(dst)])
        print(f"   -> {nb_files} files copied.")

    except Exception as e:
        print(f"❌ Error during copy: {e}")


if __name__ == "__main__":
    run()