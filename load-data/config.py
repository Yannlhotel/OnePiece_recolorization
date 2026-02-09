import os

# --- PARAMETERS ---
START_CHAPTER = 1
END_CHAPTER = 50
MAX_PAGES = 500

# --- CHUNKING PARAMETERS ---
CHUNK_SIZE = (512, 512)
OVERLAP_RATIO = 0.1

# --- PATHS ---
ROOT_DIR = "dataset_one_piece"

# Step 0: Raw download
RAW_DIR = os.path.join(ROOT_DIR, "01_raw_webp")

# Step 1: Converted images (TIFF)
PROCESSED_DIR = os.path.join(ROOT_DIR, "02_processed_tiff")
PROC_IMG_DIR = os.path.join(PROCESSED_DIR, "images")
PROC_BW_DIR = os.path.join(PROCESSED_DIR, "bw")

# Step 2: Final dataset (to be copied)
FINAL_DIR = os.path.join(ROOT_DIR, "03_dataset_final")
TRAIN_DIR = os.path.join(FINAL_DIR, "train")
VAL_DIR = os.path.join(FINAL_DIR, "val")

# Step 3: Cluster destination
# Using abspath to resolve ".." properly
CLUSTER_DATA_DIR = os.path.abspath(os.path.join("ColorizationOnCluster", "data"))

# Create local folders
for d in [RAW_DIR, PROC_IMG_DIR, PROC_BW_DIR, TRAIN_DIR, VAL_DIR]:
    os.makedirs(d, exist_ok=True)