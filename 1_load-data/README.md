# Load Data Pipeline

Automated pipeline to scrape, convert, split, chunk, and sync One Piece manga images for the colorization project.

## Overview

This pipeline processes raw manga images through 5 sequential steps:

1. **Step 0: Scraping** - Download raw images from scan-vf.net
2. **Step 1: Conversion** - Convert WebP to TIFF (LAB color + B&W)
3. **Step 2: Split** - Random train/validation split (80/20)
4. **Step 3: Chunking** - Tile images into 512×512 patches with overlap
5. **Step 4: Sync** - Copy final dataset to Cluster

## Prerequisites


### Directory Structure

Before running, ensure you have:

```
ColorizationOnCluster/
    data/              (will be created/synced by step 4)
```

## Quick Start

### Run the Interactive Menu

```bash
python load-data/main.py
```

This opens an interactive menu where you can:
- Run all steps (1–4)
- Run individual steps
- Clean temporary files

### Run All Steps

```bash
python load-data/main.py
# Then select option "1"
```

## Configuration

Edit `load-data/config.py` to customize:

```python
# Scraping range
START_CHAPTER = 1
END_CHAPTER = 50

# Max pages per chapter (limit for testing)
MAX_PAGES = 50

# Chunk size (default: 512×512)
CHUNK_SIZE = (512, 512)
OVERLAP_RATIO = 0.1  # 10% overlap for edge preservation

# Root dataset directory
ROOT_DIR = "dataset_one_piece"
```

## Step Details

### Step 0: Scraping

Downloads manga chapters from scan-vf.net.

**Location**: `step0_scrap.py`

**Output**: `dataset_one_piece/01_raw_webp/`

**Features**:
- Skips chapters already downloaded (>5 images)
- Filters small images (<10 KB) and ads
- Anti-ban delay (1–2 sec between chapters)
- Robust error handling

**Run individually**:
```bash
python -c "import load-data.step0_scrap as s; s.run()"
```

### Step 1: Conversion

Converts WebP images to TIFF format in two variants:

- **LAB color** (for input to colorization model)
- **Grayscale** (L channel, for reference)

**Location**: `step1_convert.py`

**Output**: 
- `dataset_one_piece/02_processed_tiff/images/` (color/LAB)
- `dataset_one_piece/02_processed_tiff/bw/` (grayscale)

**Features**:
- Skips already-converted images
- Robust format handling
- Preserves original chapter structure

### Step 2: Split

Randomly splits images into train (80%) and validation (20%) sets.

**Location**: `step2_split.py`

**Output**:
- `dataset_one_piece/03_dataset_final/train/` (images + bw)
- `dataset_one_piece/03_dataset_final/val/` (images + bw)

**Features**:
- Fixed seed (42) for reproducible splits
- Paired image/BW copy (integrity check)
- Comprehensive stats

### Step 3: Chunking

Tiles images into fixed-size patches with controllable overlap.

**Location**: `step3_chunk.py`

**Key Function**: `get_axis_coords()`
- Ensures 100% image coverage
- Handles edge cases (images smaller than chunk)
- Aligns final chunk to image boundary

**Output**: Same folders, original images replaced by chunks
- Naming: `chapitre_001_page_001_chunk000.tif`

**Features**:
- Configurable chunk size and overlap
- Photometric LAB preservation for color images
- Automatic removal of original images

### Step 4: Sync

Copies the final dataset to the Cluster container.

**Location**: `step4_sync.py`

**Source**: `dataset_one_piece/03_dataset_final/`

**Destination**: `ColorizationOnCluster/data/`

**Features**:
- Safety checks (folder existence)
- Full directory replacement
- File count verification
- Windows file-lock handling

## Directory Tree

After full pipeline run:

```
dataset_one_piece/
├── 01_raw_webp/              [RAW, can be deleted]
│   ├── chapitre_1/
│   │   ├── page_001.webp
│   │   └── ...
│   └── ...
├── 02_processed_tiff/        [INTERMEDIATE, can be deleted]
│   ├── images/
│   │   └── chapitre_001_page_001.tif (LAB)
│   └── bw/
│       └── chapitre_001_page_001.tif (L)
└── 03_dataset_final/         [FINAL DATASET]
    ├── train/
    │   ├── images/
    │   │   └── chapitre_001_page_001_chunk000.tif
    │   └── bw/
    │       └── chapitre_001_page_001_chunk000.tif
    └── val/
        ├── images/
        └── bw/

ColorizationOnCluster/
└── data/                      [SYNCED TO CLUSTER]
    ├── train/
    └── val/
```

## Usage Examples

### Example 1: Full Pipeline (All Steps)

```bash
cd /path/to/Projet_OP_reco
python load-data/main.py
# Select option 1
# Answer "y" when asked to clean temporary files
```

**Result**: Fully processed dataset in `ColorizationOnCluster/data/`

### Example 2: Run Only Conversion & Split

```bash
python load-data/main.py
# Select option 3 (Step 1: Conversion)
# Then select option 4 (Step 2: Split)
```

### Example 3: Re-chunk with Different Settings

Edit `config.py`:
```python
CHUNK_SIZE = (256, 256)  # Smaller chunks
OVERLAP_RATIO = 0.2      # More overlap
```

Then run:
```bash
python load-data/main.py
# Select option 5 (Step 3: Chunking)
```

### Example 4: Clean Temporary Files

```bash
python load-data/main.py
# Select option 7 (Cleanup)
```

## File Naming Convention

- **Raw**: `page_001.webp`, `page_002.webp`, ...
- **Processed**: `chapitre_001_page_001.tif`
- **Chunked**: `chapitre_001_page_001_chunk000.tif`, `chapitre_001_page_001_chunk001.tif`, ...

## Error Handling

| Error | Solution |
|-------|----------|
| "Source folder not found" (Step 4) | Run steps 2 & 3 first |
| "No images found in processed folder" (Step 2) | Run step 1 first |
| "No chunks created. Did you run step 2?" | Run step 2 first |
| Download failures (Step 0) | Network issue; retry or check chapter URL |

## Performance Notes

- **Scraping**: ~1–2 sec per chapter (includes anti-ban delay)
- **Conversion**: ~1–5 sec per image (depends on resolution)
- **Splitting**: <1 sec per image
- **Chunking**: ~2–10 sec per image (depends on overlap)
- **Sync**: Depends on disk speed; typically 5–30 min for full dataset

## Cleanup

To save disk space, remove temporary folders:

```bash
python load-data/main.py
# Select option 7 (Cleanup)
```

Or manually:
```bash
rm -rf dataset_one_piece/01_raw_webp
rm -rf dataset_one_piece/02_processed_tiff
```

## Troubleshooting

### Pipeline Hangs During Scraping

- Network issue or website blocking
- Check internet connection
- Increase delay in `step0_scrap.py` if 429 (rate limit) errors occur

### Conversion Errors

- Ensure Pillow supports WebP: `python -c "from PIL import WebPImagePlugin"`
- Try updating Pillow: `pip install --upgrade pillow`

### Out of Memory

- Reduce `MAX_PAGES` in config
- Process fewer chapters at a time
- Increase system RAM or use cloud storage

### Sync Fails on Windows

- Ensure no files are open in `ColorizationOnCluster/data/`
- Try running as Administrator
- Check disk space

## Module Reference

### config.py

Centralized configuration file with all paths and parameters.

```python
import config
print(config.TRAIN_DIR)  # Access configuration
```

### main.py

Interactive menu dispatcher.

```python
from load-data import main
main.main()  # Run CLI
```

### step*.py

Individual pipeline steps with `run()` function.

```python
from load-data import step0_scrap
step0_scrap.run()  # Run scraping
```

## Advanced Usage

### Process Specific Chapters

Edit `config.py`:
```python
START_CHAPTER = 100
END_CHAPTER = 120
```

Then run Step 0.

### Change Output Directory

Edit `config.py`:
```python
ROOT_DIR = "/mnt/external_drive/manga_data"
```

### Parallel Processing (Manual)

For large datasets, manually split chapters across machines:

**Machine 1**:
```python
config.START_CHAPTER = 1
config.END_CHAPTER = 100
```

**Machine 2**:
```python
config.START_CHAPTER = 101
config.END_CHAPTER = 377
```

Run both instances, then merge results.

## License

Project-specific pipeline for the BHT LoI Colorization project.

## Support

For issues, check:
1. Logs in terminal output
2. Internet connection (for Step 0)
3. Disk space availability
4. File permissions
5. Antivirus/firewall (may block scraping)
