# 🏴‍☠️ One Piece Recolorization
## 🎯 Project Objective
The goal of this project is to automatically recolorize One Piece manga chapters. While the official release is in black and white, high-quality fan-colored versions are available.

To achieve this, we utilize a supervised learning approach:

We take the existing fan-colored images (Ground Truth).

We mathematically convert them to grayscale (Input).

We train Machine Learning models to reconstruct the color information from the grayscale input.

## 📂 Project Structure
Here is the organization of the repository:
```
.
├── .gitignore
├── README.md
├── AlgoColorization/                    # 🧪 Development Sandbox
│   ├── colorize_example_for_cluster.py
│   ├── Emma/                            # Emma's dev space
│   ├── Quentin/                         # Quentin's dev space
│   └── Yann/                            # Yann's dev space
├── ColorizationOnCluster/               # 🚀 Cluster Deployment
│   ├── colorize.py                      # Main production script
│   ├── Dockerfile                       # Container definition
│   ├── Makefile                         # Automation router
│   ├── Makefile.linux                   # Linux/Mac commands
│   ├── Makefile.win                     # Windows commands
│   ├── README.md                        # Cluster specific documentation
│   ├── requirements.txt                 # Python dependencies
│   ├── data/                            # Data structure for the cluster
│   │   ├── test/
│   │   │   ├── color/
│   │   │   ├── colored_by_cluster/
│   │   │   └── gray/
│   │   └── train/
│   │       ├── color/
│   │       └── gray/
│   └── yaml_files/                      # Kubernetes configurations
│       ├── colorize-job.yaml
│       ├── pvc-color.yaml
│       └── tool-pod-color.yaml
├── dataset_one_piece/                   # 💾 Local Dataset (shared across team)
│   ├── test/
│   │   ├── images/
│   │   ├── colored_by_cluster/          # Results retrieved from cluster
│   │   └── bw/
│   └── val/
│       ├── images/
│       └── bw/
├── results/                             # Final output storage
└── best_model_test/                     # Best model storage (with colored images)
```

### 1. Algorithm Development (/AlgoColorization)
We develop and prototype our algorithms locally within the AlgoColorization folder. Each team member (Emma, Quentin, Yann) has a dedicated subfolder for experimentation.

### 2. Cluster Deployment (/ColorizationOnCluster)
Once an algorithm is stable, the logic is transferred to the colorize.py script located in the ColorizationOnCluster directory. We then use the Makefile workflow (detailed in the folder's README) to deploy the job to the GPU cluster for training on the full dataset.

## 📦 Installation & Setup

### Prerequisites
- Python 3.8+
- PyTorch with CUDA support (for GPU acceleration)
- Docker and Kubernetes (for cluster deployment)
- Make (for workflow automation)

### Local Development Setup

Clone the repository and install dependencies:

```bash
# Clone the repository
git clone <repository-url>
cd Projet_OP_reco

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Cluster Setup

For cluster deployment, refer to [ColorizationOnCluster/README.md](ColorizationOnCluster/README.md) for detailed instructions on:
- Setting up Docker and Kubernetes
- Configuring namespace and storage
- Managing the deployment pipeline

## 🚀 Quick Start

### 1. Prepare Your Data

Use the data pipeline to fetch and process One Piece manga images:

```bash
python load-data/main.py
```

This interactive tool will:
- Scrape images from source
- Convert formats (WebP to TIFF)
- Split into train/validation (80/20)
- Chunk images into 512×512 patches
- Sync to cluster storage

See [load-data/README.md](load-data/README.md) for detailed configuration options.

### 2. Develop Locally

Test algorithms in the AlgoColorization folder:

```bash
cd AlgoColorization
python your_script.py
```

Each team member has a dedicated subfolder for experimentation.

### 3. Deploy to Cluster

Once your algorithm is stable, move it to ColorizationOnCluster and deploy:

```bash
cd ColorizationOnCluster
make setup      # One-time setup
make run-job    # Start training job
make logs       # Monitor job progress
make get-results   # Download results
```

Refer to [ColorizationOnCluster/README.md](ColorizationOnCluster/README.md) for complete workflow documentation.

## 🎨 Models

We compare two architectures with different loss functions:

### Model Variants

| Model | Loss | Purpose |
|-------|------|---------|
| **DeepColor512 (MAE)** | Mean Absolute Error | Lower gradient penalty |
| **DeepColor512 (MSE)** | Mean Squared Error | Higher gradient penalty |
| **U-Net (MAE)** | Mean Absolute Error | Better edge preservation |
| **U-Net (MSE)** | Mean Squared Error | Smoother predictions |

### Model Results

Results are stored in the `results/` directory with structure:
```
results/
├── 1_DeepColor512_MAE/
├── 1_DeepColor512_MSE/
├── 2_Unet_MAE/
└── 2_Unet_MSE/
```

Each model directory contains:
- `colorize.py` - Model definition and inference code + CNN architecture
- `model.pth` - Trained weights
- `metrics.txt` - Performance metrics (PSNR, SSIM)

### Evaluation Metrics

Models are evaluated using:
- **PSNR** (Peak Signal-to-Noise Ratio) - Overall image quality
- **SSIM** (Structural Similarity Index) - Perceived quality

## 📊 Results & Evaluation

### Best Model

The best performing model is stored in [best_model_test/](best_model_test/):
- Pre-trained weights
- Test images
- Colorized outputs for comparison

### Running Inference

To colorize images using a trained model:

```python
from results.DeepColor512_MSE.colorize import colorize_image
import cv2

# Load a grayscale image (which need to be 512*512)
bw_image = cv2.imread('image_bw.png', cv2.IMREAD_GRAYSCALE)

# Colorize
colored = colorize_image(bw_image)

# Save result
cv2.imwrite('image_colored.png', colored)
```

## 👥 Team

- **Emma** - AlgoColorization/Emma/
- **Quentin** - AlgoColorization/Quentin/
- **Yann** - AlgoColorization/Yann/

## 📚 References

