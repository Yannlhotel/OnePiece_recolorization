# 🏴‍☠️ One Piece Recolorization
## 🎯 Project Objective
The goal of this project is to automatically recolorize One Piece manga chapters. While the official release is in black and white, high-quality fan-colored versions are available.

To achieve this, we utilize a supervised learning approach:

We take the existing fan-colored images (Ground Truth).

We mathematically convert them to grayscale (Input).

We train Machine Learning models to reconstruct the color information from the grayscale input.

## 📂 Project Structure
Here is the organization of the repository (current folder names):
```
.
├── .gitignore
├── README.md
├── 1_load-data/                         # 💾 Data ingestion & preprocessing pipeline
├── 2-1_AlgoColorization/                # 🧪 Development Sandbox (personal workspaces)
│   ├── colorize_example_for_cluster.py
│   ├── perso/
│   │   ├── Emma/
│   │   ├── Quentin/
│   │   └── Yann/
│   └── tools/                           # Local helper scripts
│       ├── imshow.py
│       ├── reconstruct_DeepColor512.py
│       └── reconstruct_Unet.py
├── 2_ColorizationOnCluster/             # 🚀 Cluster Deployment (Makefile + Docker)
│   ├── colorize.py
│   ├── colorizeMAE.py
│   ├── colorizeMSE.py
│   ├── Dockerfile
│   ├── Makefile
│   ├── Makefile.linux
│   ├── Makefile.win
│   ├── README.md
│   ├── requirements.txt
│   ├── data/
│   └── yaml_files/
├── 4_results/                            # Final output storage (trained model outputs)
│   ├── 1_DeepColor512_MAE/
│   ├── 1_DeepColor512_MSE/
│   ├── 2_Unet_MAE/
│   └── 2_Unet_MSE/
└── 5_best_model_test/                    # Best model storage (with colored images)
```

### 1. Algorithm Development (/2-1_AlgoColorization)
We develop and prototype our algorithms locally within the `2-1_AlgoColorization` folder. Each team member (Emma, Quentin, Yann) has a dedicated subfolder for experimentation (under `perso/`).

### 2. Cluster Deployment (/2_ColorizationOnCluster)
Once an algorithm is stable, the logic is transferred to the `colorize.py` script located in the `2_ColorizationOnCluster` directory. We then use the Makefile workflow (detailed in that folder's README) to deploy the job to the GPU cluster for training on the full dataset.

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

For cluster deployment, refer to [2_ColorizationOnCluster/README.md](2_ColorizationOnCluster/README.md) for detailed instructions on:
- Setting up Docker and Kubernetes
- Configuring namespace and storage
- Managing the deployment pipeline

## 🚀 Quick Start

### 1. Prepare Your Data

Use the data pipeline to fetch and process One Piece manga images:

```bash
python 1_load-data/main.py
```

This interactive tool will:
- Scrape images from source
- Convert formats (WebP to TIFF)
- Split into train/validation (80/20)
- Chunk images into 512×512 patches
- Sync to cluster storage

See [1_load-data/README.md](1_load-data/README.md) for detailed configuration options.

### 2. Develop Locally

Test algorithms in the development folder:

```bash
cd 2-1_AlgoColorization
python your_script.py
```

Each team member has a dedicated subfolder for experimentation.

### 3. Deploy to Cluster

Once your algorithm is stable, move it to `2_ColorizationOnCluster` and deploy:

```bash
cd 2_ColorizationOnCluster
make setup       # One-time setup: build image, deploy infra and upload data
make run-job     # Start training job on the cluster
make logs        # Monitor job logs
make get-results # Download colorized outputs
```

Refer to [2_ColorizationOnCluster/README.md](2_ColorizationOnCluster/README.md) for complete workflow documentation.

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

Results are stored in the `4_results/` directory with structure:
```
4_results/
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

The best performing model is stored in [5_best_model_test/](5_best_model_test/):
- Pre-trained weights
- Test images
- Colorized outputs for comparison

### Running Inference

To colorize images using a trained model:

```bash
# Example: run the provided inference script for a trained model
python 4_results/1_DeepColor512_MSE/colorize.py 
```

## 👥 Team

- **Emma** - 2-1_AlgoColorization/perso/Emma/
- **Quentin** - 2-1_AlgoColorization/perso/Quentin/
- **Yann** - 2-1_AlgoColorization/perso/Yann/

## 📚 References
- Colorful Image Colorization, Richard Zhang, Phillip Isola, Alexei A. Efros : https://arxiv.org/pdf/1603.08511
- Color Parsing, Hui Ren, Jia Li, Nan Gao : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8944253
- U-net , https://medium.com/data-science/colorizing-black-white-images-with-u-net-and-conditional-gan-a-tutorial-81b2df111cd8
- U-net, http://conference.ioe.edu.np/ioegc10/papers/ioegc-10-124-10162.pdf
- With also the help of Gemini3 for cluster setup and load-data pipeline.