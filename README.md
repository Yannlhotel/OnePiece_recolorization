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
├── .vscode/
│   └── settings.json
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
├── data/                                # 💾 Local Dataset (shared across team)
│   ├── test/
│   │   ├── color/
│   │   ├── colored_by_cluster/          # Results retrieved from cluster
│   │   └── gray/
│   └── train/
│       ├── color/
│       └── gray/
└── results/                             # Final output storage
```

## ⚙️ Workflow

We follow a two-step workflow to ensure efficiency:

### 1. Algorithm Development (/AlgoColorization)
We develop and prototype our algorithms locally within the AlgoColorization folder. Each team member (Emma, Quentin, Yann) has a dedicated subfolder for experimentation.

### 2. Cluster Deployment (/ColorizationOnCluster)
Once an algorithm is stable, the logic is transferred to the colorize.py script located in the ColorizationOnCluster directory. We then use the Makefile workflow (detailed in the folder's README) to deploy the job to the GPU cluster for training on the full dataset.