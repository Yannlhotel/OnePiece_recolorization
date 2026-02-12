# Cluster utilisation 

## 📂 Project Structure

Here is the required file organization:

```plaintext
.
├── Makefile              # Orchestrator (Main commands)
├── Dockerfile            # Environment definition (PyTorch, OpenCV...)
├── requirements.txt      # Python dependencies
├── colorize.py           # Training and inference script
├── README.md             # This file
├── yaml_files/           # Kubernetes configuration
│   ├── pvc-color.yaml    # Persistent storage (5Gi)
│   ├── tool-pod-color.yaml # Utility pod for data transfer
│   └── colorize-job.yaml # Training job on GPU V100
└── data/                 # Your local data
    ├── train/
    │   ├── gray/         # Training images (B&W)
    │   └── color/        # Training images (Color - Ground truth)
    └── test/
        ├── color/        # Test images to colorize
```

⚙️ Prerequisites

Before you start, make sure you have:

- Docker Desktop launched and configured
- Kubectl configured with cluster access
- Make installed (via Chocolatey on Windows or native on Linux/Mac)
- Data properly placed in the data/ folder

🚀 Quick Setup

Open the Makefile and modify the NAMESPACE variable at the beginning of the file to match your identifier:

```makefile
# In the Makefile
NAMESPACE := yalh1571  <-- Replace with YOUR namespace
```

🛠️ Usage (Workflow)

Thanks to the Makefile, you only need a few commands to manage everything.

## 1. Initialization (Do this once)

This command builds the Docker image, pushes it to the registry, creates the storage volume (PVC), launches the utility pod, and sends your local data to the cluster.

```bash
make setup
```

**Note:** Data transfer may take time depending on your dataset size.

## 2. Development Cycle (The magic command ✨)

When you modify your code (colorize.py) or hyperparameters, simply run:

```bash
make update
```

This command will automatically:

- Rebuild and push the new Docker image
- Delete the old Job if it exists
- Launch the new Job on the GPU V100
- Wait for the pod to start
- Display logs in real time

## 3. Retrieve Results

Once the work is complete (visible in the logs), download the colorized images to your local machine:

```bash
make get-results
```

Results will be available in `data/test/colored_by_cluster/`.

## 4. Cleanup

To delete all resources on the cluster (Job, Pod, PVC) and free up space:

```bash
make clean
```

## 🧠 Technical Architecture

### Kubernetes Components

**PVC (pvc-color):** A ReadWriteMany volume of 5Gi or 10Gi. It stores the dataset and results so they persist across Job restarts.

**Tool Pod (tool-pod-color):** A lightweight Debian container that stays idle (sleep infinity). It serves as a "gateway" to copy files (kubectl cp) to the shared volume.

**GPU Job (colorize-job):** The PyTorch container that runs the script. It mounts the PVC in /volume, reads the data, trains the model, and saves the results.

---

## How Can Another User Adapt This for Their Setup?

### 1. Get the Project

They need to copy the entire folder (with the Makefile, Dockerfile, Python scripts, and yaml_files folder) to their own machine or workspace.

### 2. Adapt the Makefile

They must open the Makefile and modify the NAMESPACE variable at the top to use their own identifier:

```makefile
# In the Makefile for user quco4265
NAMESPACE := quco4265  # <--- Replace yalh1571 with their ID
REGISTRY := registry.datexis.com
```

### 3. Adapt the Job YAML File

This is the tricky part! The file `yaml_files/colorize-job.yaml` has a hardcoded Docker image path. They must modify the `image` line to point to their registry (the one defined in their Makefile):

**File: yaml_files/colorize-job.yaml**

```yaml
containers:
  - name: colorize-container
    # They must replace 'yalh1571' with 'quco4265' here too!
    image: registry.datexis.com/quco4265/colorize-app:latest
    # ...
```

### 4. Launch the Setup (make setup)

Since they don't have access to your data (your PVC), they must create their own volume and send their own data to it (or a copy of yours if they have it locally).

They run:

```bash
make setup
```

This will:

- Build the Docker image and push it to `registry.datexis.com/quco4265/...`
- Create their own PVC `pvc-color` in their namespace
- Send their local data to their PVC

### 5. Run the Script

Finally, they can execute the job exactly like you:

```bash
make run-job
make logs
```

---

## Summary of Differences

| Resource | Yours (yalh1571) | Theirs (quco4265) |
|----------|-----------------|------------------|
| Namespace | yalh1571 | quco4265 |
| Docker Image | .../yalh1571/colorize-app | .../quco4265/colorize-app |
| Data (PVC) | Your volume pvc-color | Their volume pvc-color (independent) |

Each person works on their own isolated "island". If they want to work with the same data as you, you must give them the files (via USB, Git, Drive, etc.) so they can put them in their `data/` folder locally before running `make setup`.
