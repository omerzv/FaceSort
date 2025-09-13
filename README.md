# 🎯 Face Clustering System v2.0

**Advanced face clustering with hybrid algorithms, interactive chat interface, and production‑ready performance.**

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8%2B-informational">
  <img alt="CUDA optional" src="https://img.shields.io/badge/CUDA-optional-blue">
  <img alt="Platforms" src="https://img.shields.io/badge/OS-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey">
  <img alt="License" src="https://img.shields.io/badge/License-Choose%20one-important">
</p>

---

## 📚 Table of Contents

* [Quick Start](#-quick-start)
* [Key Features](#-key-features)
* [Performance](#-performance)
* [Interactive Chat Interface](#-interactive-chat-interface)
* [Configuration](#-configuration)
* [Advanced Clustering](#-advanced-clustering)
* [Quality Assessment](#-quality-assessment)
* [Installation](#-installation)
* [Project Structure](#-project-structure)
* [Usage Examples](#-usage-examples)
* [Output](#-output)
* [Troubleshooting](#-troubleshooting)
* [Roadmap](#-roadmap)
* [Contributing](#-contributing)
* [License](#-license)
* [Acknowledgements](#-acknowledgements)

---

## ⚡ Quick Start

```bash
# 1) Create virtual environment (recommended)
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the application
python face_clustering_main.py                                    # Interactive mode
python face_clustering_main.py --input ./photos --output ./results  # Direct CLI
```

> **Tip:** GPU is optional. If CUDA is installed, embeddings and clustering steps will automatically leverage the GPU.

---

## 🎯 Key Features

* **🧠 Hybrid Clustering** — HDBSCAN → DBSCAN noise rescue → intelligent cluster merging.
* **💬 Interactive Chat Interface** — Real‑time parameter tuning with explanations.
* **⚡ GPU Acceleration** — CUDA support with batch processing optimization.
* **🎨 Rich CLI Visualization** — Progress bars, statistics, and cluster summaries.
* **⚙️ Configurable** — YAML config with environment variable overrides.
* **🔧 Quality Assessment** — Blur detection, brightness analysis, pose confidence.
* **📊 System Monitoring** — CPU, GPU, and memory usage tracking.
* **🎭 Typing Animation** — Natural chat experience with customizable speed.

---

## 🚀 Performance

* **Batch Processing** — Optimized to saturate GPU utilization.
* **Parallel Processing** — Multi‑core CPU support.
* **Memory Management** — Efficient cleanup and resource handling.
* **Smart Caching** — Avoids redundant computations for repeated runs.

<details>
<summary>High‑level pipeline</summary>

```
[Load Images] -> [Detect Faces] -> [Compute Embeddings] -> [HDBSCAN]
                                         |                     |
                                         v                     |
                              [Noise Points Identified]        |
                                         |                     |
                                         v                     v
                                   [DBSCAN Rescue] -> [Similarity‑based Merge]
                                                        |
                                                        v
                                               [Quality‑Weighted Ranking]
```

</details>

---

## 💬 Interactive Chat Interface

```bash
python face_clustering_main.py
```

**Available Commands**

* `input <path>` — Set photo directory
* `output <path>` — Set results directory
* `run` — Start face clustering
* `tune` — View all parameters with explanations
* `set <param> <value>` — Adjust clustering parameters
* `config` — Show current settings
* `status` — System resource monitoring
* `help` — Command reference

**Chat Features**

* Real‑time typing animation (configurable speed)
* Parameter tuning with detailed explanations
* Progress visualization and statistics
* Smart length cap: very long outputs can be printed instantly

---

## 🔧 Configuration

**Quick Parameter Tuning (from chat):**

```bash
>>> tune                      # Show all parameters
>>> set eps 0.20              # Stricter clustering (temporary)
>>> set eps 0.20 --cfg        # Persist to config file
>>> set min_samples 3         # Larger minimum clusters
>>> typing fast               # Speed up chat animation
>>> typing cap 200            # Long messages print instantly
```

**Environment Variables**

```
FACESORTER_EPS=0.20
FACESORTER_DEVICE=cuda:0      # or 'cpu'
```

**Config File Support**

* `config.yaml` — Main configuration file (included)
* Environment variables override file values
* Runtime tuning via chat persists when using `--cfg`

---

## 🎛️ Advanced Clustering

**Hybrid Algorithm**

1. **HDBSCAN** — Initial density‑based clustering.
2. **DBSCAN Rescue** — Recover noise points with relaxed parameters.
3. **Cluster Merging** — Intelligent similarity‑based merging.
4. **Quality Weighting** — Prioritizes high‑quality faces.

**Key Parameters**

* `eps` (default: `0.22`) — Similarity threshold (lower = stricter)
* `min_samples` (default: `2`) — Minimum faces per cluster
* `algorithm` (default: `hybrid`) — Clustering method
* `similarity_threshold` (default: `0.80`) — Cluster merge threshold

---

## 📊 Quality Assessment

**Automatic Quality Scoring**

* **Blur Detection** — Laplacian variance analysis
* **Brightness Analysis** — Optimal lighting detection
* **Pose Confidence** — Face orientation scoring
* **Distance to Centroid** — Cluster cohesion measurement

---

## 🛠️ Installation

### System Requirements

* Python 3.8+
* CUDA‑compatible GPU (optional but recommended)
* 4GB+ RAM (8GB+ recommended)

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd face-clustering-system
```

### Step 2: Create Virtual Environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Manual Installation**

```bash
pip install insightface>=0.7.3 opencv-python>=4.8.0 scikit-learn>=1.3.0
pip install hdbscan>=0.8.33 numpy>=1.24.0 pyyaml>=6.0.1 colorama>=0.4.4
```

> **Why a virtual environment?**
> Isolates dependencies, prevents conflicts, and makes your setup reproducible.

---

## 📁 Project Structure

```
├── core/                    # Core processing modules
│   ├── face_detection.py    # InsightFace-based detection
│   ├── face_clustering.py   # Hybrid clustering algorithm
│   ├── face_embeddings.py   # ONNX-optimized embeddings
│   ├── face_pipeline.py     # Main processing pipeline
│   └── models.py            # Data structures
├── chat/                    # Interactive chat interface  
│   ├── base_interface.py    # Chat foundation
│   ├── command_processor.py # Command handling
│   ├── parameter_tuner.py   # Real-time parameter tuning
│   └── system_monitor.py    # Resource monitoring
├── utils/                   # Utilities and helpers
│   ├── config.py            # Configuration management
│   ├── colors.py            # CLI visualization
│   ├── typing_animation.py  # Chat typing effects
│   └── logger.py            # Structured logging
└── face_clustering_main.py  # Main entry point
```

---

## 🎯 Usage Examples

**Basic Clustering**

```bash
python face_clustering_main.py --input photos/ --output results/
```

**Interactive Mode**

```bash
python face_clustering_main.py
>>> input photos/
>>> output results/
>>> set eps 0.18           # Stricter clustering
>>> set min_samples 3      # Require at least 3 faces per cluster
>>> run
```

**Advanced Configuration**

```bash
>>> tune                   # View all parameters with explanations
>>> set algorithm hybrid   # Use hybrid clustering
>>> set min_samples 2      # Allow 2-face clusters
>>> config                 # Review current settings
>>> run                    # Start processing
```

---

## 📈 Output

**Generated Results**

* **Organized Clusters** — `cluster_001/`, `cluster_002/`, …
* **Face Images** — High‑quality cropped faces per cluster
* **Statistics** — Processing metrics and performance data
* **CLI Visualization** — Rich terminal output with progress tracking

---

## 🐛 Troubleshooting

* **GPU OOM / Memory Limits** — Lower `batch_size` in config.
* **Too Many Clusters** — Increase `eps`.
* **Too Few Clusters** — Decrease `eps`.
* **Chat Feels Slow** — Use `typing fast` or `typing off`; try `typing cap 200`.
* **Slow Startup** — First run may download models; subsequent runs are faster.

---

## 🗺️ Roadmap

* [ ] Optional Web UI (local)
* [ ] Export embeddings & metadata (Parquet/Feather)
* [ ] Interactive 3D visualization (UMAP/PCA)
* [ ] Semi‑supervised merge/split tooling
* [ ] Dataset audit reports (per‑cluster quality)

---

## 🤝 Contributing

Contributions are welcome! Please open an issue to discuss larger changes, or submit a PR with:

* A clear description & motivation
* Minimal reproducible example (if bug)
* Tests and/or before/after results when relevant

---

## 📄 License

Choose a license (e.g., MIT) and update the badge + this section.

---

## 🙏 Acknowledgements

Built with ❤️ using **InsightFace**, **HDBSCAN**, **scikit‑learn**, **OpenCV**, and modern Python practices.
