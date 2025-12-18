# Applied Computer Vision: Explainable & Edge AI Portfolio

A comprehensive portfolio exploring critical frontiers in Applied AI:  Medical Interpretability and Industrial Edge Optimization.

**Project Contributors:** Prahas Hegde, Rohan Sanjay Patil, Vidya Padmanabha

**Academic Context:** Portfolio 2, Computer Vision

---

## 📋 Table of Contents

- [Overview](#overview)
- [Part I:  Explainable AI in Medical Imaging](#-part-i-explainable-ai-in-medical-imaging)
- [Part II: Edge AI for Industry 4.0](#-part-ii-edge-ai-for-industry-40)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Results & Performance](#results--performance)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository contains a two-part portfolio addressing critical challenges in modern AI:

1. **Explainable AI (XAI)**: Bridging the "trust gap" in medical imaging through interpretable deep learning
2. **Edge AI**: Deploying optimized models for real-time inference on resource-constrained hardware

Both projects demonstrate the transition from theoretical ML to production-ready systems that prioritize both performance and interpretability.

---

## 🏥 Part I: Explainable AI in Medical Imaging

### The Challenge

Making opaque deep learning models transparent for clinicians in life-or-death medical decisions.  Trust in AI systems requires understanding *why* a prediction was made, not just *what* the prediction is.

### Technical Overview

**Model Architecture:**
- Base:  Fine-tuned ResNet50 for binary classification
- Task: Normal vs.  Pneumonia detection in chest X-rays
- Training Data: Public medical imaging datasets
- Framework: TensorFlow/PyTorch

**Performance Metrics:**
- Overall Accuracy: **87%**
- Recall: **94%** (Critical:  ensures pneumonia cases are not missed)
- Precision: Balanced to minimize false positives

### Explainability Pipeline

Two complementary XAI methods integrated for comprehensive interpretation:

#### 1. **Grad-CAM (Gradient-weighted Class Activation Mapping)**
- **Approach**: "White-box" method using internal model gradients
- **How it works**: Computes gradients of the classification output with respect to feature maps
- **Output**: High-resolution attention maps showing discriminative regions
- **Advantages**: 
  - Reproducible and mathematically rigorous
  - Provides precise localization of decision factors
  - Works across different architectures

#### 2. **LIME (Local Interpretable Model-agnostic Explanations)**
- **Approach**: "Black-box" method that perturbs image regions locally
- **How it works**:  Generates synthetic data around the prediction to understand local decision boundaries
- **Output**: Superpixel-level feature importance highlighting
- **Advantages**:
  - Model-agnostic (works with any model)
  - Intuitive interpretation at human-understandable scale
  - No access to internal model parameters required

### Medical Imaging Workflow

```
Medical Image (Chest X-Ray)
        ↓
   ResNet50 Model
        ↓
   Classification Output + Confidence Score
        ↓
   ┌─────────────────┬──────────────────┐
   ↓                 ↓
Grad-CAM           LIME
(High-Res)      (Superpixel)
   ↓                 ↓
Attention Maps    Feature Maps
   └─────────────────┬──────────────────┘
        ↓
Clinician Decision Support

```

---

## 🏭 Part II: Edge AI for Industry 4.0

### The Challenge

Developing a high-speed, reliable vehicle verification system that performs local inference on factory floors without cloud dependency.  The system must operate in real-time with minimal latency while maintaining accuracy standards.

### Technical Overview

**Model Optimization:**
- Base Model: MobileNetV2 (efficient architecture for edge devices)
- Target Sparsity: **50%**
- Target Accuracy: **>95%** (only minimal degradation acceptable)
- Hardware Target:  NVIDIA Jetson Nano

**Optimization Comparison:**

| Method | Sparsity | Accuracy Drop | Model Size | Inference Time |
|--------|----------|---------------|------------|-----------------|
| One-Shot Pruning | 50% | -1.2% | 6.8 MB | 65ms |
| **Iterative L2 Pruning** | 50% | -0.5% | 6.3 MB | 62ms |
| **Iterative Taylor Pruning** | 50% | -0.2% | **6.1 MB** | **~60ms** |

**Best Method:  Iterative Taylor Pruning**
- Maintains 95.2% accuracy
- Minimal performance degradation
- Optimal model compression

### Performance Achievements

✅ **Compression**: Model size reduced from 8.9 MB → 6.1 MB (31% reduction)

✅ **Latency**:  Inference at ~60ms (16. 67 FPS, exceeding >10 FPS factory requirement)

✅ **Accuracy**: 95.2% maintained with only -0.2% drop from baseline

✅ **Deployment Ready**: Successfully optimized for NVIDIA Jetson Nano deployment

### System Architecture

The complete edge AI system comprises three layers:

```
┌─────────────────────────────────────────────────┐
│         DECISION LAYER                          │
│  Industrial Ethernet → Pneumatic Sorting Gates   │
│         (Gate A / Gate B Control)               │
└────────────────────┬────────────────────────────┘
                     ↑
┌─────────────────────────────────────────────────┐
│    EDGE COMPUTING LAYER                         │
│  NVIDIA Jetson Nano                             │
│  - TensorRT Inference Engine                    │
│  - Real-time Processing                         │
│  - Local Decision Making                        │
└────────────────────┬────────────────────────────┘
                     ↑
┌─────────────────────────────────────────────────┐
│         SENSING LAYER                           │
│  - GigE Vision Overhead Cameras                 │
│  - Position Sensors                             │
│  - Industrial I/O                               │
└─────────────────────────────────────────────────┘
```

**Data Flow:**
1. Overhead GigE cameras capture vehicle images
2. Position sensors trigger inference pipeline
3. Jetson Nano processes image through optimized MobileNetV2
4. TensorRT executes quantized model for ultra-low latency
5. Classification result → Industrial Ethernet command
6. Pneumatic gates redirect vehicle (A/B sorting)

---

## Project Structure

```
Explainable-and-Edge-Al/
├── README.md                      # Project documentation (this file)
├── Part-I-Medical-XAI/
│   ├── notebooks/
│   │   ├── 01_data_exploration.ipynb
│   │   ├── 02_model_training.ipynb
│   │   ├── 03_gradcam_implementation.ipynb
│   │   ├── 04_lime_implementation.ipynb
│   │   └── 05_xai_comparison.ipynb
│   ├── src/
│   │   ├── model. py               # ResNet50 architecture
│   │   ├── gradcam.py             # Grad-CAM implementation
│   │   ├── lime_explainer.py      # LIME integration
│   │   └── utils.py               # Utility functions
│   ├── data/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── results/
│   │   ├── attention_maps/
│   │   └── performance_metrics. json
│   └── requirements. txt
├── Part-II-Edge-AI/
│   ├── notebooks/
│   │   ├── 01_baseline_model.ipynb
│   │   ├── 02_one_shot_pruning.ipynb
│   │   ├── 03_iterative_pruning.ipynb
│   │   ├── 04_quantization. ipynb
│   │   └── 05_edge_deployment.ipynb
│   ├── src/
│   │   ├── mobilenetv2.py         # Model definition
│   │   ├── pruning.py             # Pruning algorithms (L2, Taylor)
│   │   ├── quantization. py        # Model quantization
│   │   ├── jetson_inference.py    # TensorRT deployment
│   │   └── utils.py               # Utility functions
│   ├── models/
│   │   ├── baseline_model.pt      # Original model
│   │   ├── pruned_model.pt        # Optimized model
│   │   └── quantized_model.trt    # TensorRT engine
│   ├── data/
│   │   ├── train/
│   │   └── test/
│   ├── results/
│   │   ├── pruning_comparison.csv
│   │   ├── latency_benchmarks.json
│   │   └── deployment_logs/
│   └── requirements.txt
├── assets/
│   ├── images/
│   │   ├── medical_xai_pipeline.png
│   │   ├── attention_maps_examples.png
│   │   ├── edge_system_architecture.png
│   │   └── performance_comparison.png
│   └── diagrams/
├── docs/
│   ├── INSTALLATION.md            # Detailed setup instructions
│   ├── MEDICAL_XAI_GUIDE.md       # Part I documentation
│   ├── EDGE_AI_GUIDE.md           # Part II documentation
│   ├── METHODOLOGY.md             # Research methodology
│   └── RESULTS.md                 # Detailed results & analysis
└── . gitignore
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA 11.0+ (for GPU acceleration)
- pip or conda for package management
- For Edge Deployment: NVIDIA Jetson Nano with JetPack 4.6+

### Clone the Repository

```bash
git clone https://github.com/PrahasHegde/Explainable-and-Edge-Al.git
cd Explainable-and-Edge-Al
```

### Part I:  Medical XAI Setup

```bash
# Create virtual environment
python -m venv venv_medical
source venv_medical/bin/activate  # On Windows: venv_medical\Scripts\activate

# Install dependencies
cd Part-I-Medical-XAI
pip install -r requirements. txt
```

**Key Dependencies:**
- TensorFlow/PyTorch
- OpenCV
- Matplotlib/Seaborn
- scikit-learn
- jupyter

### Part II: Edge AI Setup

```bash
# Create virtual environment
python -m venv venv_edge
source venv_edge/bin/activate  # On Windows: venv_edge\Scripts\activate

# Install dependencies
cd Part-II-Edge-AI
pip install -r requirements.txt
```

**Key Dependencies:**
- PyTorch
- TensorRT (for Jetson deployment)
- ONNX
- OpenCV
- NumPy

### For Jetson Nano Deployment

```bash
# SSH into Jetson Nano
ssh your_user@jetson_nano_ip

# Install JetPack dependencies
sudo apt-get update
sudo apt-get install python3-pip

# Install TensorRT (pre-installed in JetPack)
# Install pytorch with ARM support
pip3 install torch torchvision torchaudio
```

---

## Usage

### Part I:  Explainable AI in Medical Imaging

#### Training the Model

```bash
cd Part-I-Medical-XAI
python src/train. py --epochs 50 --batch-size 32 --learning-rate 0.001
```

#### Generating Explanations

```bash
# Using Grad-CAM
python src/gradcam.py --image path/to/chest_xray.jpg --model path/to/model.pth

# Using LIME
python src/lime_explainer.py --image path/to/chest_xray.jpg --model path/to/model.pth
```

#### Interactive Notebook

```bash
jupyter notebook notebooks/05_xai_comparison.ipynb
```

This notebook demonstrates:
- Side-by-side Grad-CAM and LIME comparisons
- Classification confidence scores
- Clinician-friendly visualization

### Part II: Edge AI for Industry 4.0

#### Pruning Experiments

```bash
cd Part-II-Edge-AI

# One-Shot Pruning
python src/pruning.py --method one-shot --sparsity 0.5 --model models/baseline_model.pt

# Iterative L2 Pruning
python src/pruning.py --method iterative-l2 --sparsity 0.5 --model models/baseline_model.pt

# Iterative Taylor Pruning
python src/pruning.py --method iterative-taylor --sparsity 0.5 --model models/baseline_model.pt
```

#### Model Quantization

```bash
python src/quantization.py --model models/pruned_model.pt --output models/quantized_model. onnx
```

#### Jetson Nano Deployment

```bash
# On Jetson Nano
python src/jetson_inference.py --model models/quantized_model.trt --camera gige --output-port 50000

# Monitor performance
python src/benchmark.py --model models/quantized_model. trt --num-iterations 1000
```

#### Performance Benchmarking

```bash
jupyter notebook notebooks/05_edge_deployment.ipynb
```

This notebook includes:
- Latency measurements
- Throughput benchmarks
- Accuracy comparison across optimization methods

---

## Technologies

### Part I: Medical XAI

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Model Architecture** | ResNet50 (TensorFlow/PyTorch) | Feature extraction & classification |
| **XAI Method 1** | Grad-CAM | Gradient-based attention visualization |
| **XAI Method 2** | LIME | Local perturbation-based explanation |
| **Visualization** | Matplotlib, Plotly | Interactive result visualization |
| **Data Processing** | OpenCV, Pillow | Image preprocessing |
| **Evaluation** | scikit-learn | Metrics calculation |

### Part II: Edge AI

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Base Model** | MobileNetV2 | Lightweight architecture |
| **Pruning** | PyTorch Native | L2 & Taylor pruning implementation |
| **Quantization** | TensorRT, ONNX | Model compression |
| **Deployment** | NVIDIA Jetson Nano | Edge inference |
| **Communication** | Industrial Ethernet | Factory integration |
| **Sensing** | GigE Vision | High-speed camera interfacing |

### Cross-Project

- **Version Control:** Git & GitHub
- **Documentation:** Markdown, Jupyter Notebooks
- **Reproducibility:** Requirements. txt, conda environments
- **Deployment:** Docker (optional)

---

## Results & Performance

### Part I:  Explainable AI

**Classification Performance:**
```
Accuracy:   87%
Precision: 84%
Recall:    94% ✓ (Critical for medical imaging)
F1-Score:  89%
```

**Explainability Comparison:**

| Metric | Grad-CAM | LIME |
|--------|----------|------|
| **Resolution** | High (layer feature maps) | Medium (superpixels) |
| **Reproducibility** | ✓ Deterministic | ✗ Stochastic |
| **Interpretability** | Gradient-based (technical) | Intuitive (domain expert) |
| **Speed** | Fast (single pass) | Slower (multiple perturbations) |
| **Use Case** | Technical validation | Clinical explanation |

### Part II: Edge AI

**Model Optimization Summary:**

```
Baseline Model:          8.9 MB, 95.4% accuracy
↓
Best Approach:           Iterative Taylor Pruning
↓
Optimized Model:         6.1 MB, 95.2% accuracy
↓
Inference Latency:       ~60ms (16. 67 FPS)
↓
Jetson Nano Ready:       ✓ Deployment Successful
```

**Factory Requirements Met:**
- ✅ Model Size: <10 MB
- ✅ Inference Speed: >10 FPS (achieved 16.67 FPS)
- ✅ Accuracy:  >95% (achieved 95.2%)
- ✅ Local Processing: ✓ Cloud-independent

---

## Key Insights & Learnings

### Medical XAI
1. **Trust through transparency**:  Combined Grad-CAM + LIME provides both technical rigor and clinical interpretability
2. **High recall priority**: Pneumonia detection prioritizes sensitivity to minimize missed cases
3. **Interpretability trade-offs**: Grad-CAM is precise but technical; LIME is intuitive but slower

### Edge AI
1. **Iterative pruning outperforms one-shot**:  Taylor method preserves task-critical weights better
2. **Edge inference is practical**: 95%+ accuracy maintained with 31% model compression
3. **Hardware-aware optimization**: Jetson Nano deployment requires TensorRT optimization

---

## Contributing

This is an academic portfolio project. For suggestions or improvements: 

1. Fork the repository
2. Create a feature branch:  `git checkout -b improvement/your-improvement`
3. Make your changes and commit: `git commit -m "Add improvement description"`
4. Push to your fork: `git push origin improvement/your-improvement`
5. Open a Pull Request with a clear description

---

## License

This portfolio and all contained works are proprietary.  Please refer to the LICENSE file for usage terms and conditions.

---

## Contact

**Project Team:**
- **Prahas Hegde** - Lead Developer & Researcher
  - GitHub: [@PrahasHegde](https://github.com/PrahasHegde)
  - Email: [Add your email here]
  
- **Rohan Sanjay Patil** - Co-Researcher
- **Vidya Padmanabha** - Co-Researcher

**Portfolio Links:**
- GitHub Repository: [Explainable-and-Edge-Al](https://github.com/PrahasHegde/Explainable-and-Edge-Al)
- LinkedIn: [Add your LinkedIn profile here]

---

## Acknowledgments

This project was completed as part of Portfolio 2 for the Computer Vision course.  Special thanks to advisors, collaborators, and the open-source community for tools and resources. 

---

**Last Updated:** December 18, 2025

*Feel free to reach out with inquiries about this work, technical questions, or collaboration opportunities! *
