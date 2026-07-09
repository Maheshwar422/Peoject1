# QT-2.3 — Quantum Vision Defect Detection System

**Quant-A-thon 2026** | Automated Manufacturing Quality Inspection

## Overview

A hybrid **quantum-classical vision system** that detects product defects in manufacturing images. The system combines classical computer vision (edge detection, texture analysis) with a variational quantum classifier built on PennyLane to deliver accurate, innovative defect detection.

## Deliverables

| Deliverable | Implementation |
|---|---|
| Image Analysis Model | Classical CV + 4-qubit VQC (PennyLane) |
| Defect Detection Report | TXT + JSON downloadable reports |
| User Interface | Streamlit web app |

## Architecture

```
Upload Image
     │
     ▼
Feature Extraction (OpenCV)
  • Edge density, texture variance
  • Patch anomaly scores, contour metrics
     │
     ├──────────────────┐
     ▼                  ▼
Classical Model    Quantum Vision Model
(weighted scoring) (4-qubit VQC, angle encoding)
     │                  │
     └────────┬─────────┘
              ▼
      Hybrid Fusion (55/45)
              │
              ▼
   Final Verdict + Report
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate sample test images (optional)
python scripts/generate_samples.py

# 3. Launch the app
streamlit run app/app.py
```

Open `http://localhost:8501` in your browser and upload a product image.

## Project Structure

```
hackathon_project/
├── app/
│   └── app.py              # Streamlit UI
├── models/
│   ├── classical_model.py  # Classical CV defect detector
│   ├── quantum_model.py    # PennyLane quantum vision classifier
│   └── hybrid_model.py     # Fusion orchestrator
├── utils/
│   ├── feature_extraction.py  # Image feature pipeline
│   └── report_generator.py    # Report builder
├── scripts/
│   └── generate_samples.py    # Sample good/defect images
├── samples/                   # Generated test images
└── requirements.txt
```

## Quantum Innovation

- **Angle encoding** of 8 image features onto 4 qubits
- **3-layer variational circuit** with ring CNOT entanglement
- **PauliZ measurement** mapped to defect probability via sigmoid
- Captures non-linear feature correlations inaccessible to linear classical models

## Evaluation Alignment

| Criterion | Weight | How We Address It |
|---|---|---|
| Detection Accuracy | 35% | Multi-metric classical + quantum fusion |
| Innovation | 20% | Full quantum vision pipeline with VQC |
| Efficiency | 20% | Fast local inference, no GPU required |
| Usability | 15% | Drag-and-drop Streamlit UI + reports |
| Presentation | 10% | Visual heatmaps, circuit diagram, metrics |

## Tech Stack

- **OpenCV** — image processing & feature extraction
- **PennyLane** — quantum machine learning
- **Streamlit** — interactive web interface
- **NumPy** — numerical computation
