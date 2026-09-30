# Hemodynamic Correction and Behavioral Prediction LSTMs

Code accompanying:
> Inferring norepinephrine dynamics from partial observations reveals the temporal structure of elevations during arousal — Erin Neyhart, Brandon R. Munn, Na Zhou, Peilin Yang, Jiesi Feng, Yulong Li, James M. Shine, Jacob Reimer

## Overview

This repository contains two LSTM models developed to infer norepinephrine
(NE) dynamics from recordings with varying levels of available information:

1. **Hemodynamic correction LSTM** (`01_hemodynamic_correction_lstm.ipynb`):
   Predicts the hemodynamic artifact in the NE fluorescence signal using the NE
   trace itself plus optional spatial brightness bins and behavioral variables.
   Four predictor sets are compared (rNE only; rNE + brightness bins; rNE +
   behavior; rNE + bins + behavior).

2. **Behavioral prediction LSTM** (`02_behavioral_prediction_lstm.ipynb`):
   Predicts NE fluorescence from behavioral variables alone (pupil dilation and
   locomotion).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── utils.py                                  # Shared signal processing functions
├── 01_hemodynamic_correction_lstm.ipynb      # Hemodynamic correction LSTM (4 predictor sets)
├── 02_behavioral_prediction_lstm.ipynb       # Behavioral prediction LSTM (raw + corrected NE)
├── data/
│   ├── README.md                             # Dataset description, DOI, and column definitions
│   ├── hemodynamic_prediction/
│   │   ├── rNE-mut_prediction_data/          # Main recording CSVs (hemodynamic correction model)
│   │   └── rNE-BrightnessBins/              # Brightness-bin CSVs (hemodynamic correction model)
│   ├── rNE0.5/                               # rNE0.5 recording CSVs (behavioral prediction model)
│   └── GrabNE2h/                             # GrabNE2h recording CSVs (behavioral prediction model)
└── output/
    ├── mut_model_0_weights.pth               # Hemodynamic correction model weights (rNE only)
    ├── mut_model_1_weights.pth               # Hemodynamic correction model weights (rNE + bins)
    ├── mut_model_2_weights.pth               # Hemodynamic correction model weights (rNE + behavior)
    ├── mut_model_3_weights.pth               # Hemodynamic correction model weights (rNE + bins + behavior)
    └── NE_bx_model_raw_weights.pth           # Behavioral prediction model weights (raw NE)
```

## Setup

### Requirements

Python 3.9+ is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

> **Note:** Model training was performed using PyTorch 2.8.0 with CUDA 12.8.
> A CPU-only installation (`pip install torch==2.8.0`) is sufficient for
> running inference with the provided pre-trained weights.

### Data

Download the dataset from https://dandiarchive.org/dandiset/001865 and place it under `data/`
following the directory structure described in `data/README.md`.

## Running the notebooks

Run notebooks in any order.

## Trained model weights

Pre-trained weights are provided in `output/` so you can run inference without
retraining. Each notebook includes a dedicated "load saved weights" cell at the
end. See the notebooks for details on input formatting.
