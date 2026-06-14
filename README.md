# Heart Rate Estimation from NIR Facial Videos

## Overview

This project aims to estimate Heart Rate (HR) in Beats Per Minute (BPM) from monocular Near-Infrared (NIR) facial video sequences using deep learning techniques.

The work is based on the MR-NIRP-D dataset.

## Problem Statement

Remote Photoplethysmography (rPPG) enables contactless measurement of physiological signals by analysing subtle skin reflectance changes caused by blood flow.

The objective is to predict Heart Rate (BPM) directly from NIR facial videos.

## Dataset

MR-NIRP-D Dataset

Dataset Link:
https://computationalimaging.rice.edu/mr-nirp-dataset/

Only NIR videos are used.

## Project Pipeline

1. Load NIR video sequences
2. Extract face region
3. Preprocess frames
4. Create temporal video segments
5. Generate ground-truth BPM from physiological signals
6. Train deep learning model
7. Predict BPM
8. Evaluate using MAE, RMSE and Pearson Correlation

## Model Architecture



## Evaluation Metrics


## Results

| Metric | Value |
|----------|----------|


## Installation

```bash

```

## Run Training

```bash

```

## Run Evaluation

```bash

```

## Future Improvements

- 3D CNN architectures
- Transformer-based temporal modelling
- Motion artifact removal
- Frequency-domain supervision
- Self-supervised pretraining

## Author

Royston Rex Fernandez