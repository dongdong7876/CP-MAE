# CP-MAE: Contextual Predictability Masked Autoencoder for Robust Time Series Anomaly Detection

> 🚨 **CONFIDENTIALITY NOTICE & NOTE TO REVIEWERS**
>
> This repository contains the official source code for the manuscript: **"CP-MAE: Contextual Predictability Masked Autoencoder for Robust Time Series Anomaly Detection"**, which is currently under review at **Knowledge-Based Systems (KBS)**.
>
> **Intellectual Property & Usage Restriction:**
> This codebase is provided **solely for the purpose of peer review** by the handling editors and reviewers of the journal. To protect the intellectual property of our novel CP-MAE framework prior to formal publication, we currently restrict the usage of this code strictly to academic evaluation. 
>
> * No part of this repository (including the core dual-branch architecture, Monte Carlo inference pipeline, and data processing scripts) may be copied, distributed, modified, or utilized for any commercial or non-review academic purposes.
> * All rights are reserved by the authors. A standard open-source license (e.g., MIT License) will be applied, and the full repository will be officially released for public use only upon the acceptance and publication of the manuscript.
>
> Thank you for your time and effort in reviewing our work.

This repository contains the official PyTorch implementation of the paper **"CP-MAE: Contextual Predictability Masked Autoencoder for Robust Time Series Anomaly Detection"**.

## 🛠️ Requirements

Install the necessary dependencies using `pip install -r requirements.txt`

## 📂 Directory Structure

Plaintext

```
CP-MAE/
├── config/                  # Configuration files for different datasets (.conf)
├── data_factory/            # Data loading and preprocessing scripts
├── dataset/                 # Put your dataset files here
├── evaluation/              # Advanced TSAD metrics (VUS, Affiliation, Range-AUC)
├── model/                   # Core CP-MAE model architecture (Attn, Embed, Decoder)
├── main.py                  # Entry point for training and testing
├── solver.py                # Core training, validation, and testing loops
└── README.md                # This document
```

## 📊 Datasets

Please place the downloaded dataset files into the `dataset/` directory (e.g., `dataset/SMD/`, `dataset/SWaT/`).

The code currently supports various benchmarks including: `SMD`, `SWaT`, `WADI`, `PSM`, and `LTDB`.

## 🏃‍♂️ Usage & Invocation

Our code is designed to be easily executed and configured via the command-line interface (CLI). By default, the hyperparameters are automatically loaded from the corresponding `.conf` file in the `config/` directory based on the selected dataset.

### 1. Standard Run

To train and evaluate the model on the default dataset (SMD), simply run:

Bash

```
python main.py
```

### 2. Specify a Dataset

To run the model on a specific dataset, use the `--dataset` argument. The script will automatically load the appropriate configuration file (e.g., `config/SWaT.conf`).

Bash

```
python main.py --dataset SWaT
```

### 3. Reproducibility and Ablation Studies

You can explicitly set the random seed, execution mode (`train`, `test`), or override hyperparameters directly from the command line:

Bash

```
python main.py --dataset WADI --seed 42 --mode train --batch_size 256 --lr 0.0005
```

### 4. Tuning Masking Ratios

To experiment with different masking ratios for the strict informational bottleneck (e.g., for sensitivity analysis), use the following arguments:

Bash

```
python main.py --dataset SMD --st_mask_ratio 0.75 --tf_mask_ratio 0.90
```

## 📈 Evaluation Metrics

During the testing phase, the script will automatically calculate and output comprehensive, multi-dimensional evaluation metrics, including:

- Standard Point-wise Metrics: AUC-ROC
- Advanced Metrics: **VUS-ROC, VUS-PR** (Volume Under the Surface)
- Event-wise Metrics: **Affiliation Precision/Recall/F1**
- Range-based Metrics: **Range-AUC-ROC, Range-AUC-PR**

Results are appended and saved to `results/results_CP-MAE.csv` and logged in the `result/` folder.
