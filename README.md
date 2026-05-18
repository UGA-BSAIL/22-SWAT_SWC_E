# 22-SWAT_SWC_E

# SWAT-Informed Deep Learning Framework for Soil Water Prediction

This repository provides Python scripts for a hybrid process-based and deep learning framework that integrates Soil and Water Assessment Tool (SWAT) outputs with in-situ soil moisture sensor observations to improve soil water prediction under data-scarce conditions.

The workflow combines:

- SWAT hydrologic model outputs
- In-situ soil moisture sensor observations
- Sequence-based deep learning models
- LSTM, attention-enhanced LSTM, and Transformer architectures
- Source-site training, transfer learning, and multi-site training strategies

---

## Project Objectives

The main objectives of this workflow are to:

- Prepare site-level deep learning input files by combining SWAT outputs and sensor observations.
- Compare climate-only inputs with climate + SWAT hydrology inputs.
- Evaluate multiple deep learning architectures for soil water prediction.
- Apply transfer learning from a data-rich source site to data-limited target sites.
- Train models using data from multiple monitoring sites while avoiding cross-site sequence mixing.
- Export predictions, performance metrics, model checkpoints, and attention diagnostics.

---

## Repository Structure

```text
SWAT-DL-soil-water/
│
├── README.md
├── requirements.txt
├── config.yaml
│
├── scripts/
│   ├── 1_input_creator_e.py
│   ├── 2_Pre-train_Source_e.py
│   ├── 3_FT-Source_e.py
│   └── 4_Multi_train_e.py
│
└── outputs/
```

---

## Path Configuration

All scripts read the main working directory from `config.yaml`.

to change the path open th `config.yaml` in the main repository folder and define the project path:

```yaml
base_dir: "/C:/22-SWAT_SWC"
```

To run the workflow on another computer or HPC system, replace this path with your own project directory.

Example for Windows:

```yaml
base_dir: "E:/22-SWAT_SWC"
```

Example for Linux, macOS, or HPC:

```yaml
base_dir: "/home/user/SWAT_DL/22-SWAT_SWC"
```

The `base_dir` folder should contain the SWAT output files, sensor data, generated input files, and model output folders.

---

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:

```text
numpy
pandas
openpyxl
matplotlib
seaborn
scikit-learn
torch
pyyaml
```

---

## Input Data

The workflow uses the following main data sources:

- SWAT HRU output file: `output.hru`
- Soil moisture sensor observations: `GALR_soilmoisture2020_F.csv`

The SWAT model outputs are expected in the project directory, for example:

```text
Model22/output.hru
Model22/output.swr
```

The sensor file is expected in the main working directory:

```text
GALR_soilmoisture2020_F.csv
```

---


## Data Availability

Large input data files are archived separately on Google Drive due to GitHub file-size limits.

The dataset required to run this workflow is available at:

**https://drive.google.com/drive/folders/115dS2yAV6wopJEPYCzyvpkHjCuO7ipi4**


## Workflow Overview

The workflow should be run in the following order:

```text
1_input_creator_e.py
2_Pre-train_Source_e.py
3_FT-Source_e.py
4_Multi_train_e.py
```

Each script is described below.

---

## 1. Input Preparation

### Script

```text
1_input_creator_e.py
```

### Purpose

This script creates deep learning input files for each monitoring site by integrating SWAT outputs with in-situ soil moisture observations.

For each site, the script reads raw sensor CSV data, harmonizes column names, filters observations by sensor ID, cleans and interpolates measurements, aggregates observations to daily values, and computes representative daily variables such as mean soil moisture and mean soil temperature across depths.

The script then extracts the corresponding SWAT HRU outputs for each target HRU and merges SWAT variables with observed soil moisture. After creating site-level files, it combines all site sheets into one multi-sheet Excel workbook used by the deep learning scripts.

### Main Operations

- Reads soil moisture sensor data from `GALR_soilmoisture2020_F.csv`
- Extracts SWAT HRU variables from `output.hru`
- Cleans negative and invalid sensor values
- Interpolates missing sensor measurements
- Aggregates sensor data to daily scale
- Merges daily SWAT outputs with observed soil moisture
- Produces one Excel input file per site
- Combines site-level files into one multi-sheet workbook

### Data Quality Note

Sensor observations were manually checked. A few unrealistic values caused by sensor errors were manually corrected by replacing them with the average of previous valid values.

### Main Output

```text
output/1-Inputs_model22.xlsx
```

Each sheet corresponds to one site/HRU, for example:

```text
DL_In_site22_Hru_494_output
DL_In_site27_Hru_276_output
DL_In_site31_Hru_116_output
```

---

## 2. Source-Site Pretraining

### Script

```text
2_Pre-train_Source_e.py
```

### Purpose

This script performs source-site pretraining using five deep learning architectures. The source-site workflow trains models on a selected site/HRU and evaluates them using fixed train, validation, and test date ranges.

The script evaluates the following models:

- LSTM baseline
- LSTM with feature attention
- LSTM with temporal attention
- LSTM with combined feature and temporal attention
- Transformer

The script compares two input feature blocks:

- Climate-only inputs
- Climate + SWAT hydrology inputs

### Main Input

```text
output/1-Inputs_model22.xlsx
```

Source-site sheet:

```text
DL_In_site31_Hru_116_output
```

### Main Columns

- Observed soil moisture target: `mois_avg`
- SWAT baseline soil moisture: `VSM2mm/mm`
- Predictors from `FEATURE_BLOCKS`
- Optional seasonal variables: `doy_sin`, `doy_cos`

### Main Operations

- Builds 10-day sliding-window sequence samples
- Applies training-only Min-Max scaling to avoid data leakage
- Trains each model using deterministic random seeds
- Saves model checkpoints for each run
- Exports train, validation, and test predictions
- Computes performance metrics including `R`, `RMSE`, `MAE`, `NSE`, `PBIAS`, and `KGE`
- Compares deep learning predictions with the SWAT baseline
- Exports attention and diagnostic outputs for applicable models

### Main Outputs

Outputs are written under:

```text
1_Train_{type}_{step}_{model}_{runs}/{Block}/{Model}/
```

Example:

```text
1_Train_single_s10_model22_R20/All/LSTM_baseline/
```

Main output files include:

```text
best_{model}_RunXXX.pt
loss_{model}_run_XXX.xlsx
loss_all_runs_{model}.xlsx
metrics_pretrained
temporal_attention_all_runs_{model}.xlsx
feature_attention_all_runs_{model}.xlsx
diagnostics_summary.csv
```

The file `pre_trained_{model}.xlsx` includes:

```text
train_pretrained
val_pretrained
test_pretrained
metrics_pretrained
```

---

## 3. Fine-Tuning and Transfer Learning

### Script

```text
3_FT-Source_e.py
```

### Purpose

This script performs transfer learning for soil moisture prediction. It loads pretrained source-site checkpoints and fine-tunes them on target sites using different amounts of local data.

### Main Operations

- Loads pretrained checkpoints from the source-site training stage
- Uses the same feature definitions as source-site training
- Reuses source-site Min-Max scalers for scientific consistency
- Builds 10-day sliding-window sequences
- Fine-tunes models on multiple target sites
- Evaluates performance on a fixed test period
- Uses a paired-by-run design so only checkpoint run IDs available across all models are compared
- Saves predictions and metrics for each model, run, site, and fine-tuning budget

### Main Outputs

Outputs are written under:

```text
2_FT_{type}_{step}_{model}_{runs}/
```

Example:

```text
2_FT_single_s10_model22_R20
```

The script exports:

```text
ft_metrics.csv
preds_years{k}_last_k.csv
```

where `k` is the number of fine-tuning years.

---

## 4. Multi-Site Training

### Script

```text
4_Multi_train_e.py
```

### Purpose

This script trains the same set of deep learning models using data from multiple sites.

The main difference from the source-site script is that sequences are built separately for each site and then concatenated. This prevents cross-site window mixing and avoids artificial sequence transitions between unrelated sites.

### Main Operations

- Loads multiple site sheets from the combined input workbook
- Builds sequences separately within each site
- Concatenates site-level sequences for model training
- Fits the Min-Max scaler only on the combined training split
- Uses train and validation sequences from `TRAIN_SHEETS`
- Uses test sequences from `TEST_SHEETS`
- Computes both overall and per-site metrics
- Compares deep learning predictions with SWAT baseline simulations
- Saves model checkpoints, predictions, metrics, and diagnostic outputs

### Important Design Choices

- No cross-site sequence mixing
- Training-only scaling to avoid data leakage
- Per-site evaluation for validation and test periods
- Overall aggregated metrics across all sites

### Main Outputs

Outputs are written under:

```text
1_Train_multi_s10_model22_R20/{Block}/{Model}/
```

Example:

```text
1_Train_multi_s10_model22_R20/All/LSTM_baseline
```

Main output files include:

```text
best_{model}_RunXXX.pt
pre_trained_{model}.xlsx
per_site_metrics_val_test_RunXXX.csv
loss_all_runs_{model}.xlsx
diagnostics_summary.csv
```

---

## Model Architectures

The workflow includes the following architectures:

- `LSTM_baseline`
- `LSTM_featureAttn`
- `LSTM_temporalAttn`
- `LSTM_feat+tempAttn`
- `Transformer`

The LSTM attention variants include feature attention, temporal attention, and combined feature-temporal attention. The Transformer model uses positional encoding and a Transformer encoder for sequence regression.

---

## Feature Blocks

The scripts compare two main feature blocks.

### Climate-Only Inputs

```python
["Rain", "SOLARMJ/m2", "TMP_MXdgC"]
```

### Climate + SWAT Hydrology Inputs

```python
[
    "Rain",
    "SOLARMJ/m2",
    "TMP_MXdgC",
    "ETmm",
    "PERCmm",
    "GW_RCHGmm",
    "SURQ_GENmm",
    "LATQGENmm",
    "DAILYCN",
    "SOL_TMPdgCS",
    "LAI",
    "doy_sin",
    "doy_cos"
]
```

---

## Performance Metrics

The workflow computes the following performance metrics:

- Pearson correlation coefficient, `R`
- Root mean square error, `RMSE`
- Mean absolute error, `MAE`
- Nash-Sutcliffe efficiency, `NSE`
- Percent bias, `PBIAS`
- Kling-Gupta efficiency, `KGE`

---

## Reproducibility Features

The scripts include several reproducibility controls:

- Fixed train, validation, and test date ranges
- Deterministic random seeds
- Training-only scaling
- Structured output folders
- Saved model checkpoints
- Per-run prediction exports
- Per-site evaluation in multi-site training
- Paired checkpoint comparison for transfer learning

---

## Running the Workflow

Run the scripts in order:

```bash
python scripts/1_input_creator_e.py
python scripts/2_Pre-train_Source_e.py
python scripts/3_FT-Source_e.py
python scripts/4_Multi_train_e.py
```

On HPC systems, it is recommended to run the scripts through batch job files if the full training workflow is computationally intensive.

---

## Notes

- Update `config.yaml` before running the scripts.
- Make sure `base_dir` points to the folder containing the SWAT model outputs and sensor data.
- The processed input workbook must exist before running the training scripts.
- The transfer learning script requires pretrained checkpoints from the source-site training script.
- The multi-site training script can be run independently after the combined input workbook has been created.

