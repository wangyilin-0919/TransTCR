# TransTCR: A Dual-Encoder Framework for TCR-pHLA Binding Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

## Overview

**TransTCR** is a frequency-stratified transfer learning framework for dual-encoder, chain-flexible TCR–pHLA binding prediction. It explicitly models the joint participation of TCR α and β chains and introduces a frequency-stratified negative sampling strategy to handle the highly skewed, long-tailed distribution of epitopes in real-world datasets.

## Requirements & Environment Setup

We recommend using Anaconda to create a virtual environment to run the model.

```bash
# Create and activate environment (Optional)
conda create -n transtcr python=3.9
conda activate transtcr

# Install required packages
pip install torch>=1.10.0
pip install pandas==1.5.3
pip install numpy==1.22.4
```

## Quick Start

### 1. Data Preparation

Prepare your input data as a `.csv` file (e.g., `./input/test.csv`). The file must contain the following columns:

- `TCRB`: TCR β-chain CDR3 sequence (e.g., "CASSLGQAYEQYF")
- `TCRA`: TCR α-chain CDR3 sequence (Optional, fill with "None" if only β-chain is available)
- `MHC`: HLA allele (e.g., "A*02:01")
- `Peptide`: Antigenic peptide sequence (e.g., "GILGFVFTL")

### 2. Run Prediction

Run the inference script using the following command:

```bash
python predict.py --input_csv ./input/test.csv
```

**Optional Parameters:**

- `--output_csv`: Path to save the results (default: `./output/predictions.csv`)
- `--batch_size`: Batch size for inference (default: 2048)
- `--device`: `cuda:0` or `cpu` (default: `cpu`)
- `--tcr_model`：Path to the pretrained TCR model (default: `./model/TCR_pretrain_model.pkl`)
- `--pmhc_model`:Path to the pretrained pMHC model (default: `./model/pMHC_pretrain_model.pkl`)
- `--transtcr_model`:Path to the downstream TransTCR model (default: `./model/TransTCR_model.pkl`)

*Example*

```bash
python predict.py --input_csv ./input/test.csv --output_csv ./output/results.csv --batch_size 512
```

### 3. Output Format

The script will generate a new CSV file containing the original data plus two new columns:

- **Prediction**: The predicted class (0 or 1).
- **Score**: The predicted binding probability (ranging from 0 to 1).
