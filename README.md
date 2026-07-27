# Gene Context Model

## Overview

Self-Supervised Multimodal Biological Representation Learning Model (PyTorch)

This repository implements a self-supervised transformer architecture for learning gene representations from multimodal genomic data, including RNA expression, somatic single-nucleotide variants (SNVs), and copy number variations (CNVs). The training pipeline is inspired by DINO-style teacher–student learning and includes EMA updates, mixed-precision training, gradient accumulation, and VICReg regularization.

## Key Features

- Self-supervised multimodal learning over RNA, SNV, and CNV modalities
- DINO-style teacher–student training with momentum EMA model updates
- Masked gene-token modeling for representation learning
- VICReg regularization to prevent collapse and promote feature diversity
- Scalable preprocessing and embedding pipelines for TCGA cancer genomics datasets
- Evaluation-ready embedding diagnostics via PCA, cosine similarity, and nearest-neighbor retrieval

## Architecture

- `src/model/gene_model.py`
  - `GeneTokenizer` constructs gene tokens by combining modality-specific embeddings
  - `GeneModel` applies a transformer encoder and projection head to produce gene representations
- `src/model/transformer.py`
  - Transformer encoder composed of multiple self-attention + feed-forward blocks
- `src/model/rna_encoder.py`, `src/model/snv_encoder.py`, `src/model/cnv_encoder.py`
  - Modality-specific embedding layers for RNA expression, SNV, and CNV data
- `src/model/fusion.py`
  - Combines embeddings with learned gene positional tokens
- `src/model/mask.py`
  - Random gene masking for student input and masked-token learning
- `src/model/projection.py`
  - Projection head and teacher centering mechanism for DINO-style loss

## Training Pipeline

- `src/model/train.py`
  - Initializes student and teacher models with shared architecture
  - Uses AdamW optimizer, mixed-precision training, and gradient accumulation
  - Computes DINO cross-entropy loss between teacher and student projections
  - Applies VICReg variance and covariance penalties on masked embeddings
  - Updates teacher network parameters with exponential moving average (EMA)

## Data Processing

- `src/model/data_retrieval.py`
  - Loads raw TCGA and Xena data files for clinical records, CNV, SNV, and RNA
  - Aligns patient IDs and gene sets across modalities
  - Converts RNA Entrez IDs to gene symbols via MyGene
- `src/model/data_loader.py`
  - Loads processed Parquet datasets from `data/processed`
  - Aligns patients and genes, removes duplicates, and filters low-variance genes
- `src/model/dataclass.py`
  - Prepares PyTorch dataset and DataLoader objects for model training
  - Builds modality tensors and masks for each patient sample

## Dependencies

Core runtime requirements are defined in `pyproject.toml` and include:

- Python >= 3.12
- PyTorch 2.5.1
- NumPy
- pandas
- scikit-learn
- scikit-survival
- matplotlib
- mygene
- pyarrow
- tqdm

## Installation

1. Create and activate a virtual environment.
2. Install dependencies from `pyproject.toml` using your preferred tool.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

> Note: The project currently uses `pyproject.toml` with a PyTorch CUDA index configuration for `torch`, `torchvision`, and `torchaudio`.

## Usage

The repository includes a notebook entrypoint at `src/model/run.ipynb`, along with training and model utilities. Example workflow:

1. Prepare the raw TCGA/Xena data under `data/`
2. Run `src/model/data_retrieval.py` to generate processed Parquet files in `data/processed`
3. Use `src/model/train.py` to initialize models and run training
4. Inspect embeddings and diagnostics using the notebook or custom evaluation scripts

## Project Structure

- `main.py` — basic entrypoint stub
- `pyproject.toml` — project metadata and dependency declarations
- `src/model/` — core modeling, preprocessing, and training code
- `data/` — raw and processed genomic data files

## Notes

- The current implementation focuses on representation learning, not direct supervised cancer outcome prediction.
- Processed dataset files are stored in `data/processed` and loaded by the training and dataset utilities.
- The codebase is built for GPU acceleration and mixed-precision training via PyTorch AMP.

## License

Update this section with your preferred license.
