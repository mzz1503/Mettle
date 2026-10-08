# Mettle

Mettle predicts likely metabolites of a query molecule from its SMILES string. This repository contains the source code and the files needed to run the released model.

The trained parameters are available. The metabolism reaction database is not. 
## Prerequisites

- Linux (tested on CentOS 7.9.2009)
- Python 3.7 (tested on Python 3.7.13)
- An NVIDIA GPU (tested on a Tesla V100S-PCIE-32GB)
- CUDA (tested on CUDA 11.8)

## Setup

From the repository root:

    conda env create -f environment.yml
    conda activate Mettle

## Trained parameters

Download `trained_models.rar` from https://doi.org/10.6084/m9.figshare.30827438 and extract it so the repository looks like this:

    trained_models
    ├── Base_Model
    ├── HybridMix_Chemical_feature_interaction
    ├── HybridMix_Contrastive_learning
    └── HybridMixMerged

Each folder contains `roc_best/` and `prc_best/`. The checkpoint file in each fold is `params.ckpt`. Prediction below uses `HybridMixMerged`.

## Predict

Run this from the `code` directory. Templates in `dataset/Templates/` are loaded automatically.

    cd code
    python predict.py --smiles "CC(C1=CN=C(NC2=CC(F)=C(O)C(F)=C2)N=C1N3C4CCCC4)(C)OC3=O" --output 14f

The ranked candidates are written to `14f.xlsx`.

## Test

The two external test sets are included as `dataset/Test_data/test_1.xlsx` and `dataset/Test_data/test_2.xlsx`. From the `code` directory, with the weights extracted as above:

    python 4_test.py --test_file ../dataset/Test_data/test_1.xlsx
    python 4_test.py --test_file ../dataset/Test_data/test_2.xlsx

`--model_name` accepts `HybridMixMerged` (default), `HybridMix_Contrastive_learning`, `HybridMix_Chemical_feature_interaction`, or `Base_Model`.

## Retraining

`1_generate_templates.py`, `2_parallel_data_process_tasks.sh`, `generate_candidates.py`, `preprocess_candidates.py`, and `3_main.py` expect a reaction table supplied by the user.

## License

The code in this repository is released under the [MIT License](LICENSE). That license does not cover the trained-weight archive on Figshare, and it does not cover records from third-party databases used to build the training set. The MDL Drug Data Report (MDDR, version 2010) is a commercial database and is not redistributed here.
