# Mettle

### Prerequisites
- Linux (Tested on CentOs 7.9.2009)
- Python 3.7 (Tested on Python 3.7.13)
- NVIDIA GPU (Tested on Tesla V100S-PCIE-32GB on local workstations)
- CUDA (Tested on CUDA 11.8)

## Environment Setup

### Using Conda Environment File (Recommended)
A complete environment configuration file is provided, and installation can be completed with just two commands:

```bash
# Create environment using the configuration file
conda env create -f environment.yml

# Activate the environment
conda activate Mettle
```

## Download trained models and processed data.

The "trained models", "matched_data" and "processed_data" are ziped. 
1. Download"data_processed.rar" under 10.6084/m9.figshare.30827390
Unzip to "dataset" directory.
2. Download"trained_models.rar" under 10.6084/m9.figshare.30827438
Unzip to "trained_models" directory.

The files' structure are:
```
├─dataset
│  ├─matched_data
│  ├─processed_data
│  ├─Templates
│  └─Test_data
└─trained_models
│  ├─HybridMixMerged
│  ├─HybridMix_Contrastive_learning
│  ├─HybridMix_Chemical_feature_interaction
│  └─Base_Model
```

## Code Base Structure
The code base structure is explained below:

Process dataset:
- **1_generate_templates.py**: Extracting templates with greater universality for metabolic reactions for training.
- **2_parallel_data_process_tasks.sh**: Set generate_candidates.py and preprocess_candidate.py to be sequential multitask processes
    - **generate_candidates.py**: Match the template to get more negative samples for training. Result are saved in "/dataset/matched_data".
    - **preprocess_candidate.py**: Generate molecular fingerprints for all positive and negative samples. Result are saved in "/dataset/processed_data".

Train and test:
- **3_main.py**: Script for running the training process. Input data are saved in "/dataset/processed_data"
- **4_test.py**: Script for testing the external test sets 1 and 2.
- **predict.py**: Script for predicting the metabolites of a compound.

- **/code/train_eval.py**: Script for detailed process of model training and validation.
- **/code/model.py**: Contains PyTorch model definitions for the network.
- **/code/utils.py**: Contains definitions for data preprocessing, etc...

## Data matching and processing
Here are example commands.
###
To generate templates for training
```
python 1_generate_templates.py 
```
###
To get candidates for preparing positive samples and amount of negative samples, then get features for training. Because this task consumes a lot of CPU resources, multi-tasking processes are set up, and you can change the value of TOTAL_TASKS in parallel_data_process_tasks.sh to set a different number of processes
```
sh 2_parallel_data_process_tasks.sh
```

## Training and Evaluation
Here are example commands for training
###
Example shown below for training
```
python 3_main.py
```
Example shown below for testing
```
python 4_test.py
```
Example shown below for predict based on smiles, you need to input SMILES.
```
python predict.py --smiles "CC(C1=CN=C(NC2=CC(F)=C(O)C(F)=C2)N=C1N3C4CCCC4)(C)OC3=O" output 14f
```
