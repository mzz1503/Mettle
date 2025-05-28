
### Prerequisites
- Linux (Tested on CentOs 7.9.2009)
- NVIDIA GPU (Tested on Tesla V100S-PCIE-32GB on local workstations)
- CUDA (Tested on CUDA 11.8.)
- torch = 11.7
- openbabel
- chembl_structure_pipeline
  
## Code Base Structure
The code base structure is explained below: 

Process dataset:
- **1_generate_templates.py**: Extracting templates with greater universality for metabolic reactions for training.
- **2_parallel_data_process_tasks.sh**: Set generate_candidates.py and preprocess_candidate.py to be sequential multitask processes
    - **generate_candidates.py**: Match the template to get more negative samples for training. Result are saved in "/dataset/matched_data".
    - **preprocess_candidate.py**: Generate molecular fingerprints for all positive and negative samples. Result are saved in "/dataset/processed_data".

Train and test:
- **3_main.py**: Script for running the training process.Input data are saved in "/dataset/processed_data"
- **4_test.py.py**: Script for testing the external test sets 1 and 2.
- **5_predict.py**: Script for predicting the metabolites of a compound.

- **/code/train_eval.py**: Script for detailed process of model training and validation.
- **/code/model.py**: Contains PyTorch model definitions for the network.
- **/code/utils.py**: Contains definitions for data preprocessing, etc...

## Training and Evaluation
Here are example commands for training.
### 
Example shown below for training
```
python main.py 
```
Example shown below for testing
```
python test.py 
```

## Data matching and processing
Here are example commands.
###
To generate templates for training
```
python generate_templates.py 
```
###
To get candidates for preparing positive samples and amount of negetive samples ,then get features for training.Because this task consumes a lot of cpu resources, multi-tasking processes are set up, and you can change the value of TOTAL_TASKS in parallel_data_process_tasks.sh to set a different number of processes
```
bash parallel_data_process_tasks.sh
```
