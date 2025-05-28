import argparse
import random
import dill
import os
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

from train_eval import run_classification
from model import *
from utils import *
from config import config_dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(86)
torch.cuda.manual_seed(6)
np.random.seed(6)
random.seed(6)

def reset_seeds():
    torch.manual_seed(86)
    torch.cuda.manual_seed(6)
    np.random.seed(6)
    random.seed(6)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    #模型参数
    parser.add_argument('--model_name', type=str, default='HybridMixMerged',choices=['Base_Model', 'HybridMix_Contrastive_learning', 'HybridMix_Chemical_feature_interaction', 'HybridMixMerged'],
                        help='Model architecture')
    parser.add_argument('--batch_size', type = int, default = 1024, help = 'Batch size, default 1024')
    parser.add_argument('--vt_batch_size', type = int, default = 1024, help = 'vt Batch size')

    #训练设置
    parser.add_argument('--lr', type = float, default = 0.0001,help = 'Learning rate, default 0.0001')
    parser.add_argument('--weight_decay',type = float, default = 0.00005,help = 'learning rate decay factor ,default 0.00005')
    parser.add_argument('--early_stopping',type = int, default = 50,help = 'learning rate decay factor ,default 50')
    parser.add_argument('--optimizer', type = str, default = 'Adam',help = 'Optimizer to use, default Adam')
    
    parser.add_argument('--clambda', type =float, default = 0.1, help = 'The coefficient between two loss functions,0.1')
    parser.add_argument('--TC_1', type =float, default = 100, help = 'temperature coefficient,default 100')
    parser.add_argument('--seed', type=float, default=42, help='seed')

    # 候选产物数量
    parser.add_argument('--Nc', type=int, default=128,help='Number of canidates to truncate to during training, default 1000')

    args               = parser.parse_args()
    batch_size         = int(args.batch_size)
    vt_batch_size      = int(args.vt_batch_size)
    lr                 = float(args.lr)
    weight_decay       = float(args.weight_decay)
    early_stopping     = int(args.early_stopping)
    optimizer          = args.optimizer
    
    clambda            = float(args.clambda)
    TC_1               = float(args.TC_1)
    max_N_c            = int(args.Nc)
    seed               = int(args.seed)
    hidden_units       = [1024,512]
    hidden_units_2     = [512,512]
    hidden_units_cat   = [1024,512,128,32]
    model_name         = args.model_name


    
    print(f"Using model: {model_name}")

    config = config_dict["data"]

    x1 = np.empty(shape=(0,2346),dtype=np.float16)
    x2 = np.empty(shape=(0,2346),dtype=np.float16)
    y = np.empty(shape=(0))
    num = np.empty(shape=(0))
    candidate_smiles = []
    reactant_smiles = []
    for i in range(50):
        with open(config['processed_data_path']+'trainvalid_loader_save_{}.pkl'.format(i), 'rb') as f1:
            x1i, x2i, yi,ni,candidate_smiles_i, reactant_smiles_i = dill.load(f1)
            x1 = np.append(x1, x1i, axis=0)
            x2 = np.append(x2, x2i, axis=0)
            y = np.append(y, yi)
            num = np.append(num, ni)
            candidate_smiles.extend(candidate_smiles_i)
            reactant_smiles.extend(reactant_smiles_i)

    print(f"Loaded data shapes: x1={x1.shape}, x2={x2.shape}, y={y.shape}")
    custom_dataset = CustomDataset(x1,x2,y,num,candidate_smiles,reactant_smiles,max_N_c)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    dataloaders = []

    for train_idx, val_idx in kf.split(custom_dataset):
        train_subset = torch.utils.data.Subset(custom_dataset, train_idx)
        val_subset = torch.utils.data.Subset(custom_dataset, val_idx)
        train_loader = CustomDataLoader(train_subset, batch_size=batch_size, shuffle=False, pin_memory=False,num_workers=32)
        val_loader = CustomDataLoader(val_subset, batch_size=vt_batch_size, shuffle=False, pin_memory=False,num_workers=16)
        dataloaders.append((train_loader, val_loader))
    

    print('Preparing model...')
    if model_name == 'HybridMixMerged':
        # Pretrained model
            # 创建保存目录
        save_dir_1 = '../trained_models/HybridMix_Chemical_feature_interaction/'
        if not os.path.exists(save_dir_1):
            os.makedirs(save_dir_1)
        reset_seeds()
        run_classification(
            dataloaders=dataloaders,
            batch_size=batch_size, 
            vt_batch_size=vt_batch_size, 
            lr=lr,
            weight_decay=weight_decay, 
            early_stopping=early_stopping, 
            clambda=clambda, 
            TC_1=TC_1,
            save_dir=save_dir_1,
            hidden_units=hidden_units,
            hidden_units_2=hidden_units_2,
            hidden_units_cat=hidden_units_cat,
            model_name="HybridMix_Chemical_feature_interaction"
        )

        reset_seeds()
        save_dir_2 = '../trained_models/HybridMix_Contrastive_learning/'
        if not os.path.exists(save_dir_2):
            os.makedirs(save_dir_2)
        run_classification(
            dataloaders=dataloaders,
            batch_size=batch_size, 
            vt_batch_size=vt_batch_size, 
            lr=lr,
            weight_decay=weight_decay, 
            early_stopping=early_stopping, 
            clambda=clambda, 
            TC_1=TC_1,
            save_dir=save_dir_2,
            hidden_units=hidden_units,
            hidden_units_2=hidden_units_2,
            hidden_units_cat=hidden_units_cat,
            model_name="HybridMix_Contrastive_learning"
        )
        #fusion model
        reset_seeds()
        save_dir_3 = '../trained_models/HybridMixMerged/'
        if not os.path.exists(save_dir_3):
            os.makedirs(save_dir_3)
        run_classification(
            dataloaders=dataloaders,
            batch_size=batch_size, 
            vt_batch_size=vt_batch_size, 
            lr=lr,
            weight_decay=weight_decay, 
            early_stopping=early_stopping, 
            clambda=clambda, 
            TC_1=TC_1,
            save_dir=save_dir_3,
            hidden_units=hidden_units,
            hidden_units_2=hidden_units_2,
            hidden_units_cat=hidden_units_cat,
            model_name="HybridMixMerged"
        )
    else:
        save_dir = '../trained_models/'+model_name+'/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        run_classification(
            dataloaders=dataloaders,
            batch_size=batch_size, 
            vt_batch_size=vt_batch_size, 
            lr=lr,
            weight_decay=weight_decay, 
            early_stopping=early_stopping, 
            clambda=clambda, 
            TC_1=TC_1, 
            save_dir=save_dir,
            hidden_units=hidden_units,
            hidden_units_2=hidden_units_2,
            hidden_units_cat=hidden_units_cat,
            model_name=model_name
        )
    
    print(f"Training complete!")