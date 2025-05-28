# Import relevant packages
# -*- coding: UTF-8 -*-
from __future__ import print_function
import torch
import sys
import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit import Chem, RDLogger
from model import *
from utils import *
from utils import _calcPR
import numpy as np
from tqdm import tqdm
from sklearn.metrics import auc, roc_auc_score,roc_curve
from rdkit import RDLogger
import argparse
RDLogger.DisableLog('rdApp.*')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_Conformer(smiles, bond_length_ranges=None):
    try:
        # 生成分子对象
        mol = Chem.MolFromInchi(smiles)
        Chem.SanitizeMol(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol)  # 力场优化

        # 获取分子的三维坐标
        conf = mol.GetConformer()
    except Exception as e:
        print(f"Error: {e}")
        return False

# 单分子预测函数，参考4_test.py

def prediction(templates_d, smiles, model_path, model_name, metric_type='both'):
    can_list = []
    can_list_fp = []
    reactants = Chem.MolFromSmiles(smiles)
    smartt = Chem.MolToSmiles(reactants)
    candidate_list, templs = reactants_to_candidate(reactants, templates_d)
    can_list.append(candidate_list)
    data = {'candidate_list': candidate_list, 'templs_id': templs}
    df = pd.DataFrame.from_dict(data)
    df['reactant'] = smiles
    le = len(candidate_list)
    if le == 0:
        return smartt, df, le
    can_list = np.array(can_list)
    x1, x2, padUpTo = preprocess_candidate(reactants, candidate_list)

    hidden_units = [1024, 512]
    hidden_units_2 = [512, 512]
    hidden_units_cat = [1024, 512, 128, 32]

    metric_folders = ["roc_best", "prc_best"] if metric_type == 'both' else [f"{metric_type}_best"]

    for metric_folder in metric_folders:
        for i in range(5):
            torch.cuda.empty_cache()
            # 根据model_name选择模型
            if model_name == 'Base_Model':
                model = Base_Model(hidden_units, hidden_units_2, hidden_units_cat)   
            if model_name == 'HybridMix_Contrastive_learning':
                model = HybridMix_Contrastive_learning(hidden_units, hidden_units_2, hidden_units_cat)    
            if model_name == 'HybridMix_Chemical_feature_interaction':
                model = HybridMix_Chemical_feature_interaction(hidden_units, hidden_units_2, hidden_units_cat)   
            if model_name == 'HybridMixMerged':
                model = HybridMixMerged_test(hidden_units, hidden_units_2, hidden_units_cat,i)   
            else:
                raise ValueError(f"Unknown model_name: {model_name}")

            model = model.to(device)
            model.eval()
            model_file_path = os.path.join(model_path, metric_folder, str(i), 'params.ckpt')
            if not os.path.exists(model_file_path):
                print(f"Warning: Model file {model_file_path} not found, skipping...")
                continue
            model.load_state_dict(torch.load(model_file_path, map_location='cpu'))
            with torch.no_grad():
                out = model(torch.Tensor(x1).to(device), torch.Tensor(x2).to(device))
            if isinstance(out, tuple):
                pred = torch.sigmoid(out[0].squeeze())
            else:
                pred = torch.sigmoid(out)
            metric_short = metric_folder.split('_')[0]
            column_name = f"{metric_short}_{i}"
            df[column_name] = pred.cpu().detach().numpy()

    roc_cols = [col for col in df.columns if col.startswith('roc_')]
    prc_cols = [col for col in df.columns if col.startswith('prc_')]
    if len(roc_cols) > 0:
        df['roc_mean'] = df[roc_cols].mean(axis=1)
    else:
        df['roc_mean'] = 0.0
    if len(prc_cols) > 0:
        df['prc_mean'] = df[prc_cols].mean(axis=1)
    else:
        df['prc_mean'] = 0.0
    if metric_type == 'both':
        if len(roc_cols) > 0 and len(prc_cols) > 0:
            df['pred_mean'] = (df['roc_mean'] + df['prc_mean']) / 2
        elif len(roc_cols) > 0:
            df['pred_mean'] = df['roc_mean']
        elif len(prc_cols) > 0:
            df['pred_mean'] = df['prc_mean']
        else:
            df['pred_mean'] = 0.0
    else:
        metric_short = metric_type
        df['pred_mean'] = df[f'{metric_short}_mean']
    df = df.sort_values(by=['pred_mean'], ascending=[False])
    df = df.reset_index(drop=True)
    return smartt, df, le

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles', type=str, required=True, help='输入分子的SMILES')
    parser.add_argument('--output', type=str, required=True, help='输出Excel文件名')
    parser.add_argument('--model_name', type=str, default='HybridMixMerged', help='模型类型')
    parser.add_argument('--model_path', type=str, default='../trained_models/HybridMixMerged', help='模型参数路径')
    parser.add_argument('--metric_type', type=str, default='both', choices=['roc', 'prc', 'both'], help='评估指标类型')
    args = parser.parse_args()

    smiles = to_can(args.smiles)
    output_filename = args.output
    model_name = args.model_name
    model_path = args.model_path
    metric_type = args.metric_type

    expert_rules_only = False
    templates_d = load_templates(expert_rules_only)
    smart, df, lencan = prediction(templates_d, smiles, model_path, model_name, metric_type)
    df.to_excel(f"{output_filename}.xlsx", index=False)
    print(f"预测完成，结果已保存到 {output_filename}.xlsx")