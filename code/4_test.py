# Import relevant packages
# -*- coding: UTF-8 -*-
from __future__ import print_function
import torch
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
import os
RDLogger.DisableLog('rdApp.*')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prediction(templates_d, smiles, model_path, model_name, metric_type='both'):
    can_list = []
    can_list_fp = []
    reactants = Chem.MolFromSmiles(smiles)

    # Report New Reactant Smiles String
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
    # Convert to matrices
    x1, x2, padUpTo = preprocess_candidate(reactants, candidate_list)

    hidden_units = [1024, 512]
    hidden_units_2 = [512, 512]
    hidden_units_cat = [1024, 512, 128, 32]
    
    
    
    # 同时加载并预测ROC和PRC最优模型
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


            model = model.to(device)
            model.eval()
            # 构建模型路径
            model_file_path = os.path.join(model_path, metric_folder, str(i), 'params.ckpt')
            
            # 检查文件是否存在
            if not os.path.exists(model_file_path):
                print(f"Warning: Model file {model_file_path} not found, skipping...")
                continue
                
            # try:
            model.load_state_dict(torch.load(model_file_path, map_location='cpu'))

            with torch.no_grad():
                out = model(torch.Tensor(x1).to(device), torch.Tensor(x2).to(device))

            if isinstance(out, tuple):
                pred = torch.sigmoid(out[0].squeeze())
            else:
                pred = torch.sigmoid(out)
                
            # 将列名更改为包含指标类型和折数，例如 roc_0, prc_0
            metric_short = metric_folder.split('_')[0]  # 获取 'roc' 或 'prc'
            column_name = f"{metric_short}_{i}"
            df[column_name] = pred.cpu().detach().numpy()
            # except Exception as e:
            #     print(f"Error loading or using model {model_file_path}: {str(e)}")
            #     continue
    
    # 计算ROC和PRC模型的平均预测值
    roc_cols = [col for col in df.columns if col.startswith('roc_')]
    prc_cols = [col for col in df.columns if col.startswith('prc_')]
    
    # 检查是否有足够的预测结果
    if len(roc_cols) > 0:
        df['roc_mean'] = df[roc_cols].mean(axis=1)
    else:
        df['roc_mean'] = 0.0
        
    if len(prc_cols) > 0:
        df['prc_mean'] = df[prc_cols].mean(axis=1)
    else:
        df['prc_mean'] = 0.0
    
    # 将两种模型的预测结果平均作为最终预测
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
        # 如果只使用一种模型，则直接使用该模型的平均预测
        metric_short = metric_type
        df['pred_mean'] = df[f'{metric_short}_mean']
            
    df = df.sort_values(by=['pred_mean'], ascending=[False])
    return smartt, df, le

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='HybridMixMerged',choices=['Base_Model', 'HybridMix_Contrastive_learning', 'HybridMix_Chemical_feature_interaction', 'HybridMixMerged'],
                        help='Model architecture')
    parser.add_argument('--model_path', type=str, default='../trained_models/', 
                       help='Base directory containing model checkpoints')
    parser.add_argument('--metric_type', type=str, default='both', choices=['roc', 'prc', 'both'],
                       help='Which metric to use for loading the best model (roc, prc, or both for voting)')
    parser.add_argument('--test_file', type=str, default='../dataset/Test_data/test_1.xlsx',
                       help='Path to test data file')
    args = parser.parse_args()

    
    model_name = args.model_name
    model_path = args.model_path
    metric_type = args.metric_type
    test_file = args.test_file
    model_path = os.path.join(model_path, model_name)
    print(f"Using model: {model_name}")
    print(f"Model path: {model_path}")
    print(f"Using metric type: {metric_type}")
    print(f"Test file: {test_file}")

    expert_rules_only = False
    templates_d = load_templates(expert_rules_only)
    
    # 加载测试数据
    mt = pd.read_excel(test_file)
    mt = mt.groupby(['Substrate']).apply(lambda x: [','.join(x['Metabolite'])]).apply(lambda x: x[0]).reset_index(
    name='product')                         
    
    # 初始化统计数据 - 为每个TOP值创建一个统计字典（1到20）
    stat_tops = {}
    for k in range(1, 21):
        stat_tops[k] = {"pre": 0., "rec": 0., "one": 0., "half": 0., "all": 0.}
    
    stat_topall = {"pre": 0., "rec": 0., "one": 0., "half": 0., "all": 0.}

    # 初始化预测和标签列表
    preds_tops = {}
    labels_tops = {}
    for k in range(1, 21):
        preds_tops[k] = []
        labels_tops[k] = []
    
    preds_all = []
    labels_all = []
    df_all = pd.DataFrame()

    # 处理每个测试样本
    for indexx, doc in tqdm(mt.iterrows(), total=len(mt), desc="Processing test samples"):
        # print(doc['Substrate'])
        pro = set(str(doc['product']).split(','))
        profps = []
        # if indexx <71:
        #     continue
        for p in pro:
            
            p = to_can(p)
            prsmi = Chem.MolToSmiles(Chem.MolFromSmiles(p))
            fp = Chem.MolToInchi(Chem.MolFromSmiles(prsmi))
            profps.append(fp)

            smart, df, lencan = prediction(templates_d, to_can(doc['Substrate']), model_path, model_name, metric_type)
            
            # 如果没有候选产物，跳过
            if lencan == 0:
                print(f"No candidates for substrate {doc['Substrate']}")
                continue
        
        # 初始化每个top值的预测列表
        tops = {}
        preds = {}
        for k in range(1, 21):
            tops[k] = []
            preds[k] = []
        
        all_cand = []
        all_preds = []

        i = 0
        for count, no in df.iterrows():
            mol = Chem.MolFromInchi(no['candidate_list'])
            canonical_smi = Chem.MolToSmiles(mol)

            # 对每个TOP值添加预测
            for k in range(1, 21):
                if i < k:
                    tops[k].append(canonical_smi)
                    preds[k].append(no['pred_mean'])
                    preds_tops[k].append(no['pred_mean'])

            all_cand.append(canonical_smi)
            all_preds.append(no['pred_mean'])
            preds_all.append(no['pred_mean'])
            i = i + 1

        # 计算每个TOP值的统计指标
        for k in range(1, 21):
            stat_tops[k], labels_k = _calcPR(tops[k], preds[k], profps, stat_tops[k], k)
            labels_tops[k].extend(labels_k)
            
        stat_topall, labelall = _calcPR(all_cand, all_preds, profps, stat_topall, i)
        # print(sum(labelall))
        df['label'] = labelall
        df_all = pd.concat([df_all, df])
        labels_all.extend(labelall)

    # 保存预测结果
    results_dir = os.path.join(model_path, f"results_{metric_type}")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    df_all.to_excel(os.path.join(results_dir, f"predictions_{model_name}.xlsx"), index=False)

    # 打印评估结果
    print("\n===== MODEL EVALUATION RESULTS =====")
    print(f"Model: {model_name}, Metric: {metric_type}")
    
    # 创建结果汇总DataFrame
    summary_df = pd.DataFrame(
        columns=['TOP', 'At_least_one', 'At_least_half', 'All_metabolites', 'Precision', 'Recall', 'True_positives', 'Total_predictions']
    )
    
    # 打印每个TOP值的评估结果并保存到汇总DataFrame
    for k in range(1, 21):
        print(f"\n----- TOP {k} -----")
        print(f"At least one metabolite = {(stat_tops[k]['one'] / len(mt)):.4f}")
        print(f"At least half metabolite = {(stat_tops[k]['half'] / len(mt)):.4f}")
        print(f"All metabolites = {(stat_tops[k]['all'] / len(mt)):.4f}")
        print(f"Precision = {(stat_tops[k]['pre'] / len(mt)):.4f}")
        print(f"Recall = {(stat_tops[k]['rec'] / len(mt)):.4f}")
        
        true_pos = labels_tops[k].count(1) if 1 in labels_tops[k] else 0
        total_preds = len(labels_tops[k])
        
        print(f"Number of true positives = {true_pos}")
        print(f"Number of predictions = {total_preds}")
        
        # 添加到汇总DataFrame
        summary_df = summary_df.append({
            'TOP': k,
            'At_least_one': (stat_tops[k]['one'] / len(mt)),
            'At_least_half': (stat_tops[k]['half'] / len(mt)),
            'All_metabolites': (stat_tops[k]['all'] / len(mt)),
            'Precision': (stat_tops[k]['pre'] / len(mt)),
            'Recall': (stat_tops[k]['rec'] / len(mt)),
            'True_positives': true_pos,
            'Total_predictions': total_preds
        }, ignore_index=True)

    print("\n----- ALL -----")
    print(f"At least one metabolite = {(stat_topall['one'] / len(mt)):.4f}")
    print(f"At least half metabolite = {(stat_topall['half'] / len(mt)):.4f}")
    print(f"All metabolites = {(stat_topall['all'] / len(mt)):.4f}")
    print(f"Precision = {(stat_topall['pre'] / len(mt)):.4f}")
    print(f"Recall = {(stat_topall['rec'] / len(mt)):.4f}")
    
    true_pos_all = labels_all.count(1) if 1 in labels_all else 0
    print(f"Number of true positives = {true_pos_all}")
    print(f"Number of predictions = {len(labels_all)}")
    
    # 添加ALL到汇总DataFrame
    summary_df = summary_df.append({
        'TOP': 'ALL',
        'At_least_one': (stat_topall['one'] / len(mt)),
        'At_least_half': (stat_topall['half'] / len(mt)),
        'All_metabolites': (stat_topall['all'] / len(mt)),
        'Precision': (stat_topall['pre'] / len(mt)),
        'Recall': (stat_topall['rec'] / len(mt)),
        'True_positives': true_pos_all,
        'Total_predictions': len(labels_all)
    }, ignore_index=True)
    
    # 保存评估结果汇总到Excel
    summary_df.to_excel(os.path.join(results_dir, f"summary_{model_name}.xlsx"), index=False)
    
    # 保存详细评估结果到文本文件
    with open(os.path.join(results_dir, f"evaluation_{model_name}.txt"), 'w') as f:
        f.write(f"Model: {model_name}, Metric: {metric_type}\n\n")
        
        for k in range(1, 21):
            f.write(f"----- TOP {k} -----\n")
            f.write(f"At least one metabolite = {(stat_tops[k]['one'] / len(mt)):.4f}\n")
            f.write(f"At least half metabolite = {(stat_tops[k]['half'] / len(mt)):.4f}\n")
            f.write(f"All metabolites = {(stat_tops[k]['all'] / len(mt)):.4f}\n")
            f.write(f"Precision = {(stat_tops[k]['pre'] / len(mt)):.4f}\n")
            f.write(f"Recall = {(stat_tops[k]['rec'] / len(mt)):.4f}\n")
            
            true_pos = labels_tops[k].count(1) if 1 in labels_tops[k] else 0
            total_preds = len(labels_tops[k])
            
            f.write(f"Number of true positives = {true_pos}\n")
            f.write(f"Number of predictions = {total_preds}\n\n")
        
        f.write("----- ALL -----\n")
        f.write(f"At least one metabolite = {(stat_topall['one'] / len(mt)):.4f}\n")
        f.write(f"At least half metabolite = {(stat_topall['half'] / len(mt)):.4f}\n")
        f.write(f"All metabolites = {(stat_topall['all'] / len(mt)):.4f}\n")
        f.write(f"Precision = {(stat_topall['pre'] / len(mt)):.4f}\n")
        f.write(f"Recall = {(stat_topall['rec'] / len(mt)):.4f}\n")
        f.write(f"Number of true positives = {true_pos_all}\n")
        f.write(f"Number of predictions = {len(labels_all)}\n")