# -*- coding: UTF-8 -*-
import pandas as pd
import argparse
from predict import prediction, load_templates, to_can
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_file', type=str, required=True, help='包含SMILES的txt文件，每行一个')
    parser.add_argument('--output', type=str, required=True, help='输出Excel文件名')
    parser.add_argument('--model_name', type=str, default='HybridMixMerged', help='模型类型')
    parser.add_argument('--model_path', type=str, default='../trained_models/HybridMixMerged', help='模型参数路径')
    parser.add_argument('--metric_type', type=str, default='both', choices=['roc', 'prc', 'both'], help='评估指标类型')
    args = parser.parse_args()

    # 读取SMILES列表
    with open(args.smiles_file, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    expert_rules_only = False
    templates_d = load_templates(expert_rules_only)

    all_results = []
    for smiles in smiles_list:
        print(smiles)
        can_smiles = to_can(smiles)
        smart, df, lencan = prediction(templates_d, can_smiles, args.model_path, args.model_name, args.metric_type)
        df['input_smiles'] = smiles  # 保留原始输入
        all_results.append(df)

    # 合并所有结果
    result_df = pd.concat(all_results, ignore_index=True)
    result_df.to_excel(f"{args.output}.xlsx", index=False)
    print(f"所有化合物预测完成，结果已保存到 {args.output}.xlsx")
