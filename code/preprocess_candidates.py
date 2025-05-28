from __future__ import print_function
import numpy as np
import pandas as pd

import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
from rdkit import RDLogger


import sys
sys.path.append("..")
from utils import *
from pubchemfp import GetPubChemFPs

import argparse
import json
import dill
import random
random.seed(6)
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from sklearn.feature_extraction.text import CountVectorizer

from collections import Counter
from cProfile import Profile

from config import config_dict

lg = RDLogger.logger()
RDLogger.DisableLog('rdApp.*')
lg.setLevel(4)



def get_candidates(example,N_m,v):

    reaction_candidate_smiles, reaction_true_onehot, reaction_true = [], [], []
    counter, count_not_found = 0, 0

    x1 = np.empty(shape=(0,2346),dtype=np.float16)
    x2 = np.empty(shape=(0,2346),dtype=np.float16)
    y = np.empty(shape=(1,0))
    num = np.empty(shape=(1, 0))
    candidate_smiles_all = []
    reactant_smiles_all = []


    # 按行迭代访问数据
    for i, reaction in example.iterrows():

        num_label = reaction['Label']
        candidate_smiles = reaction['candidates'][2:-2].split("', '")  # 直接在一行中分割，而不是创建一个新的副本
        reactant_smiles = reaction['reactant_smiles']
        product_smiles_true = reaction['product_smiles_true'].replace('\t', '').replace(' ', '').split(',')
        candidate_smiles_2 =[]
        candidate_SSmiles = candidate_smiles
        for c in candidate_smiles:
            try:
                candidate_smiles_2.append(Chem.MolToInchi(Chem.MolFromSmiles(c)))
            except:
                continue

        candidate_smiles = candidate_smiles_2

        reactants_check = Chem.MolFromSmiles(reactant_smiles)
        if not reactants_check:
            if v:print('Could not parse reactants:',reactant_smiles)
            continue
        product_inchi = [smiles_to_inchi(m) for m in product_smiles_true]
        bools = [k in product_inchi for k in candidate_smiles]

        if sum(bools) == 0:
            if v:
                print('True product not found / filtered out,reactants is:',reactant_smiles)
            count_not_found += 1
            continue
        reactant_smiles_list = [str(Chem.MolToSmiles(reactants_check)) for i in range(len(bools))]

        pairs = list(zip(bools, candidate_smiles))
        pairs.sort(reverse=True)  # 同时排序bools和candidate_smiles，而不是创建额外的列表
        bools = [tuple_[0] for tuple_ in pairs]

        prod_FPs = np.zeros((len(pairs), 2346), dtype=bool)
        reac_FPs = np.zeros((len(pairs), 2346), dtype=bool)
        nums = np.zeros((len(pairs), 1), dtype=int)


        for index, (_, candidate) in enumerate(pairs):

            prod = Chem.MolFromInchi(candidate)
            reac = Chem.MolFromSmiles(reactant_smiles)
            reac_ECFP= np.array(AllChem.GetMorganFingerprintAsBitVect(reac, 2, nBits = 1024), dtype = bool)
            prod_ECFP = np.array(AllChem.GetMorganFingerprintAsBitVect(prod, 2, nBits = 1024), dtype = bool)

            reac_ErG_FP = np.squeeze(np.array(AllChem.GetErGFingerprint(reac, fuzzIncrement=0.3, maxPath=21, minPath=1)))
            prod_ErG_FP = np.squeeze(np.array(AllChem.GetErGFingerprint(prod, fuzzIncrement=0.3, maxPath=21, minPath=1)))
            prod_pub_FP = np.squeeze(np.array(GetPubChemFPs(Chem.AddHs(prod))))
            reac_pub_FP = np.squeeze(np.array(GetPubChemFPs(Chem.AddHs(reac))))

            prod_FPs[index, :] =np.squeeze(np.concatenate((prod_ECFP,prod_pub_FP,prod_ErG_FP),axis=0))
            reac_FPs[index, :] = np.squeeze(np.concatenate((reac_ECFP,reac_pub_FP,reac_ErG_FP),axis=0))
            nums[index,:] = num_label

        index = 0

        le = len(candidate_smiles)

        if le > 128:
            N_c = 128
            sample_num = 128
            indexs = random.sample([i for i in range(le)], sample_num)
            x1 = np.append(x1, [prod_FPs[i] for i in indexs], axis=0)
            x2 = np.append(x2, [reac_FPs[i] for i in indexs], axis=0)
            y = np.append(y, [np.array(bools)[i] for i in indexs])
            num = np.append(num, [nums[i] for i in indexs])

        if le <= 128 :
            N_c = le
            x1 = np.append(x1, prod_FPs[:N_c], axis=0)
            x2 = np.append(x2, reac_FPs[:N_c], axis=0)
            y = np.append(y, np.array(bools)[:N_c])
            num = np.append(num, nums[:N_c])

        candidate_smiles = candidate_SSmiles[:N_c]
        reactant_smiles = reactant_smiles_list[:N_c]
        candidate_smiles_all.extend(candidate_smiles)
        reactant_smiles_all.extend(reactant_smiles)

        if len(candidate_smiles_all) != len(reactant_smiles_all):
            if v:print('error for match c&r:', reactant_smiles)
            break
        index += N_c

        if v:
            if y[index-N_c:index].shape[0]!= N_c:
                print("Mismatch in expected array size.")

        counter +=1
        if counter == N_m: break

    return x1, x2, y, num, candidate_smiles_all, reactant_smiles_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type = bool, default = False,
                        help = 'Verbose printing; defaults to False')
    parser.add_argument('--num_task', type = int, default = 50,required=True,
                        help = 'The reaction matching is divided into different tasks and carried out synchronously; defaults to 50')
    parser.add_argument('--index_task', type=int, default=0,required=True,
                        help='The reaction matching is divided into different tasks and carried out synchronously; defaults to 50')
    parser.add_argument('--N_max', type=int, default=10000,required=False,
                        help='The maximum number of processes, defined as a small number at test time')

    args = parser.parse_args()
    v = bool(args.v)
    index_task = int(args.index_task)
    total_parts = int(args.num_task)
    N_max = int(args.N_max)
    config = config_dict["data"]
    c = []
    df = pd.DataFrame()

    match_data_dir = config["matched_data_path"]
    for part_num in range(total_parts):
        match_data_path = match_data_dir + f"metabolite_candidates_for_tv_{part_num}.json"
        with open(match_data_path, encoding='utf-8') as f:
            i = 0
            for line in f.readlines():  # 依次读取每行
                i = i + 1
                line = line.strip(",\n").strip("[").strip("]")
                line = json.loads(line)
                c.append(line)
    df = pd.DataFrame(c)
    for index, doc in df.iterrows():
        df.at[index, 'candidates'] = str(doc['candidates'])
    df['reaction_id'] = df['reaction_id'].apply(str)
    examples = df.groupby(['reactant_smiles', 'num_candidates', 'candidates']).apply(
        lambda x: [','.join(x['product_smiles_true'])]).apply(lambda x: x[0]).reset_index(name='product_smiles_true')
    examples['Label'] = range(1, len(examples) + 1)
    df_split = np.array_split(examples, total_parts)

    i = 0
    for chunk in df_split:
        if i == index_task:
            x1, x2, y, n, candidate_smiles, reactant_smiles= get_candidates(chunk,N_max,v)
        i += 1

    y_flatten = y.flatten()
    n_flatten = n.flatten()
    n1 = np.sum(y_flatten == 1)
    n0 = np.sum(y_flatten == 0)

    with open(config["processed_data_path"]+'trainvalid_loader_save_{}.pkl'.format(index_task), 'wb') as f1:
        dill.dump((x1, x2, y_flatten,n_flatten,candidate_smiles, reactant_smiles), f1)