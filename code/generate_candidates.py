# Import relevant packages
from __future__ import print_function

import argparse
import pandas as pd
import os                          # for saving
import sys
sys.path.append("..")

import rdkit.Chem as Chem
from rdkit import RDLogger

from tqdm import tqdm
import json

from utils import *
from config import config_dict

RDLogger.DisableLog('rdApp.*')
USE_STEREOCHEMISTRY=False


def main(reactions,templates,split_list,index_task,save_dir,v,N_max,training):

    save_path = save_dir+f"metabolite_candidates_for_tv_{str(index_task)}.json"
    if os.path.exists(save_path):
        os.remove(save_path)
    with open(save_path, 'a', encoding='utf-8') as fw:
        fw.write('[')

    reaction_number = len(reactions.index)
    # print(split_list)
    for i, reaction in reactions.iterrows():
        if i in split_list:
            if v: print('## RXN {}'.format(i))
            rxn_smiles = reaction['Reaction']
            reac_all_smiles = [x for x in rxn_smiles.split('>>')[0].split('.')]
            target_all_smiles = [x for x in rxn_smiles.split('>>')[1].split('.')]
            reac_smiles = max(reac_all_smiles, key=len)
            target_smiles = max(target_all_smiles, key=len)
            reactants = Chem.MolFromSmiles(reac_smiles)
            n_reactant_atoms = len(reactants.GetAtoms())
            if n_reactant_atoms > 100:
                if v: print('Skipping huge molecule! N_reactant_atoms = {}'.format(n_reactant_atoms))
                continue
            [a.SetProp('molAtomMapNumber', str(i + 1)) for (i, a) in enumerate(reactants.GetAtoms())]

            candidate_list, templs, found_true = reactants_to_candidate(reactants, templates,training,v,target_smiles)

            if v: print(len(candidate_list))
            doc = {
                '_id': i,
                'reaction_id': reaction['Reaction_label'],
                'reactant_smiles': Chem.MolToSmiles(reactants, isomericSmiles=USE_STEREOCHEMISTRY),
                'product_smiles_true': target_smiles,
                'found': found_true,
                'num_candidates': len(candidate_list),
                'candidates': candidate_list,
                'num_templ': len(templs),
                'templs': templs,
            }
            if i == N_max:
                break

            with open(save_path, 'a', encoding='utf-8') as fw:
                json.dump(doc, fw, ensure_ascii=False)
                # if i < reaction_number:
                fw.write(",")
                fw.write("\n")


def split_list_n_list(origin_list, n):
    if len(origin_list) % n == 0:
        cnt = len(origin_list) // n
    else:
        cnt = len(origin_list) // n + 1

    for i in range(0, n):
        yield origin_list[i * cnt:(i + 1) * cnt]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', type = bool, default = False,
                        help = 'Verbose printing; defaults to False')
    parser.add_argument('--num_task', type = int, default = 50,required=True,
                        help = 'The reaction matching is divided into different tasks and carried out synchronously; defaults to 50')
    parser.add_argument('--index_task', type=int, default=0,required=True,
                        help='The reaction matching is divided into different tasks and carried out synchronously; defaults to 50')
    parser.add_argument('--N_max', type=int, default=100000,required=False,
                        help='The maximum number of processes, defined as a small number at test time')
    args = parser.parse_args()
    v = bool(args.v)
    index_task = int(args.index_task)
    num_task = int(args.num_task)
    N_max = int(args.N_max)

    config = config_dict["data"]

    save_dir = config["matched_data_path"]

    reactions = pd.read_excel(config["database_path"])
    expert_rules_only = False
    training = True
    templates_path_for_training=config["Templates_training"]
    templates = load_templates(expert_rules_only, training, templates_path_for_training)

    tasks_index = [i for i in range(len(reactions))]
    split_list = split_list_n_list(tasks_index, num_task)
    i = 0
    for split_list_sub in split_list:
        if i == index_task:
            main(reactions,templates,split_list_sub,index_task,save_dir,v,N_max,training)
        i = i + 1
