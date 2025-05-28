from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score,accuracy_score
from sklearn import metrics
import torch
from rdkit.Chem import AllChem
import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import copy
import torch.nn.functional as F
from pubchemfp import *
from openbabel import openbabel
from chembl_structure_pipeline import standardizer
from config import config_dict
config_data = config_dict["data"]

# contrastive loss
class contrastive_loss(nn.Module):
    def __init__(self, TC_1, clambda, criterion):
        super(contrastive_loss, self).__init__()
        self.TC_1 = TC_1
        self.clambda = clambda
        self.criterion = criterion

    def get_per_loss(self, y_reactant, y_candidate, y_true, tem_1):
        similarity = F.cosine_similarity(y_reactant, y_candidate, dim=1) / self.TC_1
        ex = torch.exp(torch.diag(similarity))
        su = torch.sum(ex) - torch.sum(ex * y_true)
        loss = -torch.log(ex / (ex + su)) * y_true

        return loss.sum()

    def forward(self, y_pred, y_true, num):
        loss_BCE = self.criterion(y_pred[0].squeeze(), y_true.squeeze().float())
        loss_BCE = loss_BCE.sum()
        unique_nums = np.unique(num)  # 获取唯一的 num 值
        per_loss = 0.0
        for num_val in unique_nums:
            mask = (num == num_val)  # 创建掩码，选择具有相同 num 的数据
            per_loss += self.get_per_loss(torch.squeeze(y_pred[1])[mask], torch.squeeze(y_pred[2])[mask],
                                          torch.squeeze(y_true)[mask], self.TC_1)
            per_loss += self.get_per_loss(torch.squeeze(y_pred[2])[mask], torch.squeeze(y_pred[1])[mask],
                                          torch.squeeze(y_true)[mask], self.TC_1)
        return (loss_BCE + self.clambda * per_loss*0.5)

#data load

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, batch_size,shuffle=True,pin_memory=True, num_workers=4):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)
    def __iter__(self):
        batch_x_1 = []
        batch_x_2 = []
        batch_y = []
        batch_n = []
        batch_can = []
        batch_rea = []
        count = 0
        for data in self.dataset:
            batch_x_1.extend(data[0])
            batch_x_2.extend(data[1])
            batch_y.extend(data[2])
            batch_n.extend(data[3])
            batch_can.extend(data[4])
            batch_rea.extend(data[5])
            while len(batch_n) >= self.batch_size:
                yield (np.array(batch_x_1[:self.batch_size]),
                       np.array(batch_x_2[:self.batch_size]),
                       np.array(batch_y[:self.batch_size]),
                       np.array(batch_n[:self.batch_size]),
                       np.array(batch_can[:self.batch_size]),
                       np.array(batch_rea[:self.batch_size]),
                       )
                batch_x_1 = batch_x_1[self.batch_size:]
                batch_x_2 = batch_x_2[self.batch_size:]
                batch_y = batch_y[self.batch_size:]
                batch_n = batch_n[self.batch_size:]
                batch_can = batch_can[self.batch_size:]
                batch_rea = batch_rea[self.batch_size:]
        if len(batch_n) > 0:
            yield (np.array(batch_x_1), np.array(batch_x_2),
                   np.array(batch_y), np.array(batch_n),
                   np.array(batch_can), np.array(batch_rea),
                   )

class CustomDataset(Dataset):
    def __init__(self, x1,x2,y,num,candidate_smiles,reactant_smiles,n_max):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.n = num
        self.n_max = n_max
        self.can = np.array(candidate_smiles)
        self.rea = np.array(reactant_smiles)
        self.unique_ns = torch.unique(torch.tensor(self.n))  # 获取所有唯一的编号n
        self.grouped_data = self.group_data()

    def group_data(self):
        grouped_data = {}
        n_np = np.array(self.n)
        for current_n in self.unique_ns:
            indices = np.where(n_np == np.array(current_n))[0]
            indices = torch.tensor(indices)
            le = indices.shape[0]
            if le < self.n_max:
                sefold = self.n_max // le + 1
                indices = np.tile(indices, sefold)[:self.n_max]
            grouped_data[current_n.item()] = (self.x1[indices],self.x2[indices],
                                              self.y[indices],self.n[indices],
                                              self.can[indices],self.rea[indices])

        return grouped_data

    def __len__(self):
        return len(self.unique_ns)

    def __getitem__(self, idx):
        current_n = list(self.grouped_data.keys())[idx]
        # 从 grouped_data 中获取相同编号为 current_n 的数据
        return self.grouped_data[current_n]


# preparation for reactant and metabolites

def load_templates(expert_rules_only,training=False,templates_path_for_training=None):
    templates_list = []
    templates_smart_list = []
    reaction_type_list = []
    templates_id_list = []
    templates_expert = pd.read_excel(config_data["Templates_expert"])

    templates_expert.drop_duplicates(subset=['reaction_smarts'], keep='first', inplace=True)
    for i, template in templates_expert.iterrows():
        reaction_smarts = template['reaction_smarts']
        try:
            transform_tem = AllChem.ReactionFromSmarts(reaction_smarts)
            if transform_tem.Validate() == (0, 0):
                if transform_tem.Validate()[1] == 0:
                    templates_list.append(transform_tem)
                    templates_smart_list.append(template['reaction_smarts'])
                    reaction_type_list.append(template['Reaction_type'])
                    templates_id_list.append(template['Template_ID'])
        except Exception as e:
            print('Couldnt load: {}: {}'.format(reaction_smarts, e))

    if expert_rules_only != True:
        if training:
            templates_add = pd.read_excel(templates_path_for_training)
        else:
            templates_add = pd.read_excel(config_data["Templates_added"])
        templates_add.drop_duplicates(subset=['reaction_smarts'], keep='first', inplace=True)

        for i,template in templates_add.iterrows():
            reaction_smarts = template['reaction_smarts']
            try:
                transform_tem = AllChem.ReactionFromSmarts(reaction_smarts)
                if transform_tem.Validate() == (0, 0):
                    if transform_tem.Validate()[1] == 0:
                        templates_list.append(transform_tem)
                        templates_smart_list.append(template['reaction_smarts'])
                        reaction_type_list.append(template['Reaction_type'])
                        templates_id_list.append(template['Template_ID'])
            except Exception as e:
                print('Couldnt load: {}: {}'.format(reaction_smarts, e))

    print("there are {} templates in templates Database".format(len(templates_list)))

    keys = ['templates_list', 'templates_smart_list','reaction_type_list','Template_ID']
    values = [templates_list, templates_smart_list,reaction_type_list,templates_id_list]
    dic = dict(zip(keys, values))
    templates = pd.DataFrame(dic)

    return templates

def reactants_to_candidate(reactant,templates,training = False,v = False,target_smiles = None):
    found_true = False
    candidate_list = []  # list of tuples of (product smiles, edits required)
    candidate_list_inchi = []
    templs = []
    parent_atoms = reactant.GetNumAtoms()
    threshold = 0.15 * parent_atoms  # 15% of parent atoms
    AllChem.Compute2DCoords(reactant)

    for i,template in templates.iterrows():
        try:
            outcomes = template['templates_list'].RunReactants([reactant])
        except:
            continue
        for product in outcomes:
            frags = (Chem.GetMolFrags(product[0], asMols=True, sanitizeFrags=False))

            for p in frags:
                q = copy.copy(p)
                try:
                    q = sanitize_molecule_with_aromatic_fix(q)
                    candidate_smi = to_can(Chem.MolToSmiles(q))
                    candidate_inchi = Chem.MolToInchi(Chem.MolFromSmiles(candidate_smi))
                    mol = Chem.MolFromInchi(candidate_inchi)
                    atom_num = mol.GetNumAtoms()
                except:
                    continue
                if training:
                    if (candidate_smi) not in candidate_list:
                        if atom_num >= threshold:
                            templs.append(template['Template_ID'])
                            candidate_list.append(candidate_smi)
                            candidate_list_inchi.append(Chem.MolToInchi(Chem.MolFromSmiles(candidate_smi)))
                            if candidate_smi == target_smiles:
                                found_true = True
                                if v: print(found_true)
                else:
                    if (candidate_inchi) not in candidate_list_inchi:
                        if atom_num >= threshold:
                            templs.append(template['Template_ID'])
                            candidate_list_inchi.append(candidate_inchi)

    if training:
        return candidate_list, templs,found_true
    else:
        return candidate_list_inchi,templs

def preprocess_candidate(reactants, candidate_list):

    candidate_smiles = [Chem.MolToSmiles(Chem.MolFromInchi(a)) for a in candidate_list]
    reactant_smiles = [Chem.MolToSmiles(reactants) for i in range(len(candidate_smiles))]

    padUpTo = len(candidate_smiles)
    x1 = np.zeros((1, padUpTo, 2346), dtype=np.float32)
    x2 = np.zeros((1, padUpTo, 2346), dtype=np.float32)
    prod_FPs = np.zeros((padUpTo, 2346), dtype=bool)
    reac_FPs = np.zeros((padUpTo, 2346), dtype=bool)

    # Populate arrays
    for (c, smiles) in enumerate(candidate_smiles):
        # if c == padUpTo: break
        prod = Chem.MolFromSmiles(str(candidate_smiles[c]))
        if prod is not None:
            reac_ECFP = np.array(AllChem.GetMorganFingerprintAsBitVect(reactants, 2, nBits=1024), dtype=bool)
            prod_ECFP = np.array(AllChem.GetMorganFingerprintAsBitVect(prod, 2, nBits=1024), dtype=bool)

            reac_ErG_FP = np.squeeze(
                np.array(AllChem.GetErGFingerprint(reactants, fuzzIncrement=0.3, maxPath=21, minPath=1)))
            prod_ErG_FP = np.squeeze(
                np.array(AllChem.GetErGFingerprint(prod, fuzzIncrement=0.3, maxPath=21, minPath=1)))
            prod_pub_FP = np.squeeze(np.array(GetPubChemFPs(Chem.AddHs(prod))))
            reac_pub_FP = np.squeeze(np.array(GetPubChemFPs(Chem.AddHs(reactants))))

            prct = np.concatenate((prod_ECFP, prod_pub_FP, prod_ErG_FP), axis=0)
            prod_FPs[c, :] = np.squeeze(np.concatenate((prod_ECFP, prod_pub_FP, prod_ErG_FP), axis=0))
            reac_FPs[c, :] = np.squeeze(np.concatenate((reac_ECFP, reac_pub_FP, reac_ErG_FP), axis=0))

    return prod_FPs, reac_FPs, padUpTo

# fix and standardize smiles

def smiles_to_inchi(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return  Chem.MolToInchi(mol)

def remove_Stereo(smiles):
    mol = Chem.MolFromSmiles(smiles)
    Chem.RemoveStereochemistry(Chem.AddHs(mol))
    return Chem.MolToSmiles(mol)

def to_can(smiles):
    try:
        smiles = remove_Stereo(smiles)
    except:
        return ' '
    conv = openbabel.OBConversion()
    conv.SetInAndOutFormats("smi", "can")
    conv.SetOptions("K", conv.OUTOPTIONS)
    mol = openbabel.OBMol()
    pH = 7.4
    mol.AddHydrogens(True, True, pH)
    conv.ReadString(mol, smiles)
    smiles = conv.WriteString(mol)
    smiles = smiles.replace('\t\n', '')
    mol = Chem.MolFromSmiles(smiles)
    parent_mol, _ = standardizer.get_parent_mol(mol)
    # Chem.SanitizeMol(parent_mol)
    smiles = Chem.MolToSmiles(parent_mol, isomericSmiles=False, canonical=True)
    return smiles

def sanitize_molecule_with_aromatic_fix(mol):
    try:
        Chem.SanitizeMol(mol)
        return mol
    except ValueError as ve:
        error_msg = str(ve)
        if "non-ring atom 0 marked aromatic" in error_msg:
            # Try to fix aromatic bonds
            try:
                Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
            except ValueError as ve2:
                # If still cannot fix, change aromatic bonds to single or double bonds
                for atom in mol.GetAtoms():
                    if atom.GetIsAromatic() and not atom.IsInRing():
                        atom.SetIsAromatic(False)
        else:
            raise ve  # re-raise the original exception if it's not aromatic related
        return mol

# Calculate the accuracy of the test set
def _calcPR(res, pred,gt, stat,len_da):
    pos_count = 0  # number of positive predicted
    acn_1 = 0  # 1 if has at least one correct
    acn_half = 0  # 1 if half of the prediction is correct
    acn_all = 0  # 1 if all correct
    len_gt = float(len(gt))
    gt_cal=gt[:]
    gt_cal_copy = gt[:]
    label = []
    preds = []

    for i, (mol_pred,pred_mean) in enumerate(zip(res, pred)):
        fp_pred = Chem.MolToInchi(Chem.MolFromSmiles(mol_pred))
        smi = 0
        for fp_gt in gt_cal:
            if fp_gt == fp_pred:
                pos_count += 1
                smi = 1
        label.append(smi)
    if pos_count > 0: acn_1 = 1
    if pos_count >= len_gt // 2 + 1: acn_half = 1
    if pos_count == len_gt: acn_all = 1
    precision = pos_count / float(len_da)
    recall = pos_count / len_gt
    stat["pre"] += precision
    stat["rec"] += recall
    stat["one"] += acn_1
    stat["half"] += acn_half
    stat["all"] += acn_all
    return stat,label

def prc_auc(targets, preds):
    precision, recall, _ = precision_recall_curve(targets, preds)
    return auc(recall, precision)

def compute_cla_metric(targets, preds, num_tasks):
    prc_results = []
    roc_results = []
    acc_results = []

    for i in range(num_tasks):
        is_labeled = targets[:, i] == targets[:, i]  ## filter some samples without groundtruth label
        target = targets[is_labeled, i]
        pred = preds[is_labeled, i]

        try:
            prc = prc_auc(target, pred)
        except ValueError:
            prc = np.nan
            print("In task #", i + 1, " , there is only one class present in the set. PRC is not defined in this case.")
        try:
            roc = roc_auc_score(target, pred)
            fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=1)
        except ValueError:
            roc = np.nan
            print("In task #", i + 1, " , there is only one class present in the set. ROC is not defined in this case.")
        if not np.isnan(prc):
            prc_results.append(prc)
        else:
            print("PRC results do not consider task #", i + 1)
        if not np.isnan(roc):
            roc_results.append(roc)
        else:
            print("ROC results do not consider task #", i + 1)
        y_preds_ = [i + 0.5 for i in preds]
        y_preds_ = list(map(int, y_preds_))
        acc = accuracy_score(target, y_preds_)
        acc_results.append(acc)

    return prc_results, roc_results, acc_results