import time
import os
import numpy as np
import torch
from torch.optim import Adam
from model import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_classification(dataloaders, batch_size, vt_batch_size, lr,
                       weight_decay, early_stopping, clambda, TC_1,
                    save_dir, hidden_units,
                       hidden_units_2, hidden_units_cat,model_name):

    # 创建两个不同的保存目录
    save_dir_roc = os.path.join(save_dir, 'roc_best')
    save_dir_prc = os.path.join(save_dir, 'prc_best')
    
    # 确保目录存在
    if not os.path.exists(save_dir_roc):
        os.makedirs(save_dir_roc)
    if not os.path.exists(save_dir_prc):
        os.makedirs(save_dir_prc)

    best_val_roc_metrics = []
    best_val_prc_metrics = []
    # 模型初始化，根据传入的model_name选择对应的模型



        
    for fold, (train_loader, val_loader) in enumerate(dataloaders):

        # 为每个fold创建对应的保存路径
        save_dir_roc_fold = os.path.join(save_dir_roc, str(fold))
        save_dir_prc_fold = os.path.join(save_dir_prc, str(fold))
        
        if not os.path.exists(save_dir_roc_fold):
            os.makedirs(save_dir_roc_fold)
        if not os.path.exists(save_dir_prc_fold):
            os.makedirs(save_dir_prc_fold)
        
        save_path_roc = os.path.join(save_dir_roc_fold, 'params.ckpt')
        save_path_prc = os.path.join(save_dir_prc_fold, 'params.ckpt')

        print(f"ROC model save path: {save_path_roc}")
        print(f"PRC model save path: {save_path_prc}")
        # print(model)
        if model_name == 'Base_Model':
            model = Base_Model(hidden_units, hidden_units_2, hidden_units_cat)
            loss_type = "bce"
        elif model_name == "HybridMixMerged":
            model1_path = '../trained_models/HybridMix_Chemical_feature_interaction/'
            model2_path = '../trained_models/HybridMix_Contrastive_learning/'
            model = HybridMixMerged(model1_path, model2_path, hidden_units, hidden_units_2, hidden_units_cat, fold)    
            loss_type = "bce" 
        elif model_name == "HybridMix_Chemical_feature_interaction":
            model = HybridMix_Chemical_feature_interaction(hidden_units, hidden_units_2, hidden_units_cat)    
            loss_type = "bce" 
        elif model_name == "HybridMix_Contrastive_learning":
            model = HybridMix_Contrastive_learning(hidden_units, hidden_units_2, hidden_units_cat)    
            loss_type = "Contrast learning" 

        model = model.to(device)
        model = model.to(torch.float32)

        if model_name != 'HybridMixMerged':
            for m in model.modules():
                if isinstance(m, (nn.Linear)):
                    nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

        pos_weight = torch.tensor([5.0])
        pos_weight = pos_weight.to(device)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=False,
                                                threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                eps=1e-08)
        ctloss = contrastive_loss(TC_1, clambda, criterion)
        # 分别跟踪ROC和PRC的最佳值
        best_val_roc = 0
        best_val_prc = 0
        val_loss_history = []
        epoch_bvl_roc = 0
        epoch_bvl_prc = 0

        print(f"Fold {fold + 1}")
        if model_name == 'HybridMixMerged':
            epoches = 10
        else:
            epoches = 50
        for epoch in range(1, epoches+1):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_start = time.perf_counter()
            train_loss_per_smaple = train_classification(model, train_loader, loss_type, criterion, device, ctloss,
                                                        optimizer,batch_size)
            val_prc_results, val_roc_results, val_loss_per_smaple = val_classification(model, val_loader, loss_type,
                                                                                    criterion, device, ctloss,
                                                                                    vt_batch_size)

            # 无论使用哪种指标训练，都同时保存ROC和PRC最优的模型
            current_roc = np.mean(val_roc_results)
            current_prc = np.mean(val_prc_results)
            
            if current_roc > best_val_roc:
                epoch_bvl_roc = epoch
                best_val_roc = current_roc
                torch.save(model.state_dict(), save_path_roc)
                print(f"Saved new best ROC model at epoch {epoch} with ROC: {best_val_roc:.4f}")
            
            if current_prc > best_val_prc:
                epoch_bvl_prc = epoch
                best_val_prc = current_prc
                torch.save(model.state_dict(), save_path_prc)
                print(f"Saved new best PRC model at epoch {epoch} with PRC: {best_val_prc:.4f}")

            val_loss_history.append(val_loss_per_smaple)
            if early_stopping > 0 and epoch > epoches // 2 and epoch > early_stopping:
                tmp = torch.tensor(val_loss_history[-(early_stopping + 1):-1])
                if val_loss_per_smaple > tmp.mean().item():
                    break

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            t_end = time.perf_counter()

            print(
                'Epoch: {:03d}, Training Loss: {:.6f}, Val Loss: {:.6f}, Val ROC: {:.4f}, Val PRC: {:.4f}, Duration: {:.2f}'.format(
                    epoch, train_loss_per_smaple, val_loss_per_smaple,
                    np.mean(val_roc_results), np.mean(val_prc_results),
                    t_end - t_start))


        print('======================')
        print(f'Fold {fold + 1} results:')
        print(f'Best ROC: {best_val_roc:.4f} at epoch {epoch_bvl_roc}')
        print(f'Best PRC: {best_val_prc:.4f} at epoch {epoch_bvl_prc}')
        print('======================')

        best_val_roc_metrics.append(best_val_roc)
        best_val_prc_metrics.append(best_val_prc)

    print('======================')
    print(f'Average best ROC across folds: {np.mean(best_val_roc_metrics):.4f}')
    print(f'Average best PRC across folds: {np.mean(best_val_prc_metrics):.4f}')
    print('======================')


def train_classification(model, train_loader, loss_type, criterion, device, ctloss, optimizer,batch_size):
    model.train()
    losses = []
    count = 0

    for batch, data_batch in enumerate(train_loader):

        x1, x2, y, num= data_batch[0], data_batch[1], data_batch[2], \
                                   data_batch[3]

        labels = torch.tensor(y).to(device)
        out = model(torch.Tensor(x1).to(device), torch.Tensor(x2).to(device))

        loss = torch.tensor([])
        num = torch.tensor(num)
        if loss_type == 'bce':
            loss = criterion(out.squeeze(), labels.squeeze())
            loss = loss.sum()
        elif loss_type == 'Contrast learning':
            loss = ctloss(out, labels, num)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)

    return sum(losses) / (len(losses)*batch_size)


def val_classification(model, val_loader, loss_type, criterion, device, ctloss,batch_size):
    model.eval()
    preds = torch.Tensor([]).to(device)
    targets = torch.Tensor([]).to(device)
    losses = []
    count = 0
    for batch, data_batch in enumerate(val_loader):
        torch.cuda.empty_cache()

        x1, x2, y, num= data_batch[0], data_batch[1], data_batch[2], \
                                   data_batch[3]

        labels = torch.tensor(y.astype(float)).to(device)
        with torch.no_grad():
            out = model(torch.Tensor(x1).to(device), torch.Tensor(x2).to(device))


        if loss_type == 'bce':
            loss = criterion(out.squeeze(), labels.squeeze())
            loss = loss.sum()
        elif loss_type == 'Contrast learning':
            loss = ctloss(out, labels, num)
        losses.append(loss)
        if loss_type == 'bce':
            pred = torch.sigmoid(out)  ### prediction real number between (0,1)
        elif loss_type == 'Contrast learning':
            pred = torch.sigmoid(out[0])
        preds = torch.cat([preds, pred], dim=0)
        targets = torch.cat([targets.cpu(), labels.view(-1, 1).cpu()], dim=0)

    prc_results, roc_results, _ = compute_cla_metric(targets.cpu().detach().numpy(), preds.cpu().detach().numpy(), 1)
    return prc_results, roc_results, sum(losses) / (len(losses)*batch_size)
