import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def block_multiply(x1, x2, var_1,var_2,block_size):
    # 获取矩阵的维度
    batch_size, dim1, _ = x1.size()
    _, _, dim2 = x2.size()

    # 初始化结果矩阵
    result = torch.zeros(batch_size, 32, 32).to(torch.float32).to(device)

    # 分块相乘
    for i in range(0, dim1, block_size):
        end_i = min(i + block_size, dim1)
        nm = torch.matmul(x1[i:end_i,:, : ], x2[i:end_i,:, : ])
        nm = torch.matmul(var_1, nm)
        nm = torch.matmul(nm, var_2)
        result[i:end_i,:, : ] += nm

    return result

class build(torch.nn.Module):
    def __init__(self, input_dim, output_dim, dropout, activation):
        super(build, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, output_dim)

        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        self.lin1.reset_parameters()

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.lin1(x))

        return x

class DNN(nn.Module):
    def __init__(self, hidden_units, dropout=0.):
        super(DNN, self).__init__()
        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)
        self.leakyrelu = nn.LeakyReLU()
    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)
        x = self.dropout(x)
        return x


class Chemical_feature_interaction(nn.Module):

    def __init__(self,dropout):
        super(Chemical_feature_interaction, self).__init__()

        self.var_1 = torch.rand([32,1024]).to(device)
        # 计算每一列的L2范数
        # self.column_norms_1 = torch.norm(self.var_1, p=2, dim=0).to(device)
        self.column_norms_1 = torch.sqrt(torch.norm(self.var_1, p=2, dim=0)).to(device)
        # 对每一列进行除法操作
        self.var_1 = nn.Parameter(self.var_1 / self.column_norms_1,requires_grad=True).to(torch.float32)
        self.var_2 = torch.rand([1024,32]).to(device)
        # 计算每一列的L2范数
        # self.column_norms_2 = torch.norm(self.var_2, p=2, dim=0).to(device)
        self.column_norms_2 = torch.sqrt(torch.norm(self.var_2, p=2, dim=0)).to(device)
        # 对每一列进行除法操作
        self.var_2 = nn.Parameter(self.var_2 / self.column_norms_2,requires_grad=True).to(torch.float32)

        self.FP_features1 = build(1024, 512, dropout, F.relu)
        self.FP_features2 = build(512, 128, dropout, F.relu)
        self.FP_features3 = build(128, 32, dropout, F.relu)
        self.unscaled_FP_score = nn.Linear(32, 1)
        self.dropout = dropout


    def forward(self, x1,x2):
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x1 = torch.unsqueeze(x1, 2)
        x2 = torch.unsqueeze(x2, 1)
        if x1.shape[0]>=1024:
            x3 = block_multiply(x1, x2, self.var_1,self.var_2,block_size=128)
        else:
            x3 = torch.matmul(x1,x2)
            x3 = torch.matmul(self.var_1, x3)
            x3 = torch.matmul(x3, self.var_2)
        x3 = x3.permute(0, 2, 1).contiguous().view(x3.shape[0],-1)
        x3 = self.FP_features1(x3)
        x3 = self.FP_features2(x3)
        x3 = self.FP_features3(x3)
        unscaled_FP_score = self.unscaled_FP_score(x3)
        return unscaled_FP_score
    

    
class base_simple(torch.nn.Module):
    def __init__(self, hidden_units, hidden_units_2, hidden_units_cat, dropout=0.2):
        super(base_simple, self).__init__()
        self.dropout = dropout
        self.FP_features1 = build(1024, 512, dropout, F.relu)
        self.FP_features2 = build(1024, 512, dropout, F.relu)
        self.dnn_network_cat = DNN(hidden_units_cat, dropout=self.dropout)
        self.dnn_network_cat2 = DNN(hidden_units_cat, dropout=self.dropout)
        self.unscaled_FP_score = nn.Linear(hidden_units_cat[-1], 1)
        self.unscaled_FP_score2 = nn.Linear(hidden_units_cat[-1], 1)
        # self.FP_features3 = nn.Linear(512, 512)
        # self.FP_features4 = nn.Linear(512, 512)

    def forward(self, x1,x2):

        x11 = self.FP_features1(x1)
        x21 = self.FP_features2(x2)

        x = torch.cat([x11, x21], dim=1)
        x = self.dnn_network_cat(x)
        unscaled_FP_score = self.unscaled_FP_score(x)

        x_ = torch.cat([x21, x11], dim=1)
        x_ = self.dnn_network_cat2(x_)
        unscaled_FP_score_ = self.unscaled_FP_score2(x_)

        return unscaled_FP_score,unscaled_FP_score_
    

class base_SimCLR(torch.nn.Module):
    def __init__(self, hidden_units, hidden_units_2, hidden_units_cat, dropout=0.2):
        super(base_SimCLR, self).__init__()
        self.dropout = dropout
        # self.FP_features01 = build(2496,1024, dropout, F.relu)
        # self.FP_features02 = build(2496,1024, dropout, F.relu)
        self.FP_features1 = build(1024, 512, dropout, F.relu)
        self.FP_features2 = build(1024, 512, dropout, F.relu)
        self.FP_features3 = build(512, 128, dropout, F.relu)
        self.FP_features4 = build(512, 128, dropout, F.relu)
        # self.FP_features3 = nn.Linear(512, 512)
        # self.FP_features4 = nn.Linear(512, 512)

        self.dnn_network_cat = DNN(hidden_units_cat, dropout=self.dropout)
        self.dnn_network_cat2 = DNN(hidden_units_cat, dropout=self.dropout)
        self.unscaled_FP_score = nn.Linear(hidden_units_cat[-1], 1)
        self.unscaled_FP_score2 = nn.Linear(hidden_units_cat[-1], 1)

    def forward(self, x1,x2):

        x11 = self.FP_features1(x1)
        x21 = self.FP_features2(x2)

        x = torch.cat([x11, x21], dim=1)
        x = self.dnn_network_cat(x)
        unscaled_FP_score = self.unscaled_FP_score(x)

        x_ = torch.cat([x21, x11], dim=1)
        x_ = self.dnn_network_cat2(x_)
        unscaled_FP_score_ = self.unscaled_FP_score2(x_)

        x12 = self.FP_features3(x11)
        x22 = self.FP_features4(x21)

        return unscaled_FP_score,unscaled_FP_score_,x12,x22


class HybridMix_Chemical_feature_interaction(torch.nn.Module) :

    def __init__(self,hidden_units,hidden_units_2,hidden_units_cat):

        super(HybridMix_Chemical_feature_interaction,self).__init__()
        self.net_1 = base_simple(hidden_units, hidden_units_2, hidden_units_cat, dropout=0.2)
        self.net_2 = Chemical_feature_interaction(dropout=0.2)
        self.param_1 = torch.nn.Parameter(torch.ones((1,)) * 0.33,requires_grad=True)
        self.param_2 = torch.nn.Parameter(torch.ones((1,)) * 0.33,requires_grad=True)
        self.param_3 = torch.nn.Parameter(torch.ones((1,)) * 0.33,requires_grad=True)
        self.fp_cat = nn.Sequential(
            nn.Linear(2346, 1024),
            nn.GELU(),)


    def forward(self,x1, x2):
        x1 = self.fp_cat(x1)
        x2 = self.fp_cat(x2)
        xf1,xf_= self.net_1(x1,x2)
        xf2= self.net_2(x1,x2)
        psum = self.param_1 + self.param_2 + self.param_3
        xk = self.param_1 / psum * xf1 + self.param_2 / psum * xf2 + self.param_3 / psum * xf_

        return xk

    def get_feature(self, x1, x2):
        x1 = self.fp_cat(x1)
        x2 = self.fp_cat(x2)
        x11 = self.FP_features1(x1)
        x21 = self.FP_features2(x2)
        x = torch.cat([x11, x21], dim=1)
        feat = self.dnn_network_cat(x)
        return feat



class HybridMix_Contrastive_learning(torch.nn.Module) :

    def __init__(self,hidden_units,hidden_units_2,hidden_units_cat):

        super(HybridMix_Contrastive_learning,self).__init__()
        self.net_1 = base_SimCLR(hidden_units, hidden_units_2, hidden_units_cat, dropout=0.2)
        self.param_1 = torch.nn.Parameter(torch.ones((1,)) * 0.33,requires_grad=True)
        self.param_2 = torch.nn.Parameter(torch.ones((1,)) * 0.33,requires_grad=True)
        self.fp_cat = nn.Sequential(
            nn.Linear(2346, 1024),
            nn.GELU(),)


    def forward(self,x1, x2):
        x1 = self.fp_cat(x1)
        x2 = self.fp_cat(x2)

        xf1,xf_,x11,x21  = self.net_1(x1,x2)

        psum = self.param_1 + self.param_2
        xk = self.param_1 / psum * xf1 + self.param_2 / psum * xf_

        return xk,x11,x21
    

    
class Base_Model(torch.nn.Module):

    def __init__(self,hidden_units,hidden_units_2,hidden_units_cat):

        super(Base_Model,self).__init__()
        self.net_1 = base_simple(hidden_units, hidden_units_2, hidden_units_cat, dropout=0.2)
        self.param_1 = torch.nn.Parameter(torch.ones((1,)) * 0.33,requires_grad=True)
        self.param_2 = torch.nn.Parameter(torch.ones((1,)) * 0.33,requires_grad=True)
        self.fp_cat = nn.Sequential(
            nn.Linear(2346, 1024),
            nn.GELU(),)


    def forward(self,x1, x2):
        x1 = self.fp_cat(x1)
        x2 = self.fp_cat(x2)

        xf1,xf_= self.net_1(x1,x2)

        psum = self.param_1 + self.param_2
        xk = self.param_1 / psum * xf1 + self.param_2 / psum * xf_

        return xk
    
    


class HybridMixMerged(torch.nn.Module):
    def __init__(self, model1_path, model2_path, hidden_units, hidden_units_2, hidden_units_cat, fold):
        super(HybridMixMerged, self).__init__()
        
        # 检查模型文件是否存在
        
        # 初始化模型
        self.model1_roc = HybridMix_Chemical_feature_interaction(hidden_units, hidden_units_2, hidden_units_cat)
        self.model1_prc = HybridMix_Chemical_feature_interaction(hidden_units, hidden_units_2, hidden_units_cat)
        self.model2_roc = HybridMix_Contrastive_learning(hidden_units, hidden_units_2, hidden_units_cat)
        self.model2_prc = HybridMix_Contrastive_learning(hidden_units, hidden_units_2, hidden_units_cat)
        
        # 加载预训练权重
        self._load_pretrained_weights(model1_path, model2_path, fold)
        
        # 使用单个参数表示所有权重
        self.merge_weights = torch.nn.Parameter(torch.ones(4) / 4, requires_grad=True)
        
        # 冻结预训练参数
        # self._freeze_pretrained_params()
        
    
    def _load_pretrained_weights(self, model1_path, model2_path, fold):
        self.model1_roc.load_state_dict(torch.load(f"{model1_path}roc_best/{fold}/params.ckpt"))
        self.model1_prc.load_state_dict(torch.load(f"{model1_path}prc_best/{fold}/params.ckpt"))
        self.model2_roc.load_state_dict(torch.load(f"{model2_path}roc_best/{fold}/params.ckpt"))
        self.model2_prc.load_state_dict(torch.load(f"{model2_path}prc_best/{fold}/params.ckpt"))
    
    def _freeze_pretrained_params(self):
        for model in [self.model1_roc,  self.model1_prc, self.model2_roc, self.model2_prc]:
            for param in model.parameters():
                param.requires_grad = False
    
    
    def forward(self, x1, x2):
        # 获取各个模型的输出
        out1 = self.model1_roc(x1, x2)
        out1_2 = self.model1_prc(x1, x2)
        out2,  _, _ = self.model2_roc(x1, x2)
        out2_2, _, _ = self.model2_prc(x1, x2)
        
        # 使用softmax获取归一化权重
        weights = F.softmax(self.merge_weights, dim=0)
        merged_out = weights[0] * out1 + weights[1] * out1_2 + weights[2] * out2 + weights[3] * out2_2

        return merged_out
    
class HybridMixMerged_test(torch.nn.Module):
    def __init__(self, hidden_units, hidden_units_2, hidden_units_cat, fold):
        super(HybridMixMerged_test, self).__init__()
        
        # 初始化基础模型
        self.model1_roc = HybridMix_Chemical_feature_interaction(hidden_units, hidden_units_2, hidden_units_cat)
        self.model1_prc = HybridMix_Chemical_feature_interaction(hidden_units, hidden_units_2, hidden_units_cat)
        self.model2_roc = HybridMix_Contrastive_learning(hidden_units, hidden_units_2, hidden_units_cat)
        self.model2_prc = HybridMix_Contrastive_learning(hidden_units, hidden_units_2, hidden_units_cat)
        
        # 使用单个参数表示权重
        self.merge_weights = torch.nn.Parameter(torch.ones(4) / 4, requires_grad=True)
        
    def forward(self, x1, x2):
        # 获取模型输出
        out1 = self.model1_roc(x1, x2)
        out1_2 = self.model1_prc(x1, x2)
        out2, _, _ = self.model2_roc(x1, x2)
        out2_2, _, _ = self.model2_prc(x1, x2)
        
        # 确保维度正确
        out1 = out1.view(-1, 1)
        out1_2 = out1_2.view(-1, 1)
        out2 = out2.view(-1, 1)
        out2_2 = out2_2.view(-1, 1)
        
        # 使用softmax获取归一化权重
        weights = F.softmax(self.merge_weights, dim=0)
        
        # 加权融合
        merged_out = weights[0] * out1 + weights[1] * out1_2 + weights[2] * out2 + weights[3] * out2_2
        return merged_out
