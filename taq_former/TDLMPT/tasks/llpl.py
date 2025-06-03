import torch
import torch.nn as nn
import torch.nn.functional as F


class LLPL(nn.Module):
    def __init__(self, landmark_num, embed_dim, gamma):
        """
        参数:
            landmark_num: 地标点数量 M
            embed_dim: 节点嵌入维度 d
            gamma: 重构损失权重 γ
        """
        super(LLPL, self).__init__()
        self.gamma = gamma
        self.L = nn.Parameter(torch.randn(landmark_num, embed_dim))  # 地标嵌入矩阵 L ∈ R^{M×d}
        self.W_r = nn.Parameter(torch.randn(embed_dim, embed_dim))  # 可学习重构矩阵 W_r ∈ R^{d×d}

    def forward(self, H, A, landmark_indices):
        """
        参数:
            H: 节点表示矩阵 (|V| × d)，第 i 行为 h_i
            A: 原始邻接矩阵 (|V| × |V|)
            landmark_indices: 每个节点对应的地标索引 k(i)，长度为 |V|

        返回:
            LLPL loss
        """
        device = H.device
        V = H.shape[0]

        # ||h_i - L_{k(i)}||^2
        landmark_embed = self.L[landmark_indices]  # 取出对应地标表示 L_{k(i)}
        dist_loss = F.mse_loss(H, landmark_embed, reduction='sum')  # L2 距离损失项 ∑ ||h_i - L_{k(i)}||^2

        # γ ∑ BCE(σ(h_i^T W_r h_j), A_ij)
        # 计算重构邻接矩阵 Â_ij = σ(h_i^T W_r h_j)
        H_proj = torch.matmul(H, self.W_r)  # H W_r, shape: (|V| × d)
        A_pred = torch.sigmoid(torch.matmul(H_proj, H.T))  # σ(H W_r H^T), shape: (|V| × |V|)

        # 二元交叉熵损失
        bce_loss = F.binary_cross_entropy(A_pred, A, reduction='sum')

        # L_LLPL = dist + γ * BCE
        loss = dist_loss + self.gamma * bce_loss
        return loss
