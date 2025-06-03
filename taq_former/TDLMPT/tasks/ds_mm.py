import torch
import torch.nn as nn
from typing import List, Dict

class DSMM(nn.Module):
    def __init__(self, encoder, temperature=1.0):
        super(DSMM, self).__init__()
        self.encoder = encoder  # BERT
        self.temperature = temperature

    def alpha_dc(self, dc: torch.Tensor) -> torch.Tensor:
        """
        α(dc) = 1 / (1 + log(dc)) 用于计算度数 dc 的掩码比例 α
        """
        return 1.0 / (1.0 + torch.log(dc.float() + 1e-8))

    def forward(self, batch: Dict):
        """
        执行 DS-MM 编码策略与训练
        batch: {
            'center_input': [B, L], 中心节点输入
            'neighbor_input_1': [B, L], 一阶邻居输入
            'neighbor_input_2': [B, L], 二阶邻居输入
            'center_degree': [B], 中心节点度数 dc
            'masked_indices': [B, mask_len], 要预测的掩码位置
            'masked_labels': [B, mask_len], 掩码的真实 token id
        }
        """
        T_c = self.encoder(batch['center_input'])          # T^(c)
        T_n1 = self.encoder(batch['neighbor_input_1'])     # T^(n)
        T_n2 = self.encoder(batch['neighbor_input_2'])     # T^(n2)

        # 掩码比例 α(dc)
        alpha = self.alpha_dc(batch['center_degree'])      # [B]

        # 拼接所有上下文：T^(c), T^(n), T^(n2)
        context = torch.cat([T_c, T_n1, T_n2], dim=1)       # [B, L_total, H]

        # 取出被掩码位置的 token 表达，进行预测
        # encoder 输出的是 [B, L, H]
        B, L, H = T_c.shape
        mask_len = batch['masked_indices'].shape[1]
        predictions = torch.zeros(B, mask_len, self.encoder.config.vocab_size).to(T_c.device)

        for b in range(B):
            for i, idx in enumerate(batch['masked_indices'][b]):
                if idx >= context.shape[1]: continue
                token_rep = context[b, idx]  # 被掩码 token 的上下文表示
                logits = self.encoder.cls(token_rep)  #  MLM 层
                predictions[b, i] = logits

        # Loss，预测 t_i^(c)
        loss_fct = nn.CrossEntropyLoss()
        loss = 0.0
        for b in range(B):
            loss += loss_fct(predictions[b], batch['masked_labels'][b])
        loss = loss / B

        # 乘以 α(dc) 加权
        loss = torch.mean(loss * alpha)

        return loss
