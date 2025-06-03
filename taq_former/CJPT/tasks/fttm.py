import torch
import torch.nn as nn
import torch.nn.functional as F


class FTTM(nn.Module):
    def __init__(self, embed_dim=768, hidden_dim=768, proj_dim=256, num_layers=4, dropout=0.1, lambda_neg=0.1):
        super(FTTM, self).__init__()
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.temperature = 0.05  # τ
        self.lambda_neg = lambda_neg  # λ

        # 编码模块：拼接表格查询Z与文本T，使用Transformer处理，生成融合表示U
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 线性投影层，用于计算细粒度匹配分数 sij
        self.w_m = nn.Linear(embed_dim * 2, proj_dim)
        self.w_s = nn.Linear(proj_dim, 1)  # w_s 为匹配分数打分向量
        self.bias = nn.Parameter(torch.zeros(1))  # b_s 偏置项

    def compute_fine_grained_score(self, u_i, o_j):
        """
        计算查询 u_i 与文本标记 o_j 的匹配分数 s_ij
        """
        combined = torch.cat([u_i, o_j], dim=-1)  # 拼接 u_i 和 o_j，Z⊕T
        h = F.relu(self.w_m(combined))            # 投影+ReLU
        s_ij = self.w_s(h).squeeze(-1) + self.bias  # 输出匹配分数 s_ij
        return s_ij

    def compute_match_score(self, u_i, o_all):
        """
        计算第i个查询与所有文本标记 o_j 的注意力聚合匹配分数 s^match_i
        """
        L = o_all.size(1)  # 文本长度
        u_i_expand = u_i.unsqueeze(1).expand(-1, L, -1)  # 扩展 u_i 维度对齐 o_j

        # 计算每个s_ij
        s_ij = self.compute_fine_grained_score(u_i_expand, o_all)  # shape: [B, L]

        # 注意力权重 a_j = Softmax(s_ij / τ)
        a_j = F.softmax(s_ij / self.temperature, dim=-1).unsqueeze(-1)  # shape: [B, L, 1]

        # s^match = ∑ a_j * s_ij
        s_match = torch.sum(a_j.squeeze(-1) * s_ij, dim=1)  # shape: [B]
        return s_match

    def forward(self, query_repr, text_repr, labels=None, neg_repr_list=None):
        """
        query_repr: 查询表征 Z, shape [B, Nq, D]
        text_repr: 文本表征 T, shape [B, L, D]
        labels: 匹配标签，shape [B, Nq]，0或1
        neg_repr_list: List[Tensor], 每个为负样本文本表示，shape [B, L, D]
        """
        # 拼接 [Z; T] => 输入Transformer
        concat_input = torch.cat([query_repr, text_repr], dim=1)  # shape: [B, Nq + L, D]
        fused_output = self.encoder(concat_input)  # 得到融合特征 U, shape: [B, Nq + L, D]

        Nq = query_repr.size(1)
        u_all = fused_output[:, :Nq, :]  # 查询表示
        o_all = fused_output[:, Nq:, :]  # 文本表示

        # 主匹配损失计算
        loss_match = 0.0
        if labels is not None:
            match_scores = []
            for i in range(Nq):
                u_i = u_all[:, i, :]  # 第i个查询
                s_i_match = self.compute_match_score(u_i, o_all)  # s^match_i
                match_scores.append(s_i_match)

            match_scores = torch.stack(match_scores, dim=1)  # [B, Nq]
            match_probs = torch.sigmoid(match_scores)        # σ(s^match)

            # 二元交叉熵损失
            loss_fn = nn.BCELoss()
            loss_match = loss_fn(match_probs, labels.float())  # 第一项损失

        # 负样本损失 L_neg
        loss_neg = 0.0
        if neg_repr_list is not None:
            # 动态选择K个负样本，计算对比损失
            for neg_repr in neg_repr_list:
                neg_output = self.encoder(torch.cat([query_repr, neg_repr], dim=1))
                o_neg = neg_output[:, Nq:, :]  # 负样本文本表示

                # contrastive_scores = []
                for i in range(Nq):
                    u_i = u_all[:, i, :]
                    pos_score = self.compute_match_score(u_i, o_all)   # 正样本
                    neg_score = self.compute_match_score(u_i, o_neg)   # 负样本

                    logits = torch.stack([pos_score, neg_score], dim=1) / self.temperature
                    labels_contrastive = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
                    loss_neg += F.cross_entropy(logits, labels_contrastive)

            loss_neg = loss_neg / len(neg_repr_list)  # 平均负样本损失

        # 总损失 = 主损失 + λ * 负样本损失
        loss = loss_match + self.lambda_neg * loss_neg

        return loss


