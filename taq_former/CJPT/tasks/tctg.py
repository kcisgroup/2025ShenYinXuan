import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForConditionalGeneration, BertConfig, BertModel


# Q-Former：查询引导的特征抽取模块
class QFormer(nn.Module):
    def __init__(self, query_dim, num_queries, hidden_dim):
        super(QFormer, self).__init__()
        self.query_embed = nn.Parameter(torch.randn(num_queries, query_dim))  # learnable query embeddings
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    def forward(self, table_emb):
        # 输入 table_emb: 表格的单元格表示 (B, M, D)
        B, M, D = table_emb.shape
        queries = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B, num_queries, D)
        q_input = torch.cat([queries, table_emb], dim=1)  # 拼接 Query 和 Cell (B, num_queries + M, D)
        out = self.encoder(q_input.transpose(0, 1)).transpose(0, 1)  # 编码后返回 (B, num_queries + M, D)
        return out[:, :queries.shape[1], :]  # 仅返回 query 的输出部分 Q_t，作为语义特征


# 表格编码器
class TableEncoder(nn.Module):
    def __init__(self, model_path, device='cpu'):
        super(TableEncoder, self).__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        self.encoder = BertModel(config)
        state_dict = torch.load(model_path, map_location=device)
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()

    def forward(self, table_input):
        # 返回表格嵌入 Z (B, M, D)
        with torch.no_grad():
            return self.encoder(**table_input).last_hidden_state


# Cross-Attention：查询语义与单元格信息的对齐（交互特征建模）
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(CrossAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, key, value):
        attn_out, _ = self.attn(query, key, value)  # Q_t 与 Z 的交互：CrossAtt(Q_t, Z)
        x = self.norm1(query + attn_out)  # 残差连接 + LayerNorm
        ffn_out = self.ffn(x)  # 前馈网络建模非线性
        return self.norm2(x + ffn_out)  # 再次残差 + LN，输出融合后特征 H_t


class TCTG(nn.Module):
    def __init__(self, model_path, hidden_dim=768, sigma=0.1, num_heads=8, num_queries=8):
        super(TCTG, self).__init__()
        self.hidden_dim = hidden_dim
        self.sigma = sigma  # 控制噪声强度 ε（对应结构 & 数值扰动）

        # 初始化子模块
        self.table_encoder = TableEncoder(model_path)
        self.qformer = QFormer(query_dim=hidden_dim, num_queries=num_queries, hidden_dim=hidden_dim)
        self.cross_attention = CrossAttention(dim=hidden_dim, num_heads=num_heads)

        # 门控融合机制
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 文本生成器
        self.bart = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    def forward(self, table_input, decoder_input_ids, labels=None):
        # 表格嵌入 Z
        z_value = self.table_encoder(table_input)  # Z ∈ R^{B×M×D}

        # 注入结构噪声（结构扰动，扰乱表格结构顺序）
        if self.training:
            noise_idx = torch.randperm(z_value.size(1))  # (M,)
            z_value = z_value[:, noise_idx, :]  # 扰乱行顺序

        # 注入数值噪声（高斯扰动）
        if self.training:
            noise = torch.randn_like(z_value) * self.sigma
            z_value = z_value + noise  # Z' = Z + ε

        # 提取 Q-Former 特征 Q_t
        q_output = self.qformer(z_value)  # Q_t ∈ R^{B×Q×D}

        # Q_t 与 Z 交互 → 生成融合特征 H_t
        h_output = self.cross_attention(q_output, z_value, z_value)  # H_t ∈ R^{B×Q×D}

        # 门控融合（Q_t 与 H_t）
        fusion = torch.cat([q_output, h_output], dim=-1)  # [B, Q, 2D]
        gate_weight = self.gate(fusion)  # G ∈ [0, 1]
        final_feature = gate_weight * q_output + (1 - gate_weight) * h_output  # F_t

        # 用 F_t 作为 encoder 输出，输入 BART 解码器生成文本
        encoder_outputs = torch.mean(final_feature, dim=1, keepdim=True)
        encoder_outputs = encoder_outputs.repeat(1, decoder_input_ids.shape[1], 1)

        outputs = self.bart(
            inputs_embeds=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True
        )

        return outputs


