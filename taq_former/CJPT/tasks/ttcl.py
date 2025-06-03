import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig

# 加载文本编码器：BERT
class TextEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        # 使用 [CLS] token 的表示作为文本表示
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

# 加载表格编码器：TabFormer
class TableEncoder(nn.Module):
    def __init__(self, model_path= 'E:\shenyinxuan_code\TabFormer\checkpoints\TabFormer.pth', device='cpu'):
        super(TableEncoder, self).__init__()
        config = BertConfig.from_pretrained("bert-base-uncased")
        self.encoder = BertModel(config)

        # 加载预训练权重
        state_dict = torch.load(model_path, map_location=device)
        self.encoder.load_state_dict(state_dict)
        self.encoder.eval()

    def forward(self, table_data):
        with torch.no_grad():
            # 表格编码器直接输出每个表格的嵌入表示
            return self.encoder(table_data)


class TTCL(nn.Module):
    def __init__(self, temperature=0.07, num_hard_negatives=5):
        super(TTCL, self).__init__()
        self.text_encoder = TextEncoder()
        self.table_encoder = TableEncoder()
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives

    def forward(self, input_ids, attention_mask, table_data):
        # 编码
        text_emb = self.text_encoder(input_ids, attention_mask)  # t_cls: (B, H)
        table_emb = self.table_encoder(table_data)               # z_i: (B, H)

        # 归一化后可以直接用点积计算余弦相似度
        text_emb = F.normalize(text_emb, dim=1)
        table_emb = F.normalize(table_emb, dim=1)

        # 计算全部相似度矩阵 sim(z_i, t_j): (B, B)
        sim_matrix = torch.matmul(table_emb, text_emb.T)  # 每一行：一个 z_i 和所有 t_j 的相似度

        batch_size = sim_matrix.size(0)
        device = sim_matrix.device

        loss_t2z, loss_z2t = 0.0, 0.0

        for i in range(batch_size):
            # 文本 -> 表格 (text_i as anchor)
            # 正样本相似度（对应 sim(z_i, t_i)）
            pos_sim = torch.dot(text_emb[i], table_emb[i]) / self.temperature

            # 从 sim(text_i, table_j) 中排除正样本后，选最大 K 个负样本
            neg_sims = torch.cat([sim_matrix[j][i].unsqueeze(0) for j in range(batch_size) if j != i])
            neg_inds = torch.topk(neg_sims, self.num_hard_negatives)[1]

            # 获取负样本相似度值
            neg_vals = torch.cat([torch.tensor([sim_matrix[j][i] / self.temperature]).to(device)
                                  for j in range(batch_size) if j != i])[neg_inds]

            # 构造 logits：正样本在第一个位置
            logits = torch.cat([pos_sim.unsqueeze(0), neg_vals])
            labels = torch.zeros(1, dtype=torch.long).to(device)  # 正样本为第0个

            loss_t2z += F.cross_entropy(logits.unsqueeze(0), labels)

            # 表格 -> 文本 (z_i as anchor)
            pos_sim = torch.dot(table_emb[i], text_emb[i]) / self.temperature

            # 从 sim(z_i, t_j) 中排除正样本后，选最大 K 个负样本
            neg_sims = torch.cat([sim_matrix[i][j].unsqueeze(0) for j in range(batch_size) if j != i])
            neg_inds = torch.topk(neg_sims, self.num_hard_negatives)[1]

            neg_vals = torch.cat([torch.tensor([sim_matrix[i][j] / self.temperature]).to(device)
                                  for j in range(batch_size) if j != i])[neg_inds]

            logits = torch.cat([pos_sim.unsqueeze(0), neg_vals])
            labels = torch.zeros(1, dtype=torch.long).to(device)

            loss_z2t += F.cross_entropy(logits.unsqueeze(0), labels)

        # 取平均
        loss = (loss_t2z + loss_z2t) / (2 * batch_size)

        return loss, sim_matrix

