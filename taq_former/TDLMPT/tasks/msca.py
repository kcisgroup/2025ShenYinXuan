import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import defaultdict

class MSCA(nn.Module):
    def __init__(self, encoder, temperature=0.1, beta=1.0, walk_length=4, num_walks=10, num_pos=1, num_neg=5):
        super(MSCA, self).__init__()
        self.encoder = encoder              # BERT
        self.temperature = temperature      # τ 温度系数
        self.beta = beta                    # β 平衡系数
        self.walk_length = walk_length      # 随机游走长度
        self.num_walks = num_walks          # 每个节点随机游走次数
        self.num_pos = num_pos              # 正样本个数
        self.num_neg = num_neg              # 负样本个数
        self.graph = defaultdict(set)       # 图结构 G[node] = set(neighbors)

    def build_graph(self, edge_list):
        """
        自动构建有向图
        edge_list: List of (source, target)
        """
        self.graph = defaultdict(set)
        for src, tgt in edge_list:
            self.graph[src].add(tgt)

    def random_walk_neighbors(self, node):
        """
        多阶邻居采样：对节点执行多次随机游走，生成邻居集合 N_k(c)
        """
        neighbors = set()
        for _ in range(self.num_walks):
            current = node
            for _ in range(self.walk_length):
                if current in self.graph and self.graph[current]:
                    current = random.choice(list(self.graph[current]))
                    neighbors.add(current)
                else:
                    break
        return list(neighbors)

    def compute_pagerank(self, d=0.85, max_iter=100, tol=1e-6):
        """
        PageRank计算
        """
        nodes = list(self.graph.keys())
        N = len(nodes)
        pr = {node: 1.0 / N for node in nodes}

        for _ in range(max_iter):
            prev_pr = pr.copy()
            for node in nodes:
                neighbors = self.graph.get(node, [])
                pr[node] = (1 - d) / N + d * sum(
                    prev_pr.get(neigh, 0) / len(self.graph.get(neigh, []) or [1])
                    for neigh in neighbors
                )
            if sum(abs(pr[n] - prev_pr[n]) for n in nodes) < tol:
                break
        return pr

    def cosine_sim(self, h1, h2):
        """
        余弦相似度计算
        """
        h1 = F.normalize(h1, dim=-1)
        h2 = F.normalize(h2, dim=-1)
        return torch.matmul(h1, h2.T)  # shape: (B, B)

    def _stack_batch(self, input_dict):
        """
        将 dict[node] = {input_ids, attention_mask} 转换为 batched tensor
        """
        keys = list(input_dict.keys())
        stacked = {k: torch.stack([input_dict[n][k] for n in keys]) for k in input_dict[keys[0]]}
        return stacked

    def forward(self, input_text_dict, center_nodes):
        """
        input_text_dict: dict[node] = tokenized BERT输入，如 {'input_ids': ..., 'attention_mask': ...}
        center_nodes: List[str] 中心节点ID
        """

        # 编码所有节点文本
        inputs = self._stack_batch(input_text_dict)
        outputs = self.encoder(**inputs)  # 使用BERT提取所有节点的 [CLS]
        all_embeddings = outputs.last_hidden_state[:, 0]  # shape: (N, H)

        node_list = list(input_text_dict.keys())
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        pagerank = self.compute_pagerank()

        total_loss = 0.0
        for c in center_nodes:
            if c not in self.graph:
                continue

            # 多阶邻居采样：正样本 n
            neighbors = self.random_walk_neighbors(c)
            pos_nodes = random.sample(neighbors, min(len(neighbors), self.num_pos))

            # 负样本 m：从非邻居节点中采样
            non_neighbors = list(set(self.graph.keys()) - set(neighbors) - {c})
            neg_nodes = random.sample(non_neighbors, min(len(non_neighbors), self.num_neg))

            # 获取嵌入表示
            h_c = all_embeddings[node_to_idx[c]]
            h_n = torch.stack([all_embeddings[node_to_idx[n]] for n in pos_nodes])
            h_m = torch.stack([all_embeddings[node_to_idx[m]] for m in neg_nodes])

            # 正样本相似度
            sim_cn = self.cosine_sim(h_c.unsqueeze(0), h_n).squeeze(0)  # (num_pos,)
            sim_cm = self.cosine_sim(h_c.unsqueeze(0), h_m).squeeze(0)  # (num_neg,)

            # 构造 MSCA损失
            pos_term = torch.exp(sim_cn / self.temperature).sum()
            neg_term = torch.exp(sim_cm / self.temperature).sum()
            contrastive_loss = -torch.log(pos_term / (pos_term + neg_term + 1e-6))

            # PageRank约束项，||PR(c) - PR(m)||_2
            pr_c = pagerank.get(c, 0)
            pr_m = torch.tensor([pagerank.get(m, 0) for m in neg_nodes], device=contrastive_loss.device)
            pagerank_loss = F.mse_loss(torch.tensor([pr_c]*len(pr_m), device=contrastive_loss.device), pr_m)

            # 总损失
            loss = contrastive_loss + self.beta * pagerank_loss
            total_loss += loss

        return total_loss / len(center_nodes)

