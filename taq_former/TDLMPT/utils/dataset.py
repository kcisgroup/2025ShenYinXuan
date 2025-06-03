import torch


class GoodReadsDataset(torch.utils.data.Dataset):
    def __init__(self, pt_path):
        data = torch.load(pt_path)
        # self.tokenizer = tokenizer
        # self.max_length = max_length
        self.samples = data  # 每条样本是一个dict：title, node_id, neighbour, len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        center = sample['title']
        neighbors = sample['neighbour']
        node_id = sample['node_id']
        node_len = sample['len']  # 用于 LLPL 时邻接矩阵截断

        # 构造文本输入
        full_text = center + " [SEP] " + " ".join(neighbors)
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoded["input_ids"].squeeze(0),       # [seq_len]
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "node_id": node_id,
            "neighbour": neighbors,                             # 可选：为 MSCA/LLPL 留接口
            "len": node_len,                                    # 可选：为 LLPL 留接口
        }
