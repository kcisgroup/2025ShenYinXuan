import torch


def collate_goodreads_fn(batch):
    """
    用于 DataLoader 的 collate_fn，拼接 batch。
    参数:
        batch: 一个列表，列表里每个元素是 Dataset 返回的一个样本（dict）
    返回:
        一个包含拼接后张量和列表的 dict
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])            # [B, L]
    attention_mask = torch.stack([item['attention_mask'] for item in batch])  # [B, L]
    node_ids = [item['node_id'] for item in batch]                             # list of int
    neighbours = [item['neighbour'] for item in batch]                         # list of list of str
    lens = [item['len'] for item in batch]                                     # list of int

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'node_id': node_ids,
        'neighbour': neighbours,
        'len': lens
    }
