import torch
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm

def ndcg_k(scores, labels, k):
    order = np.argsort(scores)[::-1]
    labels = np.take(labels, order[:k])
    dcg = np.sum((2 ** labels - 1) / np.log2(np.arange(2, 2 + len(labels))))
    ideal_dcg = np.sum((2 ** sorted(labels, reverse=True) - 1) / np.log2(np.arange(2, 2 + len(labels))))
    return dcg / ideal_dcg if ideal_dcg != 0 else 0

def evaluate_model(model, val_loader, device):
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs, labels = batch['inputs'].to(device), batch['labels'].to(device)
            scores = model(inputs)
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 假设 label 是0/1的二分类
    ndcg5 = ndcg_k(all_scores, all_labels, 5)
    ndcg10 = ndcg_k(all_scores, all_labels, 10)
    mAP = average_precision_score(all_labels, all_scores)

    return {
        "NDCG@5": ndcg5,
        "NDCG@10": ndcg10,
        "mAP": mAP
    }

