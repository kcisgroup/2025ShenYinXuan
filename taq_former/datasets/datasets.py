import os
import json
import random
import torch
from torch.utils.data import Dataset

class WikiTableBaseDataset(Dataset):
    def __init__(self, table_dir, query_file, qrel_file):
        self.tables = self.load_tables(table_dir)
        self.queries = self.load_queries(query_file)
        self.qrels = self.load_qrels(qrel_file)  # dict[qid] = set(table_ids)

    def load_tables(self, table_dir):
        table_dict = {}
        for fname in os.listdir(table_dir):
            if fname.endswith(".json"):
                path = os.path.join(table_dir, fname)
                with open(path, 'r') as f:
                    content = json.load(f)
                table_dict[fname] = content
        return table_dict

    def load_queries(self, query_file):
        query_dict = {}
        with open(query_file, 'r') as f:
            for line in f:
                qid, query = line.strip().split('\t')
                query_dict[qid] = query
        return query_dict

    def load_qrels(self, qrel_file):
        qrels = {}
        with open(qrel_file, 'r') as f:
            for line in f:
                qid, table_id = line.strip().split('\t')
                qrels.setdefault(qid, set()).add(table_id)
        return qrels


class TTCLDataset(WikiTableBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []
        for qid in self.qrels:
            for pos_table_id in self.qrels[qid]:
                neg_table_id = random.choice(
                    [tid for tid in self.tables if tid not in self.qrels[qid]]
                )
                self.data.append((qid, pos_table_id, neg_table_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qid, pos_id, neg_id = self.data[idx]
        query = self.queries[qid]
        pos_table = self.tables[pos_id]
        neg_table = self.tables[neg_id]
        return {
            "query": query,
            "pos_table": pos_table,
            "neg_table": neg_table
        }



class TCTGDataset(WikiTableBaseDataset):
    def __init__(self, *args, noise_type='mixed', **kwargs):
        super().__init__(*args, **kwargs)
        self.pairs = []
        self.noise_type = noise_type
        for qid in self.qrels:
            for tid in self.qrels[qid]:
                self.pairs.append((qid, tid))

    def __len__(self):
        return len(self.pairs)

    def inject_noise(self, table, noise_type='mixed'):
        table = json.loads(json.dumps(table))  # deep copy
        if noise_type == 'structure' or (noise_type == 'mixed' and random.random() < 0.5):
            # Shuffle 20% of columns
            headers = table['header']
            num_cols = len(headers)
            indices = list(range(num_cols))
            n_shuffle = max(1, int(num_cols * 0.2))
            selected = random.sample(indices, n_shuffle)
            shuffled = selected[:]
            random.shuffle(shuffled)
            for i, j in zip(selected, shuffled):
                headers[i], headers[j] = headers[j], headers[i]
        elif noise_type == 'numeric':
            for row in table['data']:
                for i, val in enumerate(row):
                    if isinstance(val, (int, float)):
                        noise = random.gauss(0, 0.1)
                        row[i] = val + noise
        return table

    def __getitem__(self, idx):
        qid, tid = self.pairs[idx]
        query = self.queries[qid]
        table = self.tables[tid]
        noisy_table = self.inject_noise(table, self.noise_type)
        return {
            "query": query,
            "table": noisy_table
        }


class FTTMDataset(WikiTableBaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pairs = []
        for qid in self.qrels:
            for tid in self.qrels[qid]:
                self.pairs.append((qid, tid))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qid, tid = self.pairs[idx]
        query = self.queries[qid]
        table = self.tables[tid]
        return {
            "query": query,
            "table": table
        }
