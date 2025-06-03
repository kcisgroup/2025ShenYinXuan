
import os
from torch.utils.data import DataLoader
from datasets.datasets import TTCLDataset, TCTGDataset, FTTMDataset


def get_dataloaders(task_name, batch_size=16, shuffle=True):
    base_path = '../../datasets/wikitables/'
    table_dir = os.path.join(base_path, 'tables_redi2_1')
    query_file = os.path.join(base_path, 'queries.txt')
    qrel_file = os.path.join(base_path, 'qtrels.txt')

    if task_name == 'ttcl':
        dataset = TTCLDataset(table_dir, query_file, qrel_file)
    elif task_name == 'tctg':
        dataset = TCTGDataset(table_dir, query_file, qrel_file)
    elif task_name == 'fttm':
        dataset = FTTMDataset(table_dir, query_file, qrel_file)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
