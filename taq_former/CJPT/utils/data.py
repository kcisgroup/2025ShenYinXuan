from torch.utils.data import DataLoader, Dataset
from TDLMPT.utils.dataset import GoodReadsDataset

def get_dataloader(cfg):
    dataset = GoodReadsDataset(cfg.train_data_path)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

def get_valid_loader(cfg):
    dataset = GoodReadsDataset(cfg.valid_data_path)
    return DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
