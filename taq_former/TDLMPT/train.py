import os
import torch
import random
import warnings
import numpy as np
from config import Config
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup,logging
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torch.cuda.amp import  autocast
from tqdm import tqdm
from TDLMPT.tasks.ds_mm import DSMM
from TDLMPT.tasks.msca import MSCA
from TDLMPT.tasks.llpl import LLPL
from utils.dataset import GoodReadsDataset
from utils.collate_fn import collate_goodreads_fn

warnings.simplefilter('ignore', UserWarning)
warnings.filterwarnings('ignore')
logging.set_verbosity_error()
try:
    from apex import amp
except ImportError:
    amp = None

# 为可重复性设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if Config.device == "cuda":
        torch.cuda.manual_seed_all(seed)

def main():
    set_seed(Config.seed)

    # 加载Tokenizer、 BERT主干
    tokenizer = BertTokenizer.from_pretrained(Config.pretrained_model)
    bert = BertModel.from_pretrained(Config.pretrained_model)

    # 初始化每个子任务模块
    dsmm = DSMM(bert, temperature=1.0).to(Config.device)
    msca = MSCA(bert, temperature=Config.msca_temperature).to(Config.device)
    llpl = LLPL(landmark_num=64, embed_dim=bert.config.hidden_size, gamma=Config.llpl_gamma).to(Config.device)

    # 构建优化器
    params = list(dsmm.parameters())
    optimizer = AdamW(params, lr=Config.dsmm_lr)


    # 多卡并行
    if Config.num_gpus > 1:
        dsmm = DataParallel(dsmm)
        msca = DataParallel(msca)
        llpl = DataParallel(llpl)

    # 学习率调度器
    total_steps = 100000
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)

    # 加载 Dataset
    train_dataset = GoodReadsDataset(Config.dataset_path)
    val_dataset = GoodReadsDataset(Config.dataset_path)

    # DataLoader 加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        collate_fn=collate_goodreads_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        collate_fn=collate_goodreads_fn
    )

    # 提前停止设置
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    # 混合精度 scaler
    scaler = torch.amp.GradScaler()


    #  训练主循环
    for epoch in range(15):
        dsmm.train()
        msca.train()
        llpl.train()
        total_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for step, batch in enumerate(pbar):
            with autocast():
                loss_dsmm = dsmm(batch)
                loss_msca = msca(batch)
                loss_llpl = llpl(batch["node_embed"], batch["adj"], batch["landmark_indices"])

                loss = loss_dsmm + loss_msca + Config.llpl_gamma * loss_llpl
                loss = loss / Config.grad_accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % Config.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(train_loss=total_loss / (step + 1))

        # 验证集评估
        dsmm.eval()
        msca.eval()
        llpl.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                with autocast():
                    loss_dsmm = dsmm(batch)
                    loss_msca = msca(batch)
                    loss_llpl = llpl(batch["node_embed"], batch["adj"], batch["landmark_indices"])
                    val_loss += (loss_dsmm + loss_msca + Config.llpl_gamma * loss_llpl).item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        print(f"[Epoch {epoch+1}] Validation Loss: {avg_val_loss:.4f}")


        # 提前停止检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            save_path = os.path.join("checkpoints", "language_transformer.pth")
            os.makedirs("checkpoints", exist_ok=True)

            torch.save({
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, save_path)
            print(f"最优模型已保存至: {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("提前停止触发，训练终止")
                break

if __name__=="__main__":
    main()