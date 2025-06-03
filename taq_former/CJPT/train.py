import torch
import torch.optim as optim
from transformers import BertModel
from CJPT.config import Config
from CJPT.tasks.ttcl import TTCL
from CJPT.tasks.tctg import QFormer, TableEncoder, TCTG
from CJPT.tasks.fttm import FTTM
from CJPT.utils.data import get_dataloader, get_valid_loader
from CJPT.utils.my_logging import init_logger
from CJPT.utils.eval import evaluate_model
from CJPT.utils.grid_search import grid_search_weights

def main():
    logger = init_logger()
    cfg = Config()

    table_encoder = TableEncoder(model_path=cfg.table_encoder_path).to(cfg.device)
    text_encoder = BertModel.from_pretrained(cfg.text_encoder_model).to(cfg.device)
    qformer = QFormer(cfg.query_dim, cfg.num_queries, cfg.hidden_dim).to(cfg.device)
    ttcl = TTCL(temperature=cfg.ttcl_temperature).to(cfg.device)
    tctg = TCTG(cfg.bart_model, cfg.hidden_dim, cfg.tctg_sigma).to(cfg.device)
    fttm = FTTM(embed_dim=cfg.hidden_dim, proj_dim=cfg.fttm_proj_dim, lambda_neg=cfg.fttm_lambda_neg).to(cfg.device)

    # 优化器设置
    params_main = list(table_encoder.parameters()) + list(text_encoder.parameters()) + list(qformer.parameters())
    params_aux = list(tctg.cross_attention.parameters())  # 可独立优化

    optimizer_main = optim.Adam(params_main, lr=cfg.learning_rate)
    optimizer_aux = optim.Adam(params_aux, lr=cfg.lr_cross_attention)

    # 数据加载
    train_loader = get_dataloader(cfg)
    valid_loader = get_valid_loader(cfg)

    # 权重搜索
    best_weights = grid_search_weights(cfg, ttcl, tctg, fttm, table_encoder, text_encoder, qformer, train_loader, valid_loader)

    # 提前停止配置
    best_metric = 0.0
    best_epoch = 0
    patience_counter = 0

    # 主训练循环
    for epoch in range(cfg.num_epochs):
        table_encoder.train()
        text_encoder.train()
        qformer.train()

        for batch in train_loader:
            table = batch["table"].to(cfg.device)
            text = {k: v.to(cfg.device) for k, v in batch["text"].items()}
            labels = batch["labels"].to(cfg.device)
            generation_targets = batch["generation_targets"]

            table_repr = table_encoder(table)
            text_repr = text_encoder(**text).last_hidden_state
            query_repr = qformer(text_repr)

            loss_ttcl = ttcl(table_repr, text_repr)
            loss_tctg = tctg(query_repr, generation_targets)
            loss_fttm = fttm(table_repr, text_repr, labels)

            loss = (
                best_weights['ttcl'] * loss_ttcl +
                best_weights['tctg'] * loss_tctg +
                best_weights['fttm'] * loss_fttm
            )

            optimizer_main.zero_grad()
            loss.backward()
            optimizer_main.step()

            # Cross-attention 额外优化
            optimizer_aux.zero_grad()
            loss_tctg.backward(retain_graph=True)
            optimizer_aux.step()

        # 验证阶段
        val_metric = evaluate_model(table_encoder, text_encoder, valid_loader)
        logger.info(f"[Epoch {epoch + 1}] val_ndcg@20: {val_metric['ndcg@20']:.4f}")

        # 提前停止判断
        if val_metric['ndcg@20'] > best_metric:
            best_metric = val_metric['ndcg@20']
            best_epoch = epoch
            patience_counter = 0

            # 保存模型
            torch.save({
                'taqformer': qformer.state_dict()
            }, cfg.save_path)
            logger.info(f"New best model saved at epoch {epoch+1} with ndcg@20: {best_metric:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Patience counter: {patience_counter}/{cfg.early_stopping_patience}")
            if patience_counter >= cfg.early_stopping_patience:
                logger.info(f"Early stopping triggered. Best epoch: {best_epoch + 1}, best ndcg@20: {best_metric:.4f}")
                break

if __name__=="__main__":
    main()