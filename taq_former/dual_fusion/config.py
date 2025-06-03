import torch

class Config:

    # ========== QFormer 参数 ==========
    query_dim = 768
    num_queries = 10
    hidden_dim = 768

    # ========== 训练通用参数 ==========
    batch_size = 16
    learning_rate = 5e-5
    lr_cross_attention = 1e-4         # 单独优化 cross_attention 层
    num_epochs = 15                   # 总轮数

    # ========== TTCL 超参数 ==========
    ttcl_temperature = 0.07

    # ========== TCTG 超参数 ==========
    tctg_sigma = 0.1

    # ========== FTTM 超参数 ==========
    fttm_proj_dim = 256
    fttm_lambda_neg = 0.1

    # ========== 网格搜索设置 ==========
    lambda_grid = {
        'ttcl': [0.1, 0.3, 0.5, 1.0],
        'tctg': [0.1, 0.3, 0.5, 1.0],
        'fttm': [0.1, 0.3, 0.5, 1.0]
    }

    # ========== 设备 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ========== 提前停止 ==========
    early_stopping_patience = 3  # 超过 3

    # ========== 数据 ==========
    train_data_path= '..\datasets\GoodReads\Goodreads.pt'
    valid_data_path= '..\datasets\GoodReads\Goodreads.pt'

    # ========== 模型路径 ==========
    table_encoder_path = '..\TabFormer\checkpoints\TabFormer.pth'  # 表格编码器
    text_encoder_model = 'bert-base-uncased'  # 文本编码器
    bart_model = '..\TabFormer\checkpoints\TabFormer.pth'
