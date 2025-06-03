import torch


class Config:
    # 基本设置
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    use_apex = False
    apex_opt_level = "O1"  # O1为推荐混合精度等级

    # 数据设置
    dataset_path = "E:\shenyinxuan_code\datasets\GoodReads\Goodreads.pt"
    max_seq_length = 512
    batch_size = 32
    num_workers = 4

    # 模型设置
    pretrained_model = "bert-base-uncased"
    tokenizer_sep_token = "[SEP]"

    # 优化器 & 学习率调度
    # 任务：DS-MM
    dsmm_mask_prob_min = 0.15
    dsmm_mask_prob_max = 0.30
    dsmm_lr = 2e-5
    dsmm_scheduler_decay_steps = 10000
    dsmm_scheduler_decay_rate = 0.8
    grad_accum_steps = 4

    # 任务：MSCA
    msca_lr = 5e-5
    msca_temperature = 0.07
    msca_walk_length = 5
    msca_restart_prob = 0.2
    msca_pagerank_damping = 0.85

    # 任务：LLPL
    llpl_lr = 1e-4
    llpl_spectral_hops = 3
    llpl_lambda_consistency = 0.3
    llpl_lambda_prediction = 1.0
    llpl_gamma = 1.0

    # 训练设置
    num_epochs = 10
    save_dir = "./checkpoints"
    log_interval = 100
    save_interval = 1000
    default_lr  = 2e-5

    # 分布式训练
    distributed = False  # 可扩展为多机多卡
    local_rank = -1

    # 内存与性能优化
    use_tensor_cores = True
    enable_chunking = True  # 分块计算
    max_grad_norm = 1.0
