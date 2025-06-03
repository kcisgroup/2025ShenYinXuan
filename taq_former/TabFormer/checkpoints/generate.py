import torch
from transformers import BertModel

# 加载预训练的BERT模型，例如bert-base-uncased
model_name = 'bert-base-uncased'
model = BertModel.from_pretrained(model_name)

# 定义你想要保存的.pth文件名
custom_model_name = "TabFormer.pth"

# 保存模型的状态字典到.pth文件
torch.save(model.state_dict(), custom_model_name)

print(f"模型已成功保存为 {custom_model_name}")