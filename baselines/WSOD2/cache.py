import torch

# 选择要清空显存的 GPU
device = torch.device("cuda:2")

# 清空显存
torch.cuda.empty_cache()