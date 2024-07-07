import torch

# GPU device setting
device = "cuda" if torch.cuda.is_available() else "cpu"

# model parameter setting
max_len = 64
d_model = 384
n_layers = 5
n_heads = 8
ffn_hidden = 1024
drop_prob = 0.1

# optimizer parameter setting
init_lr = 5e-4
factor = 0.5
adam_eps = 5e-9
patience = 1
clip = 1.0
weight_decay = 5e-4
inf = float('inf')
