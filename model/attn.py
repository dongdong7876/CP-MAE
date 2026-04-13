import torch
import torch.nn as nn
import math
from math import sqrt

class AttentionLayer(nn.Module):
    def __init__(self, d_model):
        super(AttentionLayer, self).__init__()

        self.norm = nn.LayerNorm(d_model)

        self.query_projection = nn.Linear(d_model, d_model)  # Linear(in_features=128, out_features=128, bias=True)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        queries = self.query_projection(x)
        keys = self.key_projection(x).transpose(1, 2)
        values = self.value_projection(x)

        with torch.autocast(device_type=x.device.type, enabled=False):
            q_fp32 = queries.float()
            k_fp32 = keys.float()
            attn = torch.matmul(q_fp32, k_fp32) / math.sqrt(D)
            attn = torch.softmax(attn, dim=-1).type_as(queries) # ???????

        out = torch.matmul(attn, values) + x
        return self.out_projection(self.norm(out)) + out, attn
