import torch
from torch import nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, use_mask=False):
        self.use_mask = use_mask
        self.softmax = nn.Softmax(dim=1)

    def forward(self, Q, K, V):
        nom = Q.bmm(K.transpose_(1, 2))
        demon = math.sqrt(K.shape[1])
        scaled = self.softmax(nom / demon)
        res = scaled.bmm(V)
        return res


