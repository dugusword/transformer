import torch
from torch import nn
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, use_mask=False):
        super(ScaledDotProductAttention, self).__init__()
        self.use_mask = use_mask
        self.softmax = nn.Softmax(dim=1)

    def forward(self, Q, K, V):
        nom = Q.bmm(K.transpose_(1, 2))
        demon = math.sqrt(K.shape[1])
        scaled = self.softmax(nom / demon)
        res = scaled.bmm(V)
        return res


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, d_K, d_V, use_mask=False):
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.sdp_attn = ScaledDotProductAttention(use_mask)
        self.W_Q = torch.rand(d_model, d_K * h)
        self.W_K = torch.rand(d_model, d_K * h)
        self.W_V = torch.rand(d_model, d_V * h)
        self.W_O = torch.rand(d_V * h, d_model)
        

    def forward(self, Q, K, V):
        h = self.h
        d_V = V.shape[1]
        d_K = K.shape[1]

        # shape d_K by (d_V * h)
        head = torch.zeros(d_K, d_V * h)
        
        for i in range(h):
            nq = Q.bmm(W_Q[:, i * d_K])
            nk = K.bmm(W_K[:, i * d_K])
            nv = V.bmm(W_V[:, i * d_V])
            head[:, i * d_V] = self.spd_attn(nq, nk, nv)

        res = head.bmm(W_O)
        return res
