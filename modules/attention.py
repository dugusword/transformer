import torch
from torch import nn
import math

def create_mask(seq_len):
    """
    Helper function to create a square mask
    
    Parameters
    ----------
    seq_len : int
        length of the edge

    Returns
    -------
    2d tensor of the following shape
    0 1 1 ... 1
    0 0 1 ... 1
    ...
    0 0 0 ... 0
    """
    ones = torch.ones(seq_len, seq_len)
    mask = torch.triu(ones, diagonal=1).bool()
    return mask

class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention Layer

    Attributes
    ----------
    softmax : nn.Functional
        softmax function applied at the last dimension
    """
    
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask=None):
        """
        Parameters
        ----------
        Q    : 4d tensor (batch_size, h, seq_len, d_K)
        K    : 4d tensor (batch_size, h, seq_len, d_K)
        V    : 4d tensor (batch_size, h, seq_len, d_V)
        mask : 2d tensor (seq_len, seq_len)
            2d binary tensor, where 1 means connection should be blocked and 
            0 means no operation will be done
        
        Returns
        -------
        4d tensor (batch_size, h, seq_len, d_V)
        """
        scaled = torch.matmul(Q, K.transpose_(2, 3))
        scaled = scaled / math.sqrt(K.shape[3])
        if mask is not None:
            scaled.masked_fill_(mask, float('-inf'))
        scaled = self.softmax(scaled)
        attn = torch.matmul(scaled, V)
        return attn


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention Layer

    Attributes
    ----------
    h       : int
        number of parallel heads
    d_K     : int
        dimension of features in both query and key
    d_V     : int
        dimension of features in value
    d_model : int
        dimension of token embedding
    spd_attn: ScaledDotProductattention layer
        sub module to apply scaled dot product
    W_Q     : 2d tensor (d_model, d_K * h)
        learned parameters used to linearly project query to Q
    W_K     : 2d tensor (d_model, d_K * h)
        learned parameters used to linearly project key to K
    W_V     : 2d tensor (d_model, d_V * h)
        learned parameters used to linearly project val to V
    W_O     : 2d tensor (d_V * h, d_model)
        learned parameters used to linearly project scaled attention
        to the output tensor with the same dimension as input
    """
    
    def __init__(self, h, d_model, d_K, d_V):
        super(MultiHeadAttention, self).__init__()

        self.h = h
        self.d_K = d_K
        self.d_V = d_V
        self.d_model = d_model
        
        self.sdp_attn = ScaledDotProductAttention()

        # Here we used a trick to stack h (d_model by d_K) matrices
        # together instead of creating h different linear layers
        # To make the dimension more clear, I spelled out the tensors
        # explicitly instead of using using nn.Linear
        self.W_Q = nn.Parameter(torch.Tensor(d_model, d_K * h))
        self.W_K = nn.Parameter(torch.Tensor(d_model, d_K * h))
        self.W_V = nn.Parameter(torch.Tensor(d_model, d_V * h))
        self.W_O = nn.Parameter(torch.Tensor(d_V * h, d_model))
        self.reset_parameters()

    
    def reset_parameters(self):
        slope = math.sqrt(5)
        nn.init.kaiming_uniform_(self.W_Q, a=slope)
        nn.init.kaiming_uniform_(self.W_K, a=slope)
        nn.init.kaiming_uniform_(self.W_V, a=slope)
        nn.init.kaiming_uniform_(self.W_O, a=slope)

    def forward(self, query, key, val, mask=None):
        """
        Parameters
        ----------
        query : 3d tensor (batch_size, seq_len, d_model)
            embedded query sequence
        key   : 3d tensor (batch_size, seq_len, d_model)
            embedded key sequence
        val   : 3d tensor (batch_size, seq_len, d_model)
            embedded value sequence
        mask  : 2d tensor (seq_len, seq_len)
            2d binary tensor, where 1 means pass, 0 means block
        

        Returns
        -------
        3d tensor (batch_size, seq_len, d_model)
        """
        h, d_K, d_V, d_model = self.h, self.d_K, self.d_V, self.d_model
        W_Q, W_K, W_V, W_O = self.W_Q, self.W_K, self.W_V, self.W_O

        bs_q, l_q = query.shape[0], query.shape[1]
        bs_k, l_k = key.shape[0], key.shape[1]
        bs_v, l_v = val.shape[0], val.shape[1]

        Q = torch.matmul(query, W_Q)
        K = torch.matmul(key, W_K)
        V = torch.matmul(val, W_V)
        
        # Reshape (bs, len, d * h) -> (bs, len, h, d)
        Q = Q.view(bs_q, l_q, h, d_K)
        K = K.view(bs_k, l_k, h, d_K)
        V = V.view(bs_v, l_v, h, d_V)

        # Reshape (bs, len, h, d_k) -> (bs, h, len, d)
        Q.transpose_(1, 2)
        K.transpose_(1, 2)
        V.transpose_(1, 2)

        # head.shape == (bs, h, len, d)
        head = self.sdp_attn(Q, K, V, mask)
        # Reshape into (bs, len, h, d)
        head.transpose_(1, 2)
        # Reshape into (bs, len, h * d)
        head = head.reshape(bs_q, l_q, h * d_V)

        res = torch.matmul(head, W_O)
        return res
