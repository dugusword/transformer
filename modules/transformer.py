import torch
from torch import nn
from .attention import MultiHeadAttention, create_mask
from .feed_forward import FeedForwardNetwork

class EncoderUnit(nn.Module):
    """
    Encoder Unit - build block of encoder

    Attributes
    ----------
    attn  : MultiHeadAttention
    norm1 : LayerNorm
    ffn   : FeedForwardNetwork
    norm2 : LayerNorm
    """
    
    def __init__(self, h, d_model, d_K, d_V, d_ff):
        """
        Parameters
        ----------
        h       : int
            number of attention heads
        d_model : int
            size of embedding
        d_K     : int
            number of features in query and key
        d_V     : int
            number of features in value
        d_ff    : int
            dimension of the feed-forward network
        """
        super(EncoderUnit, self).__init__()
        self.attn = MultiHeadAttention(h, d_model, d_K, d_V)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, seq):
        """
        Parameters
        ----------
        seq : 3d tensor (batch_size, seq_len, d_model)
        """
        a1 = self.attn(seq, seq, seq)
        a1 = self.norm1(seq + a1)
        
        a2 = self.ffn(a1)
        a2 = self.norm2(a1 + a2)

        return a2

class DecoderUnit(nn.Module):
    """
    Decoder Unit - building block od decoder

    Attributes
    ----------
    attn1 : MultiHeadAttention
        self attention of output sequence
    norm1 : LayerNorm
    attn2 : MultiHeadAttention
        output's attention on encoded input (memory)
    norm2 : LayerNorm
    ffn   : FeedForwardNetwork
    norm3 : LayerNorm
    """
    
    def __init__(self, h, d_model, d_K, d_V, d_ff):
        """
        Parameters
        ----------
        h       : int
            number of attention heads
        d_model : int
            size of embedding
        d_K     : int
            number of features in query and key
        d_V     : int
            number of features in value
        d_ff    : int
            dimension of the feed-forward network
        """
        super(DecoderUnit, self).__init__()

        self.attn1 = MultiHeadAttention(h, d_model, d_K, d_V)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn2 = MultiHeadAttention(h, d_model, d_K, d_V)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, memory, out_seq):
        """
        Parameters
        ----------
        memory  : 3d tensor (batch_size, in_seq_len, d_model)
            output of the encoder
        out_seq : 3d tensor (batch_size, out_seq_len, d_model)
            embeded output sequence
        """
        mask = create_mask(out_seq.shape[1])
        a1 = self.attn1(out_seq, out_seq, out_seq, mask)
        a1 = self.norm1(out_seq + a1)

        a2 = self.attn2(a1, memory, memory)
        a2 = self.norm2(a1 + a2)

        a3 = self.ffn(a2)
        a3 = self.norm3(a2 + a3)

        return a3
