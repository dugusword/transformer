import math
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
    
    def __init__(self, h, d_model, d_K, d_V, d_ff, dropout=0.1):
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
        
        Returns
        -------
        3d tensor (batch_size, seq_len, d_model)
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
    
    def __init__(self, h, d_model, d_K, d_V, d_ff, dropout=0.1):
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

        self.attn1 = MultiHeadAttention(h, d_model, d_K, d_V, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.attn2 = MultiHeadAttention(h, d_model, d_K, d_V, dropout)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, memory, seq):
        """
        Parameters
        ----------
        memory  : 3d tensor (batch_size, in_seq_len, d_model)
            output of the encoder
        seq : 3d tensor (batch_size, out_seq_len, d_model)
            embeded output sequence

        Returns
        -------
        3d tensor (batch_size, seq_len, d_model)
        """
        mask = create_mask(seq.shape[1], seq.device)
        a1 = self.attn1(seq, seq, seq, mask)
        a1 = self.norm1(seq + a1)

        a2 = self.attn2(a1, memory, memory)
        a2 = self.norm2(a1 + a2)

        a3 = self.ffn(a2)
        a3 = self.norm3(a2 + a3)

        return a3


class PositionalEncoding(nn.Module):
    """
    Encode relative positional information into the input sequence tensor
    
    Attributes
    ----------
    pe : 2d tensor (max_len, d_model - if even OR d_model + 1 - if odd)  
    """
    
    def __init__(self, d_model, max_len=10000, dropout=0.1):
        """
        Parameters
        ----------
        d_model : int
            size of embedding
        max_len : int
            maximum length of input sequence
        """
        super(PositionalEncoding, self).__init__()
        # Make the cached tensor even in dim 0 for easy calculation
        if d_model % 2 != 0:
            d_model += 1
        pe = torch.zeros([max_len, d_model])
        pos = torch.arange(0, max_len).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        denom = torch.pow(10000, i / d_model)
        pe[:, 0::2] = torch.sin(pos / denom)
        pe[:, 1::2] = torch.cos(pos / denom)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)

    def forward(self, seq):
        """
        Parameters
        ----------
        seq: 3d tensor (batch_size, seq_len, d_model)
            input sequence

        Returns
        -------
        3d tensor (batch_size, seq_len, d_model)
        """
        seq = seq + self.pe[:seq.shape[1], :seq.shape[2]]
        return self.dropout(seq)

    
class Encoder(nn.Module):
    """
    A stack of EncoderUnit
    Note: This module does not contain positional encoding and token embedding
    Attributes
    ----------
    layers : nn.ModuleList of EncoderUnit 
    """
    
    def __init__(self, n, h, d_model, d_K, d_V, d_ff, dropout=0.1):
        """
        Parameters
        ----------
        n       : int
            number of EncoderUnit 
        h       : int
            number of attention head in each encoder
        d_model : int
            dimension of token embedding
        d_K     : int
            dimension of features in query and key
        d_V     : int
            dimension of features in value
        dropout : float
            dropout probability
        """
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n):
            self.layers.append(EncoderUnit(h, d_model, d_K, d_V, d_ff, dropout))

    def forward(self, seq):
        """
        Parameters
        ----------
        seq : 3d tensor (batch_size, seq_len, d_model)
        
        Returns
        -------
        3d tensor (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            seq = layer(seq)
        return seq
            

class Decoder(nn.Module):
    """
    A stack of DecoderUnit
    Note: This module does not contain positional encoding and token embedding
    Attributes
    ----------
    layers : nn.ModuleList of DecoderUnit
    """
    
    def __init__(self, n, h, d_model, d_K, d_V, d_ff, dropout=0.1):
        """
        Parameters
        ----------
        n       : int
            number of DecoderUnit 
        h       : int
            number of attention head in each decoder
        d_model : int
            dimension of token embedding
        d_K     : int
            dimension of features in query and key
        d_V     : int
            dimension of features in value
        dropout : float
            dropout probability
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n):
            self.layers.append(DecoderUnit(h, d_model, d_K, d_V, d_ff, dropout))


    def forward(self, memory, seq):
        """
        Parameters
        ----------
        memory : 3d tensor (batch_size, seq_len, d_model)
            output of encoder
        seq    : 3d tensor (batch_size, seq_len, d_model)
            embedded tensor of already produced output/target sequence

        Returns
        -------
        3d tensor (batch_size, seq_len, d_model)
        """
        for layer in self.layers:
            seq = layer(memory, seq)
        return seq


class TransformerCore(nn.Module):
    """
    Core Component of Transformer
    This module implements all parts of the Transformer model published
    in the "Attention is All You Need" paper, except for positional encoding
    and token embedding. It can be used independently if custom positional
    encoding and embedding implementations are desired.

    Attributes
    ----------
    encoder : Encoder
    decoder : Decoder
    """
    def __init__(self, n_e=6, n_d=6, h=8, d_model=512,
                 d_K=512, d_V=512, d_ff=2048, dropout=0.1):
        """
        Parameters
        ----------
        n_e     : int
            number of EncoderUnit 
        n_d     : int
            number of DecoderUnit
        h       : int
            number of attention head in each encoder/decoder
        d_model : int
            dimension of token embedding
        d_K     : int
            dimension of features in query and key
        d_V     : int
            dimension of features in value
        dropout : float
            dropout probability
        """
        super(TransformerCore, self).__init__()
        self.encoder = Encoder(n_e, h, d_model, d_K, d_V, d_ff, dropout)
        self.decoder = Decoder(n_d, h, d_model, d_K, d_V, d_ff, dropout)


    def forward(self, in_seq, out_seq):
        """
        Parameters
        ----------
        in_seq  : 3d tensor (batch_size, seq_len, d_model)
            embedded input sequence
        out_seq : 3d tensor (batch_size, seq_len, d_model)
            embedded tensor of already produced output/target sequence

        Returns
        -------
        3d tensor (batch_size, seq_len, d_model)
        """
        memory = self.encoder(in_seq)
        decoded = self.decoder(memory, out_seq)
        return decoded


class Transformer(nn.Module):
    """
    An implementation of the Transformer model, as described in 
    "Attention is All You Need" paper. This includes embedding and postional
    encoding.

    Note: This implementation assumes the vocab size is the same for both
    input and output sequence, hence embedding and positional encoding
    layers are reused. If this is not the desired setup, please use
    the TransformerCore component with customized modules.

    Attributes
    ----------
    embedding : nn.Embedding
    pe        : PositionalEncoding
    core      : TransformerCore
    linear    : nn.Linear
    """
    def __init__(self, n_vocab, n_e=6, n_d=6, h=8, d_model=512,
                 d_K=512, d_V=512, d_ff=2048, dropout=0.1):
        """
        Parameters
        ----------
        n_vocab : int
            size of vocab
        n_e     : int
            number of EncoderUnit 
        n_d     : int
            number of DecoderUnit
        h       : int
            number of attention head in each encoder/decoder
        d_model : int
            dimension of token embedding
        d_K     : int
            dimension of features in query and key
        d_V     : int
            dimension of features in value
        dropout : float
            dropout probability
        """
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(n_vocab, d_model)
        self.factor = math.sqrt(d_model)
        self.pe = PositionalEncoding(d_model)
        self.core = TransformerCore(n_e, n_d, h, d_model, d_K, d_V, d_ff)
        self.linear = nn.Linear(d_model, n_vocab)

    def forward(self, in_seq, out_seq):
        """
        Parameters
        ----------
        in_seq  : 2d tensor of int (batch_size, seq_len)
            input sequence
        out_seq : 2d tensor of int (batch_size, seq_len)
            already produced output/target sequence

        Returns
        -------
        2d tensor (batch_size, seq_len, vocab_size)
            likelyhood of each token's probability in the vocabulary to
            be the next token of out_seq
        """
        in_seq = self.embedding(in_seq) * self.factor
        in_seq = self.pe(in_seq)
        out_seq = self.embedding(out_seq)
        out_seq = self.pe(out_seq)
        probs = self.linear(out_seq)
        return probs
