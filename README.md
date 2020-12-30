# Transformer
A PyTorch Implementation of Transformer. The original architecture is described in the "Attention is All You Need" paper (https://arxiv.org/abs/1706.03762).
I wrote this for learning purpose, with detailed comments and documentation to help myself and potential reader to follow through.

## A Very Simple Example
```python3
import torch
from modules.transformer import Transformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(27)
model.to(device)

in_seq = torch.LongTensor([[1, 7, 5, 2, 3, 4, 5, 0]]).to(device)
out_seq = torch.LongTensor([[1]]).to(device)

pred = model(in_seq, out_seq)
```

## Documentation of Transformer class
```python3
class Transformer(nn.Module):
"""
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

```

## Naming Conventions of Commonly Used Variables
|Variable|Meaning|
|---|---|
|`d_model`|dimension of token embedding|
|`d_V`|number of features in value|
|`d_K`|number of features in query and key|
|`h`|number of attention heads|
|`in_seq`|input sequence|
|`out_seq`|output or target sequence|
|`query`|raw query tensor|
|`key`|raw key tensor|
|`value`|raw value tensor|
  
`key`, `query`, `value` and embedded `in_seq`, `out_seq` are 3d float tensors with the following dimension:  
|`dim`|Meaning|
|---|---|
|`0`|batch|
|`1`|token of the sequence|
|`3`|embedding|

Raw `in_seq` and `out_seq` are 2d int tensors with the following dimension:  
|`dim`|Meaning|
|---|---|
|`0`|batch|
|`1`|token of the sequence|
