# Transformer
A PyTorch Implementation of Transformer. The original architecture is described in the "Attention is All You Need" paper (https://arxiv.org/abs/1706.03762).
I wrote this for learning purpose, with detailed comments and documentation to help myself and potential reader to follow through.

## Variable Naming Convention
`query` raw query tensor  
`key` raw key tensor  
`value` raw value tensor  
`key`, `query`, `value` are 3d tensors with the following dimension:  
`0` batch  
`1` token of the sequence  
`3` embedding  

`d_model` - dimension of token embedding  
`d_V` - number of features in value  
`d_K` - number of features in query and key  
