# Transformer
A PyTorch Implementation of Transformer, for learning and research purpose.

## Variable Naming Convention
query - raw query tensor  
key - raw key tensor  
value - raw value tensor  
key, query, value are 3d tensors with the following dimension:  
0 - batch  
1 - token  
3 - embedding  

d_model - dimension of token embedding  
d_V - dimension of value's attention depth  
d_K - dimension of query and key's attention depth  
bs* - batch size  
l* - token length  
