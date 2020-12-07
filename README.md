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
d_V - number of features in value  
d_K - number of features in query and key  
bs_* - batch size  
l_* - token length  
