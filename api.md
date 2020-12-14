---
description: |
    API documentation for modules: modules, modules.attention, modules.feed_forward, modules.transformer.

lang: en

classoption: oneside
geometry: margin=1in
papersize: a4

linkcolor: blue
links-as-notes: true
...


    
# Module `modules` {#modules}




    
## Sub-modules

* [modules.attention](#modules.attention)
* [modules.feed_forward](#modules.feed_forward)
* [modules.transformer](#modules.transformer)






    
# Module `modules.attention` {#modules.attention}






    
## Functions


    
### Function `create_mask` {#modules.attention.create_mask}




>     def create_mask(
>         seq_len
>     )


Helper function to create a square mask

###### Parameters

**```seq_len```** :&ensp;<code>int</code>
:   length of the edge

###### Returns

<code>2d tensor</code> of <code>the following shape</code>
:   &nbsp;


<code>0 1 1 ... 1</code>
:   &nbsp;


<code>0 0 1 ... 1</code>
:   &nbsp;


...
<code>0 0 0 ... 0</code>
:   &nbsp;




    
## Classes


    
### Class `MultiHeadAttention` {#modules.attention.MultiHeadAttention}




>     class MultiHeadAttention(
>         h,
>         d_model,
>         d_K,
>         d_V,
>         dropout=0.1
>     )


Multi-Head Attention Layer

#### Attributes

h       : int
    number of parallel heads
d_K     : int
    dimension of features in both query and key
d_V     : int
    dimension of features in value
**```d_model```** :&ensp;<code>int</code>
:   dimension of token embedding


**```spd_attn```** :&ensp;<code>ScaledDotProductattention layer</code>
:   sub module to apply scaled dot product


W_Q     : 2d tensor (d_model, d_K * h)
    learned parameters used to linearly project query to Q
W_K     : 2d tensor (d_model, d_K * h)
    learned parameters used to linearly project key to K
W_V     : 2d tensor (d_model, d_V * h)
    learned parameters used to linearly project val to V
W_O     : 2d tensor (d_V * h, d_model)
    learned parameters used to linearly project scaled attention
    to the output tensor with the same dimension as input

Initializes internal Module state, shared by both nn.Module and ScriptModule.


    
#### Ancestors (in MRO)

* [torch.nn.modules.module.Module](#torch.nn.modules.module.Module)



    
#### Class variables


    
##### Variable `dump_patches` {#modules.attention.MultiHeadAttention.dump_patches}



Type: `bool`



    
##### Variable `training` {#modules.attention.MultiHeadAttention.training}



Type: `bool`






    
#### Methods


    
##### Method `forward` {#modules.attention.MultiHeadAttention.forward}




>     def forward(
>         self,
>         query,
>         key,
>         val,
>         mask=None
>     ) ‑> Callable[..., Any]


Parameters
-----
**```query```** :&ensp;<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   embedded query sequence


key   : 3d tensor (batch_size, seq_len, d_model)
    embedded key sequence
val   : 3d tensor (batch_size, seq_len, d_model)
    embedded value sequence
mask  : 2d tensor (seq_len, seq_len)
    2d binary tensor, where 1 means pass, 0 means block

###### Returns

<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   &nbsp;



    
##### Method `reset_parameters` {#modules.attention.MultiHeadAttention.reset_parameters}




>     def reset_parameters(
>         self
>     )




    
### Class `ScaledDotProductAttention` {#modules.attention.ScaledDotProductAttention}




>     class ScaledDotProductAttention(
>         dropout=0.1
>     )


Scaled Dot-Product Attention Layer

#### Attributes

**```softmax```** :&ensp;<code>nn.Functional</code>
:   softmax function applied at the last dimension


Initializes internal Module state, shared by both nn.Module and ScriptModule.


    
#### Ancestors (in MRO)

* [torch.nn.modules.module.Module](#torch.nn.modules.module.Module)



    
#### Class variables


    
##### Variable `dump_patches` {#modules.attention.ScaledDotProductAttention.dump_patches}



Type: `bool`



    
##### Variable `training` {#modules.attention.ScaledDotProductAttention.training}



Type: `bool`






    
#### Methods


    
##### Method `forward` {#modules.attention.ScaledDotProductAttention.forward}




>     def forward(
>         self,
>         Q,
>         K,
>         V,
>         mask=None
>     ) ‑> Callable[..., Any]


Parameters
-----
Q    : 4d tensor (batch_size, h, seq_len, d_K)
K    : 4d tensor (batch_size, h, seq_len, d_K)
V    : 4d tensor (batch_size, h, seq_len, d_V)
**```mask```** :&ensp;<code>2d tensor (seq\_len, seq\_len)</code>
:   2d binary tensor, where 1 means connection should be blocked and 
    0 means values can be fed forward to softmax

###### Returns

<code>4d tensor (batch\_size, h, seq\_len, d\_V)</code>
:   &nbsp;





    
# Module `modules.feed_forward` {#modules.feed_forward}







    
## Classes


    
### Class `FeedForwardNetwork` {#modules.feed_forward.FeedForwardNetwork}




>     class FeedForwardNetwork(
>         d_model,
>         d_ff,
>         dropout=0.1
>     )


Position-wise Feed-Forward Network
Apply the following operation
f(x) = max(0, xW1 + b1) * W2 + b2

#### Attributes

**```l1```** :&ensp;`nn.Linear (in_features=d_model, out_features=d_ff)`
:   &nbsp;


**```l2```** :&ensp;`nn.Linear (in_features=d_ff, out_features=d_model)`
:   &nbsp;

#### Parameters

**```d_model```** :&ensp;<code>int</code>
:   embedding length


d_ff    : int
    size of intermediate activation


    
#### Ancestors (in MRO)

* [torch.nn.modules.module.Module](#torch.nn.modules.module.Module)



    
#### Class variables


    
##### Variable `dump_patches` {#modules.feed_forward.FeedForwardNetwork.dump_patches}



Type: `bool`



    
##### Variable `training` {#modules.feed_forward.FeedForwardNetwork.training}



Type: `bool`






    
#### Methods


    
##### Method `forward` {#modules.feed_forward.FeedForwardNetwork.forward}




>     def forward(
>         self,
>         x
>     ) ‑> Callable[..., Any]


Defines the computation performed at every call.

Should be overridden by all subclasses.

**Note:** 
Although the recipe for forward pass needs to be defined within
this function, one should call the :class:<code>Module</code> instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.




    
# Module `modules.transformer` {#modules.transformer}







    
## Classes


    
### Class `Decoder` {#modules.transformer.Decoder}




>     class Decoder(
>         n,
>         h,
>         d_model,
>         d_K,
>         d_V,
>         d_ff,
>         dropout=0.1
>     )


A stack of DecoderUnit
Note: This module does not contain positional encoding and token embedding
#### Attributes

**```layers```** :&ensp;<code>nn.ModuleList</code> of <code>[DecoderUnit](#modules.transformer.DecoderUnit "modules.transformer.DecoderUnit")</code>
:   &nbsp;

#### Parameters

n       : int
    number of DecoderUnit 
h       : int
    number of attention head in each decoder
**```d_model```** :&ensp;<code>int</code>
:   dimension of token embedding


d_K     : int
    dimension of features in query and key
d_V     : int
    dimension of features in value
**```dropout```** :&ensp;<code>float</code>
:   dropout probability




    
#### Ancestors (in MRO)

* [torch.nn.modules.module.Module](#torch.nn.modules.module.Module)



    
#### Class variables


    
##### Variable `dump_patches` {#modules.transformer.Decoder.dump_patches}



Type: `bool`



    
##### Variable `training` {#modules.transformer.Decoder.training}



Type: `bool`






    
#### Methods


    
##### Method `forward` {#modules.transformer.Decoder.forward}




>     def forward(
>         self,
>         memory,
>         seq
>     ) ‑> Callable[..., Any]


Parameters
-----
**```memory```** :&ensp;<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   output of encoder


seq    : 3d tensor (batch_size, seq_len, d_model)
    embedded tensor of already produced output/target sequence

###### Returns

<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   &nbsp;



    
### Class `DecoderUnit` {#modules.transformer.DecoderUnit}




>     class DecoderUnit(
>         h,
>         d_model,
>         d_K,
>         d_V,
>         d_ff,
>         dropout=0.1
>     )


Decoder Unit - building block od decoder

#### Attributes

**```attn1```** :&ensp;<code>MultiHeadAttention</code>
:   self attention of output sequence


**```norm1```** :&ensp;<code>LayerNorm</code>
:   &nbsp;


**```attn2```** :&ensp;<code>MultiHeadAttention</code>
:   output's attention on encoded input (memory)


**```norm2```** :&ensp;<code>LayerNorm</code>
:   &nbsp;


ffn   : FeedForwardNetwork
**```norm3```** :&ensp;<code>LayerNorm</code>
:   &nbsp;

#### Parameters

h       : int
    number of attention heads
**```d_model```** :&ensp;<code>int</code>
:   size of embedding


d_K     : int
    number of features in query and key
d_V     : int
    number of features in value
d_ff    : int
    dimension of the feed-forward network


    
#### Ancestors (in MRO)

* [torch.nn.modules.module.Module](#torch.nn.modules.module.Module)



    
#### Class variables


    
##### Variable `dump_patches` {#modules.transformer.DecoderUnit.dump_patches}



Type: `bool`



    
##### Variable `training` {#modules.transformer.DecoderUnit.training}



Type: `bool`






    
#### Methods


    
##### Method `forward` {#modules.transformer.DecoderUnit.forward}




>     def forward(
>         self,
>         memory,
>         seq
>     ) ‑> Callable[..., Any]


Parameters
-----
memory  : 3d tensor (batch_size, in_seq_len, d_model)
    output of the encoder
**```seq```** :&ensp;<code>3d tensor (batch\_size, out\_seq\_len, d\_model)</code>
:   embeded output sequence

###### Returns

<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   &nbsp;



    
### Class `Encoder` {#modules.transformer.Encoder}




>     class Encoder(
>         n,
>         h,
>         d_model,
>         d_K,
>         d_V,
>         d_ff,
>         dropout=0.1
>     )


A stack of EncoderUnit
Note: This module does not contain positional encoding and token embedding
#### Attributes

**```layers```** :&ensp;<code>nn.ModuleList</code> of <code>[EncoderUnit](#modules.transformer.EncoderUnit "modules.transformer.EncoderUnit") </code>
:   &nbsp;

#### Parameters

n       : int
    number of EncoderUnit 
h       : int
    number of attention head in each encoder
**```d_model```** :&ensp;<code>int</code>
:   dimension of token embedding


d_K     : int
    dimension of features in query and key
d_V     : int
    dimension of features in value
**```dropout```** :&ensp;<code>float</code>
:   dropout probability




    
#### Ancestors (in MRO)

* [torch.nn.modules.module.Module](#torch.nn.modules.module.Module)



    
#### Class variables


    
##### Variable `dump_patches` {#modules.transformer.Encoder.dump_patches}



Type: `bool`



    
##### Variable `training` {#modules.transformer.Encoder.training}



Type: `bool`






    
#### Methods


    
##### Method `forward` {#modules.transformer.Encoder.forward}




>     def forward(
>         self,
>         seq
>     ) ‑> Callable[..., Any]


Parameters
-----
**```seq```** :&ensp;<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   &nbsp;

###### Returns

<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   &nbsp;



    
### Class `EncoderUnit` {#modules.transformer.EncoderUnit}




>     class EncoderUnit(
>         h,
>         d_model,
>         d_K,
>         d_V,
>         d_ff,
>         dropout=0.1
>     )


Encoder Unit - build block of encoder

#### Attributes

attn  : MultiHeadAttention
**```norm1```** :&ensp;<code>LayerNorm</code>
:   &nbsp;


ffn   : FeedForwardNetwork
**```norm2```** :&ensp;<code>LayerNorm</code>
:   &nbsp;

#### Parameters

h       : int
    number of attention heads
**```d_model```** :&ensp;<code>int</code>
:   size of embedding


d_K     : int
    number of features in query and key
d_V     : int
    number of features in value
d_ff    : int
    dimension of the feed-forward network


    
#### Ancestors (in MRO)

* [torch.nn.modules.module.Module](#torch.nn.modules.module.Module)



    
#### Class variables


    
##### Variable `dump_patches` {#modules.transformer.EncoderUnit.dump_patches}



Type: `bool`



    
##### Variable `training` {#modules.transformer.EncoderUnit.training}



Type: `bool`






    
#### Methods


    
##### Method `forward` {#modules.transformer.EncoderUnit.forward}




>     def forward(
>         self,
>         seq
>     ) ‑> Callable[..., Any]


Parameters
-----
**```seq```** :&ensp;<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   &nbsp;

###### Returns

<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   &nbsp;



    
### Class `PositionalEncoding` {#modules.transformer.PositionalEncoding}




>     class PositionalEncoding(
>         d_model,
>         max_len=10000,
>         dropout=0.1
>     )


Encode relative positional information into the input sequence tensor

#### Attributes

**```pe```** :&ensp;`2d tensor (max_len, d_model - if even OR d_model + 1 - if odd)  `
:   &nbsp;

#### Parameters

**```d_model```** :&ensp;<code>int</code>
:   size of embedding


**```max_len```** :&ensp;<code>int</code>
:   maximum length of input sequence




    
#### Ancestors (in MRO)

* [torch.nn.modules.module.Module](#torch.nn.modules.module.Module)



    
#### Class variables


    
##### Variable `dump_patches` {#modules.transformer.PositionalEncoding.dump_patches}



Type: `bool`



    
##### Variable `training` {#modules.transformer.PositionalEncoding.training}



Type: `bool`






    
#### Methods


    
##### Method `forward` {#modules.transformer.PositionalEncoding.forward}




>     def forward(
>         self,
>         seq
>     ) ‑> Callable[..., Any]


Parameters
-----
**```seq```** :&ensp;<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   input sequence

###### Returns

<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   &nbsp;



    
### Class `Transformer` {#modules.transformer.Transformer}




>     class Transformer(
>         n_vocab,
>         n_e=6,
>         n_d=6,
>         h=8,
>         d_model=512,
>         d_K=512,
>         d_V=512,
>         d_ff=2048,
>         dropout=0.1
>     )


An implementation of the Transformer model, as described in 
"Attention is All You Need" paper. This includes embedding and postional
encoding.

Note: This implementation assumes the vocab size is the same for both
input and output sequence, hence embedding and positional encoding
layers are reused. If this is not the desired setup, please use
the TransformerCore component with customized modules.

#### Attributes

**```embedding```** :&ensp;<code>nn.Embedding</code>
:   &nbsp;


pe        : PositionalEncoding
core      : TransformerCore
linear    : nn.Linear

#### Parameters

**```n_vocab```** :&ensp;<code>int</code>
:   size of vocab


n_e     : int
    number of EncoderUnit 
n_d     : int
    number of DecoderUnit
h       : int
    number of attention head in each encoder/decoder
**```d_model```** :&ensp;<code>int</code>
:   dimension of token embedding


d_K     : int
    dimension of features in query and key
d_V     : int
    dimension of features in value
**```dropout```** :&ensp;<code>float</code>
:   dropout probability




    
#### Ancestors (in MRO)

* [torch.nn.modules.module.Module](#torch.nn.modules.module.Module)



    
#### Class variables


    
##### Variable `dump_patches` {#modules.transformer.Transformer.dump_patches}



Type: `bool`



    
##### Variable `training` {#modules.transformer.Transformer.training}



Type: `bool`






    
#### Methods


    
##### Method `forward` {#modules.transformer.Transformer.forward}




>     def forward(
>         self,
>         in_seq,
>         out_seq
>     ) ‑> Callable[..., Any]


Parameters
-----
in_seq  : 2d tensor of int (batch_size, seq_len)
    input sequence
**```out_seq```** :&ensp;<code>2d tensor</code> of <code>int (batch\_size, seq\_len)</code>
:   already produced output/target sequence

###### Returns

<code>2d tensor (batch\_size, seq\_len, vocab\_size)</code>
:   likelyhood of each token's probability in the vocabulary to
    be the next token of out_seq



    
### Class `TransformerCore` {#modules.transformer.TransformerCore}




>     class TransformerCore(
>         n_e=6,
>         n_d=6,
>         h=8,
>         d_model=512,
>         d_K=512,
>         d_V=512,
>         d_ff=2048,
>         dropout=0.1
>     )


Core Component of Transformer
This module implements all parts of the Transformer model published
in the "Attention is All You Need" paper, except for positional encoding
and token embedding. It can be used independently if custom positional
encoding and embedding implementations are desired.

#### Attributes

**```encoder```** :&ensp;<code>[Encoder](#modules.transformer.Encoder "modules.transformer.Encoder")</code>
:   &nbsp;


**```decoder```** :&ensp;<code>[Decoder](#modules.transformer.Decoder "modules.transformer.Decoder")</code>
:   &nbsp;

#### Parameters

n_e     : int
    number of EncoderUnit 
n_d     : int
    number of DecoderUnit
h       : int
    number of attention head in each encoder/decoder
**```d_model```** :&ensp;<code>int</code>
:   dimension of token embedding


d_K     : int
    dimension of features in query and key
d_V     : int
    dimension of features in value
**```dropout```** :&ensp;<code>float</code>
:   dropout probability




    
#### Ancestors (in MRO)

* [torch.nn.modules.module.Module](#torch.nn.modules.module.Module)



    
#### Class variables


    
##### Variable `dump_patches` {#modules.transformer.TransformerCore.dump_patches}



Type: `bool`



    
##### Variable `training` {#modules.transformer.TransformerCore.training}



Type: `bool`






    
#### Methods


    
##### Method `forward` {#modules.transformer.TransformerCore.forward}




>     def forward(
>         self,
>         in_seq,
>         out_seq
>     ) ‑> Callable[..., Any]


Parameters
-----
in_seq  : 3d tensor (batch_size, seq_len, d_model)
    embedded input sequence
**```out_seq```** :&ensp;<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   embedded tensor of already produced output/target sequence

###### Returns

<code>3d tensor (batch\_size, seq\_len, d\_model)</code>
:   &nbsp;




-----
Generated by *pdoc* 0.9.2 (<https://pdoc3.github.io>).
