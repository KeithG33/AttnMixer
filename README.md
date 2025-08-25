# Attn-Mixer 
Implements a mixer inspired architecture from the paper [MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/abs/2105.01601) with attention blocks instead of MLPs

## Architecture
The Attn-Mixer block consists of two attention blocks and an MLP. Similar to the MLP-Mixer, there is a channel attention and a token attention where the token attention is computed by tranposing the features and treating the tokens as the sequence. 

To this end, we use a 1D relative-position attention for the tokens, and a 2D relative-position attention for the spatial channels to induce some inductive bias.

```python
# Pseudo-code for SGU block as used in SGU ChessBot
def mixer_attn_block(x):
  x = x + tok_attn(x.T).T # 1d relpos
  x = x + seq_attn(x) # 2d relpos
  x = x + mlp(x)
```

### Training Plots and Scores
*Coming soon*
<!-- ![Training Plot](path_to_training_plot.png) -->

<div align="center">

| Model Name   | Layers | Model Shape  | Params      | Weights       |  Top-1 Accuracy |
|--------------|--------|--------------|-------------|---------------| ----------------|
| Attn-Mixer | 12     | (B, 196, 96)  | ---        | [Download Coming](path_to_model) | --- |

</div>

