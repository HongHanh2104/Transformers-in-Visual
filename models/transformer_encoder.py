import torch
from torch import nn

from einops import rearrange

from models.sublayers import LayerNorm, MultiHeadAttention, MLP

class TransformerEncoder(nn.Module):
    def __init__(self, dim, 
                 n_layer,
                 n_head,
                 mlp_dim,
                 drop=0.,
                 attn_drop=0.,
                 is_visualize=False):
        super().__init__()

        self.is_visualize = is_visualize
        self.layer_stack = nn.ModuleList([])

        for _ in range(n_layer):
            self.layer_stack.append(nn.ModuleList([
                    LayerNorm(dim, MultiHeadAttention(
                                        dim=dim, 
                                        n_head=n_head, 
                                        proj_drop=drop,
                                        attn_drop=attn_drop,
                                        is_visualize=is_visualize)),
                    LayerNorm(dim, MLP(
                                    dim=dim, 
                                    mlp_dim=mlp_dim, 
                                    drop=drop))
            ]))
    
    def forward(self, x):
        attn_weights = []
        for attn, mlp in self.layer_stack:
            scores, weights = attn(x)
            #x = attn(x) + x
            x = scores + x
            x = mlp(x) + x
            if self.is_visualize:
                attn_weights.append(weights)
        return x, attn_weights