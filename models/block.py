import torch
from torch import nn

from einops import rearrange

from models.sublayers import MultiHeadAttention, MLP

class Block(nn.Module):
    def __init__(self, dim, 
                 n_layer,
                 n_head,
                 mlp_dim,
                 eps,
                 drop=0.,
                 attn_drop=0.,
                 is_visualize=False):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.norm2 = nn.LayerNorm(dim, eps=eps)

        self.attn = MultiHeadAttention(          
                        dim=dim, 
                        n_head=n_head, 
                        proj_drop=drop,
                        attn_drop=attn_drop,
                        is_visualize=is_visualize
                    )
        self.mlp = MLP(
                        dim=dim,
                        mlp_dim=mlp_dim,
                        drop=drop
                        )
    
    def forward(self, x):
        h = x
        #x = x[0] if isinstance(x, tuple) else x
        x, weights = self.attn(self.norm1(x))
        x = x + h 

        h = x
        x = self.mlp(self.norm2(x)) + h

        return x, weights