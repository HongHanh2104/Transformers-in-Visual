import torch
from torch import nn

from einops import rearrange

# class LayerNorm(nn.Module):
#     def __init__(self, dim, fn):
#         super().__init__()
#         self.layernorm = nn.LayerNorm(dim)
#         self.fn = fn
    
#     def forward(self, x, **kwargs):
#         return self.fn(self.layernorm(x), **kwargs)

class MLP(nn.Module):
    def __init__(self, dim, mlp_dim, drop=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        return self.mlp(x)

class ScaledDotProductAttention(nn.Module):
    """
    Compute scaled dot product attention
    q (Query) : given sentence that we focused on (decoder)
    k (Key) : every sentence to check relationship with Qeury(encoder)
    v (Value) : every sentence same with Key (encoder)
    """
    def __init__(self, scale, attn_drop):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
    
    def forward(self, q, k, v):
        # q: [b, h, n, head_dim]
        # k: [b, h, n, head_dim]
        # v: [b, h, n, head_dim]
        
        # Step 1: dot product q with k^T to compute similarity
        # k_tranpose: [b, h, head_dim, n]
        # attn: [b, h, n_q, n_k]
        k_T = rearrange(k, 'b h n d -> b h d n')
        attn = (q @ k_T) * self.scale
        
        # Step 2: Pass to softmax to make [0, 1] range
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # Step 3: Multiply with v
        scores = attn @ v
        return scores, attn

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, 
                 n_head=8,  
                 attn_drop=0.,
                 proj_drop=0.,
                 is_visualize=False):
        super().__init__()
        self.is_visualize = is_visualize
        head_dim = dim // n_head
        self.n_head = n_head
        self.dim = dim
        self.scale = head_dim ** (-0.5)

        self.W_qkv = nn.Linear(dim, dim * 3, bias=True) # Wq, Wk, Wv
        
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.attention = ScaledDotProductAttention(scale=self.scale, attn_drop=attn_drop)
        
    def forward(self, x): 
        # x: [b, n, dim] 
        #print('oeoe: ', len(x))
        b, n, _ = x.shape
        
        #  Step 1: dot product with weight matrices 
        qkv = self.W_qkv(x).chunk(3, dim=-1)
        
        # Step 2: split by number of heads
        # [b, head, n, head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.n_head), qkv)
        
        # Step 3: scale dot product
        score, attn = self.attention(q, k, v)
        weights = attn if self.is_visualize else None
        
        # Step 4: concat and pass to linear layer
        score = rearrange(score, 'b h n d -> b n (h d)')
        score = self.proj(score)
        score = self.proj_drop(score)

        return score, weights # score: [b, n, d]
    


