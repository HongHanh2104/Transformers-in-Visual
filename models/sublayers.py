import torch
from torch import nn

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
        k_T = k.transpose(-2, -1)
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
        b, n, c = x.shape
        
        #  Step 1: dot product with weight matrices 
        qkv = self.W_qkv(x).reshape(b, n, 3, self.n_head, c // self.n_head).permute(2, 0, 3, 1, 4)
        
        # Step 2: split by number of heads
        # [b, head, n, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Step 3: scale dot product
        score, attn = self.attention(q, k, v)
        weights = attn if self.is_visualize else None
        
        # Step 4: concat and pass to linear layer
        score = score.transpose(1, 2).reshape(b, n, c)
        score = self.proj(score)
        score = self.proj_drop(score)

        return score, weights # score: [b, n, d]
    


