import torch
from torch import nn

from os.path import join

from models.sublayers import MultiHeadAttention, MLP
from utils.helpers import np2th

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

        self.dim = dim
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
    
    def load_from(self, weights, n_block):
        ROOT = f'Transformer/encoderblock_{n_block}'
        with torch.no_grad():
            query_weight = np2th(weights[join(
                ROOT, 'MultiHeadDotProductAttention_1/query', 'kernel')
                ]).view(self.dim, self.dim).t()
            key_weight = np2th(weights[join(
                ROOT, 'MultiHeadDotProductAttention_1/key', 'kernel')
                ]).view(self.dim, self.dim).t()
            value_weight = np2th(weights[join(
                ROOT, 'MultiHeadDotProductAttention_1/value', 'kernel')
                ]).view(self.dim, self.dim).t()
            out_weight = np2th(weights[join(
                ROOT, 'MultiHeadDotProductAttention_1/out', 'kernel')
                ]).view(self.dim, self.dim).t()
            
            query_bias = np2th(weights[join(
                ROOT, 'MultiHeadDotProductAttention_1/query', 'bias')
                ]).view(-1)
            key_bias = np2th(weights[join(
                ROOT, 'MultiHeadDotProductAttention_1/key', 'bias')
                ]).view(-1)
            value_bias = np2th(weights[join(
                ROOT, 'MultiHeadDotProductAttention_1/value', 'bias')
                ]).view(-1)
            out_bias = np2th(weights[join(
                ROOT, 'MultiHeadDotProductAttention_1/out', 'bias')
                ]).view(-1)
            
            qkv_weight_list = [query_weight, key_weight, value_weight]
            #print(self.attn.W_qkv.weight.shape)
            #print(torch.cat(qkv_weight_list, dim=0).shape)
            self.attn.W_qkv.weight.copy_(torch.cat(qkv_weight_list, dim=0))
            qkv_bias_list = [query_bias, key_bias, value_bias]
            self.attn.W_qkv.bias.copy_(torch.cat(qkv_bias_list, dim=0))
            self.attn.proj.weight.copy_(out_weight)
            self.attn.proj.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[join(
                ROOT, 'MlpBlock_3/Dense_0', 'kernel')
                ]).t()
            mlp_weight_1 = np2th(weights[join(
                ROOT, 'MlpBlock_3/Dense_1', 'kernel')
                ]).t()
            mlp_bias_0 = np2th(weights[join(
                ROOT, 'MlpBlock_3/Dense_0', 'bias')
                ]).t()
            mlp_bias_1 = np2th(weights[join(
                ROOT, 'MlpBlock_3/Dense_1', 'bias')
                ]).t()
                
            self.mlp.mlp[0].weight.copy_(mlp_weight_0)
            self.mlp.mlp[0].bias.copy_(mlp_bias_0)
            self.mlp.mlp[2].weight.copy_(mlp_weight_1)
            self.mlp.mlp[2].bias.copy_(mlp_bias_1)

            self.norm1.weight.copy_(np2th(weights[join(
                ROOT, 'LayerNorm_0', 'scale')]))
            self.norm1.bias.copy_(np2th(weights[join(
                ROOT, 'LayerNorm_0', 'bias')]))
            self.norm2.weight.copy_(np2th(weights[join(
                ROOT, 'LayerNorm_2', 'scale')]))
            self.norm2.bias.copy_(np2th(weights[join(
                ROOT, 'LayerNorm_2', 'bias')]))    