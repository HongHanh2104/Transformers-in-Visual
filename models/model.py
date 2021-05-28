import torch
from torch import nn

from models.resnet import ResNetv2
from models.block import Block
from models.embedding import PatchEmbedding
from utils.helpers import np2th
from models.weight_init import trunc_normal_

import copy

class Encoder(nn.Module):
    def __init__(self, 
                dim,
                n_layer,
                n_head,
                mlp_dim,
                eps,
                drop,
                attn_drop,
                is_visualize
                ):
        super().__init__()
        self.is_visualize = is_visualize
        self.layer = nn.ModuleList()
        
        for _ in range(n_layer):
            layer = Block(
                        dim=dim,
                        n_layer=n_layer,
                        n_head=n_head,
                        mlp_dim=mlp_dim,
                        eps=eps,
                        drop=drop,
                        attn_drop=attn_drop,
                        is_visualize=is_visualize
                        )
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        attn_weights = []
        for layer_block in self.layer:
            x, weights = layer_block(x)
            if self.is_visualize:
                attn_weights.append(weights)
        
        return x, attn_weights

class ViT(nn.Module):
    def __init__(self, image_size,
                 patch_size, 
                 n_class,
                 dim,
                 n_layer,
                 n_head,
                 mlp_dim,
                 channels=3,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 hybrid_blocks=0, 
                 hybrid_width_factor=0,
                 is_visualize=False):
        super().__init__()

        self.hybrid = False
        self.image_size = self._check_size(image_size)
        
        if hybrid_blocks != 0 and hybrid_width_factor != 0:
            self.hybrid = True
            self.patch_size = (1, 1)
            n_patch = (self.image_size[0] // 16) * (self.image_size[1] // 16) 
            self.hybrid_model = ResNetv2(blocks=hybrid_blocks,
                                         width_factor=hybrid_width_factor)
            channels = self.hybrid_model.width * 16
        else:
            self.patch_size = self._check_size(patch_size)
            grid_size = (self.image_size[0] // self.patch_size[0]), (self.image_size[1] // self.patch_size[1])
            n_patch = grid_size[0] * grid_size[1]

        self.patch_embedding = PatchEmbedding(
                                   image_size=self.image_size,
                                   patch_size=self.patch_size,
                                   channels=channels,
                                   dim=dim)

        # block of transformers encoder
        self.encoder = Encoder(
                            dim=dim,
                            n_layer=n_layer,
                            n_head=n_head,
                            mlp_dim=mlp_dim,
                            eps=1e-6,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            is_visualize=is_visualize
                        )

        self.pos_embedding = nn.Parameter(torch.zeros(1, (n_patch + 1), dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.to_latent = nn.Identity()
        self.head = nn.Linear(dim, n_class)
        self.dropout = nn.Dropout(drop_rate)

        # Weight init
        # head_bias = 0.
        # trunc_normal_(self.pos_embedding, std=.02)
        # trunc_normal_(self.cls_token, std=.02)
        # self.apply(_init_vit_weights)

    def forward(self, x):
        # x is an image
        if self.hybrid:
            x = self.hybrid_model(x) # [b, in_channels, ]
            
        x = self.patch_embedding(x) # [b, patch_size, dim]
        b, n, _ = x.shape

        cls_tokens = self.cls_token.expand(b, -1, -1)

        # Prepend x_class to the sequence of embedded patches
        x = torch.cat((cls_tokens, x), dim=1) # [b, (patch_size + 1), dim]
        
        # Add pos embedding
        x += self.pos_embedding # [b, (patch_size + 1), dim]
        
        x = self.dropout(x)
        
        x, weights = self.encoder(x)
        x = self.norm(x)
        x = self.to_latent(x[:, 0])
        
        x = self.head(x)
        return x, weights

    def load_from(self, weights, custom=True):
        with torch.no_grad():
            if custom:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights['head/kernel']).t())
                self.head.bias.copy_(np2th(weights['head/bias']).t())
            #print(self.patch_embedding.to_patch_embedding.weight.shape)
            #print(np2th(weights['embedding/kernel'], conv=True).shape)
            
            #if self.patch_size[0] == 16:
            self.patch_embedding.to_patch_embedding.weight.copy_(
                        np2th(weights['embedding/kernel'], conv=True))
            self.patch_embedding.to_patch_embedding.bias.copy_(
                        np2th(weights['embedding/bias']))
        
            self.cls_token.copy_(np2th(weights['cls']))
            self.norm.weight.copy_(np2th(weights['Transformer/encoder_norm/bias']))

            pretrain_pos_embed = np2th(weights['Transformer/posembed_input/pos_embedding'])
            pos_embed = self.pos_embedding
            #print(pretrain_pos_embed.size(), pos_embed.size())
            if pretrain_pos_embed.size() == pos_embed.size():
                self.pos_embedding.copy_(pretrain_pos_embed)
            # else:
            #     raise ValueError
            
            for bname, block in self.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)
            


        


    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    def _check_size(self, x):
        # use for the case when the image/ the patch is not square
        return x if isinstance(x, tuple) else (x, x)

def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
    """
    @ Source: https://github.com/rwightman/pytorch-image-models 
    ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
    as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('to_latent'):
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if 'mlp' in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)