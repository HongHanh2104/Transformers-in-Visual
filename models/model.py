import torch
from torch import nn

from models.block import Block
from models.embedding import Embedding

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
                 is_visualize=False):
        super().__init__()

        self.embedding = Embedding(image_size=image_size,
                                   patch_size=patch_size,
                                   channels=channels,
                                   dim=dim)

        # block of transformers encoder
        self.blocks = nn.Sequential(*[
                        Block(
                            dim=dim,
                            n_layer=n_layer,
                            n_head=n_head,
                            mlp_dim=mlp_dim,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            is_visualize=is_visualize
                           )
                        for i in range(n_layer)
                ])
        
        self.norm = nn.LayerNorm(dim)
        self.to_latent = nn.Identity()
        self.head = nn.Linear(dim, n_class)
        # self.mlp_head = nn.Sequential(
        #     nn.LayerNorm(dim),
        #     nn.Linear(dim, n_class)
        # )

        self.dropout = nn.Dropout(drop_rate)

    def forward(self, img):
        x = self.embedding(img)
        x = self.dropout(x)
        x, weights = self.blocks(x)
        x = self.norm(x)
        x = self.to_latent(x[:, 0])
        x = self.head(x)
        return x, weights

