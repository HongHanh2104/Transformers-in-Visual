import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def check_size(x):
    # use for the case when the image/ the patch is not square
    return x if isinstance(x, tuple) else (x, x)

class Embedding(nn.Module):
    def __init__(self, image_size,
                 patch_size,
                 channels,
                 dim):
        super().__init__()

        img_h, img_w = check_size(image_size)
        patch_h, patch_w = check_size(patch_size)

        grid_size = (img_h // patch_h), (img_w // patch_w)
        n_patch = grid_size[0] * grid_size[1]

        self.to_patch_embedding = nn.Conv2d(
                                in_channels=channels,
                                out_channels=dim,
                                kernel_size=(patch_h, patch_w),
                                stride=(patch_h, patch_w))

        self.pos_embedding = nn.Parameter(torch.zeros(1, (n_patch + 1), dim))

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # n_patch = (img_h * img_w) // (patch_h * patch_w)
        # patch_dim = channels * patch_h * patch_w
        
        # self.pos_embedding = nn.Parameter(torch.zeros(1, (n_patch + 1), dim))
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # # img [H, W, C] -> patch [n_patch, patch_dim]
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p_h) (w p_w) -> b (h w) (p_h p_w c)', p_h = patch_h, p_w = patch_w),
        #     nn.Linear(patch_dim, dim)
        # ) # x*E

    def forward(self, img):
        x = self.to_patch_embedding(img)
        x = x.flatten(2).transpose(1, 2) # [b, patch_size, dim]
        
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        
        # Prepend x_class to the sequence of embedded patches
        x = torch.cat((cls_tokens, x), dim=1) # [b, (n + 1), dim]
        
        # # Add pos embedding
        x += self.pos_embedding # [b, (n + 1), dim]
        return x

if __name__ == '__main__':

    img = torch.randn(1, 3, 224, 224)
    embed = Embedding(image_size=224,
                           patch_size=16,
                           channels=3,
                           dim=512)
    embed(img)