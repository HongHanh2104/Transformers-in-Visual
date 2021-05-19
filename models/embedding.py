import torch
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, image_size,
                 patch_size,
                 channels,
                 dim):
        super().__init__()

        self.img_h, self.img_w = image_size
        patch_h, patch_w = patch_size

        self.to_patch_embedding = nn.Conv2d(
                                in_channels=channels,
                                out_channels=dim,
                                kernel_size=(patch_h, patch_w),
                                stride=(patch_h, patch_w))
        
    def forward(self, img):
        _, _, h, w = img.shape
        assert h == self.img_h and w == self.img_w, \
            f"Input image size ({h}*{w}) doesn't match model ({self.img_h}*{self.img_w})."

        x = self.to_patch_embedding(img)
        x = x.flatten(2).transpose(1, 2) # [b, patch_size, dim]
        return x

if __name__ == '__main__':

    img = torch.randn(1, 3, 224, 224)
    embed = PatchEmbedding(image_size=224,
                           patch_size=16,
                           channels=3,
                           dim=512)
    embed(img)