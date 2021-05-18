import torch
from models.model import ViT
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer
from pprint import pprint

if __name__ == '__main__':
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)

    model = VisionTransformer(
                img_size=224,
                patch_size=16,
                num_classes=10,
                embed_dim=512,
                depth=6,
                num_heads=8,
                mlp_ratio=2.)
    model = model.to('cuda')
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters} trainable parameters.')


    # model = ViT(image_size=224, 
    #           patch_size=16, 
    #           n_class=2, 
    #           dim=768, 
    #           n_layer=12, 
    #           n_head=12, 
    #           mlp_dim=3072,
    #           drop_rate=0.1,
    #           attn_drop_rate=0.0,
    #           is_visualize=False)

    img = torch.randn(1, 3, 224, 224)

    pred = model(img.to('cuda'))
    print(img.shape, pred.shape)
