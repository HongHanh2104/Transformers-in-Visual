import torch
import torch.nn.functional as F

import numpy as np

from models.model import ViT
from models.resnet import ResNetv2
#from timm.models.vision_transformer import VisionTransformer

if __name__ == '__main__':
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)

    # model = VisionTransformer(
    #             img_size=224,
    #             patch_size=16,
    #             num_classes=2,
    #             embed_dim=512,
    #             depth=6,
    #             num_heads=8,
    #             mlp_ratio=2.
    #         )

    model = ViT(image_size=224, 
              patch_size=16, 
              n_class=10, 
              dim=768, 
              n_layer=12, 
              n_head=12, 
              mlp_dim=3072,
              hybrid_blocks=(3, 4, 9),
              hybrid_width_factor=1,
              is_visualize=False)
    
    #model.load_from(np.load('trained/pre-trained/ViT-B_16.npz'), custom=True)
    model = model.to('cuda')
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters} trainable parameters.')
    #print(model)
    
    img = torch.randn(1, 3, 224, 224)

    pred, _ = model(img.to('cuda'))
    print(pred.shape)
    