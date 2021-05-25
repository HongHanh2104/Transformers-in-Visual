import torch
from models.model import ViT
import torch.nn.functional as F

from timm.models.vision_transformer import VisionTransformer

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
              n_class=2, 
              dim=512, 
              n_layer=6, 
              n_head=8, 
              mlp_dim=1024,
              is_visualize=True)

    model = model.to('cuda')
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters} trainable parameters.')


    
    img = torch.randn(1, 3, 224, 224)

    pred, _ = model(img.to('cuda'))
    print(pred.shape)
    