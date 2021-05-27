#from datasets.cifar10 import CIFAR10Dataset
#from datasets.dogcat import DogCatDataset
from datasets.flower102 import Flower102Dataset
from torch.utils.data import DataLoader
import torch
import torchvision
import argparse
import PIL
from tqdm import tqdm 
from models.model import ViT
    
parser = argparse.ArgumentParser()
parser.add_argument('--root')
args = parser.parse_args()

dataset = Flower102Dataset(
                        root_path=args.root,
                        n_classes=2,
                        phase='train')
# dataloader = DataLoader(
#                         dataset, 
#                         batch_size=4,
#                         shuffle=True
#                         )
#print(len(dataloader))

model = ViT(image_size=224, 
              patch_size=16, 
              n_class=2, 
              dim=768, 
              n_layer=12, 
              n_head=12, 
              mlp_dim=3072,
              drop_rate=0.1,
              attn_drop_rate=0.0,
              is_visualize=False)
model = model.to('cuda')
parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'The model has {parameters} trainable parameters.')

# print('len: ', len(dataloader))

# for i, (img, lbl) in enumerate(tqdm(dataloader)):
#     img = img.to('cuda')
#     lbl = lbl.to('cuda')
#     out, _ = model(img)
#     print(img.shape, lbl.shape, out.shape)
#     if i == 0:
#         break



