#from datasets.cifar10 import CIFAR10Dataset
#from datasets.dogcat import DogCatDataset
import torch.nn as nn
from datasets.pet import PetDataset
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

dataset = PetDataset(
                        root_path=args.root,
                        nclasses=37,
                        phase='train')

dataloader = DataLoader(
                        dataset, 
                        batch_size=8,
                        shuffle=True,
                        num_workers=8
                        )
print(len(dataset))

model = ViT(image_size=224, 
              patch_size=16, 
              n_class=37, 
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

print('len: ', len(dataloader))
optimizer = torch.optim.Adam(model.parameters(), 
                    lr=0.0001,
                )

loss = nn.CrossEntropyLoss()
loss = loss.to('cuda')

for i, (img, lbl) in enumerate(tqdm(dataloader)):
    #print(img)
    img = img.to('cuda')
    lbl = lbl.to('cuda')
    optimizer.zero_grad()
    out, _ = model(img)
    _l = loss(out, lbl)
    optimizer.step()
    



