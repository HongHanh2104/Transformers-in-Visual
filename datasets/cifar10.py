import torch
from torch.utils.data import Dataset
from torchvision import transforms as tvtf

import os 
import numpy as np
from PIL import Image
from pathlib import Path

class CIFAR10Dataset(Dataset):
    def __init__(self, 
                 root_path=None,
                 nclasses=10,
                 phase='train'):
        super().__init__()

        assert root_path is not None, "Missing root_path, should be a CIFAR10 dataset!"

        self.root_path = Path(root_path)
        self.img_path = self.root_path / phase

        self.image_files = [f for f in sorted(os.listdir(self.img_path))]
        self.nclasses = nclasses
        self.phase = phase
        
    
    def __getitem__(self, idx):
        img = self.image_files[idx]
        path = os.path.join(self.img_path, img)
        img = Image.open(path).convert('RGB')

        if self.phase == 'train':
            img = self._train_augmentation(img)
        elif self.phase == 'val':
            img = self._val_augmentation(img)
        else:
            img = tvtf.ToTensor()(img)
        lbl = self.image_files[idx][0] # get the first char
        lbl = int(lbl)
        return img, lbl

    def __len__(self):
        return len(self.image_files)

    def _train_augmentation(self, img):
        # resample=Image.BILINEAR
        img = tvtf.Compose(
            [
                tvtf.Resize((224, 224)),
                tvtf.RandomResizedCrop(224),
                tvtf.RandomHorizontalFlip(),
                tvtf.ToTensor()]
        )(img)
        return img

    def _val_augmentation(self, img):
        img = tvtf.Compose(
            [
                tvtf.Resize((224, 224)),
                tvtf.ToTensor()]
        )(img)
        return img