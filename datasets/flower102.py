import torch
from torch.utils.data import Dataset
from torchvision import transforms as tvtf

import os 
import numpy as np
from PIL import Image
from pathlib import Path

class Flower102Dataset(Dataset):
    def __init__(self, 
                 root_path=None,
                 nclasses=102,
                 phase='train'):
        super().__init__()

        assert root_path is not None, "Missing root_path, should be a CIFAR10 dataset!"

        self.root_path = Path(root_path)
        self.path = self.root_path / phase

        self.labels = [lbl for lbl in sorted(os.listdir(self.path))]
        
        self.images = []
        for f in self.labels:
            for img in os.listdir(os.path.join(self.path, f)):
                self.images.append(f + '-' + img)
        self.nclasses = nclasses
        self.phase = phase
    
    def __getitem__(self, idx):
        lbl = self.images[idx].split('-')[0]
        img = self.images[idx].split('-')[1]
        
        path = os.path.join(self.path, lbl, img)
        img = Image.open(path).convert('RGB')

        if self.phase == 'train':
            img = self._train_augmentation(img)
        elif self.phase == 'val':
            img = self._val_augmentation(img)
        # else:
        #     img = tvtf.ToTensor()(img)
        
        lbl = int(lbl) - 1
        return img, lbl

    def __len__(self):
        return len(self.images)

    def _train_augmentation(self, img):
        # resample=Image.BILINEAR
        img = tvtf.Compose(
            [
                tvtf.Resize((224, 224)),
                tvtf.RandomResizedCrop(224),
                tvtf.RandomHorizontalFlip(),
                tvtf.ToTensor()
            ]
        )(img)
        return img

    def _val_augmentation(self, img):
        img = tvtf.Compose(
            [
                tvtf.Resize((224, 224)),
                tvtf.ToTensor()
            ]
        )(img)
        return img
