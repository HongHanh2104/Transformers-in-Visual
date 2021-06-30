import torch
from torch.utils.data import Dataset
from torchvision import transforms as tvtf

import os 
import numpy as np
from PIL import Image
from pathlib import Path

class PetDataset(Dataset):
    def __init__(self, 
                 root_path=None,
                 nclasses=37,
                 phase='train'):
        super().__init__()

        assert root_path is not None, "Missing root_path, should be a CIFAR10 dataset!"

        self.root_path = Path(root_path)
        self.path = self.root_path / phase

        self.labels = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', \
          'British_Shorthair', 'Egyptian_Mau', 'Maine_Coon', \
          'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese', \
          'Sphynx', 'american_bulldog', 'american_pit_bull_terrier', \
          'basset_hound', 'beagle', 'boxer', 'chihuahua', \
          'english_cocker_spaniel', 'english_setter', 'german_shorthaired', \
          'great_pyrenees', 'havanese', 'japanese_chin', \
          'keeshond', 'leonberger', 'miniature_pinscher', \
          'newfoundland', 'pomeranian', 'pug', 'saint_bernard', \
          'samoyed', 'scottish_terrier', 'shiba_inu', \
          'staffordshire_bull_terrier', 'wheaten_terrier', \
          'yorkshire_terrier']
        
        self.images = []
        for lbl in self.labels:
            for img in os.listdir(os.path.join(self.path, lbl)):
                self.images.append(img)
        
        self.nclasses = nclasses
        self.phase = phase
    
    def __getitem__(self, idx):
        lbl = '_'.join(self.images[idx].split('_')[:-1])
        img = self.images[idx]
        
        path = os.path.join(self.path, lbl, img)
        img = Image.open(path).convert('RGB')

        if self.phase == 'train':
            img = self._train_augmentation(img)
        elif self.phase == 'val':
            img = self._val_augmentation(img)
        # else:
        #     img = tvtf.ToTensor()(img)
        
        lbl = int(self.labels.index(lbl))
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
