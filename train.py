from models.model import ViT
#from timm.models.vision_transformer import VisionTransformer

#from datasets.cifar10 import CIFAR10Dataset
from datasets.dogcat import DogCatDataset
#from metrics import AccuracyMetric, BLEUMetric
#from optimizers import NoamOptimizer
from trainer import Trainer

import torch
import torch.nn as nn
from torch import optim
from torch.optim import Adam 
from torch.utils.data import DataLoader
import torchvision

import numpy as np 
from argparse import ArgumentParser
from datetime import datetime
import random
import yaml
import argparse
import PIL 

from tqdm import tqdm  

def train(config):
    # Get device
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)
    
    # Get pretrained model
    pretrained_path = config['pretrained_path']
    pretrained = None
    if pretrained_path != None:
        pretrained = torch.load(pretrained_path, map_location=dev_id)
        for item in ["model"]:
            config[item] = pretrained["config"][item]

    name_id = config['id']   
    
    # Build dataset
    random.seed(config['seed'])
    print('Building dataset ...')
    train_dataset = DogCatDataset(
                          root_path=config['dataset']['root_dir'],
                          nclasses=config['model']['n_classes'],
                          phase='train'
                    )
    val_dataset = DogCatDataset(
                          root_path=config['dataset']['root_dir'],
                          nclasses=config['model']['n_classes'],
                          phase='val'
                    )
    
    train_loader = DataLoader(
                            train_dataset, 
                            batch_size=config['dataset']['train']['batch_size'],
                            shuffle=True,
                            num_workers=8
                        )
    val_loader = DataLoader(
                            val_dataset,
                            batch_size=config['dataset']['val']['batch_size'],
                            shuffle=True,
                            num_workers=8
    )  

    # transform = torchvision.transforms.Compose(
    #         [torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
    #         torchvision.transforms.RandomAffine(8, translate=(.15,.15)),
    #         torchvision.transforms.ToTensor(),
    #         ])


    # train_dataset = torchvision.datasets.CIFAR10('./data/test', train=True,
    #                                     download=True, transform=transform)

    # test_dataset = torchvision.datasets.CIFAR10('./data/test', train=False,
    #                                     download=True, transform=transform)

    # train_loader = DataLoader(train_dataset, batch_size=10,
    #                                         shuffle=True)

    # val_loader = DataLoader(test_dataset, batch_size=10,
    #                                         shuffle=False)


    # Define model
    random.seed(config['seed'])
    print('Building model ...')
    model = ViT(
                image_size=config['model']['img_size'], 
                patch_size=config['model']['patch_size'], 
                n_class=config['model']['n_classes'], 
                dim=config['model']['dim'], 
                n_layer=config['model']['n_layer'], 
                n_head=config['model']['n_head'], 
                mlp_dim=config['model']['mlp_dim'],
                is_visualize=config['model']['is_visualize']
                )
    model = model.to(device)

    # Train from pretrained if it is not None
    if pretrained is not None:
        model.load_state_dict(pretrained['model_state_dict'])
    
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters} trainable parameters.')

    # Define loss
    random.seed(config['seed'])
    loss = nn.CrossEntropyLoss()
    loss = loss.to(device)

    # Define metrics
    #random.seed(config['seed'])
    #metric = BLEUMetric()

    # Define Optimizer 
    random.seed(config['seed'])
    optimizer = Adam(model.parameters(), 
                    lr=config['trainer']['lr']
                )
                    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer=optimizer,
                            verbose=True,
                            factor=config['optimizer']['factor'],
                            patience=config['optimizer']['patience']
                        )

    print('Start training ...')

    # Define trainer
    random.seed(config['seed'])
    trainer = Trainer(model=model,
                      device=device,
                      dataloader=(train_loader, val_loader),
                      loss=loss,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      config=config
                    )
    
    # Start to train 
    random.seed(config['seed'])
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    print('Start to train')
    train(config)