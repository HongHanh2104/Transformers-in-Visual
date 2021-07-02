from models.model import ViT
#from timm.models.vision_transformer import VisionTransformer
import torchvision.models as models
from datasets.cifar import CIFARDataset
#from datasets.dogcat import DogCatDataset
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
    
    name_id = config['id']   
    
    # Build dataset
    random.seed(config['seed'])
    print('Building dataset ...')
    train_dataset = CIFARDataset(
                          root_path=config['dataset']['root_dir'],
                          nclasses=config['model']['n_classes'],
                          phase='train'
                    )
    val_dataset = CIFARDataset(
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
                            shuffle=False,
                            num_workers=8
    )  

    
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
                drop_rate=config['model']['drop_rate'],
                attn_drop_rate=config['model']['attn_drop_rate'],
                is_visualize=config['model']['is_visualize']
            )

    # Get pretrained model
    pretrained_path = config['pretrained_path']
    if pretrained_path != None:
        model.load_from(np.load(pretrained_path), custom=True)
        print("Load from pre-trained")
    # model = models.resnet50(pretrained=True)
    # dim_in = model.fc.in_features
    # model.fc = nn.Linear(dim_in, config['model']['n_classes'])
    model = model.to(device)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters} trainable parameters.')

    # Define loss
    random.seed(config['seed'])
    loss = nn.CrossEntropyLoss()
    loss = loss.to(device)

    # Define Optimizer 
    random.seed(config['seed'])
    optimizer = Adam(model.parameters(), 
                    lr=config['trainer']['lr'],
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
