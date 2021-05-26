import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model import ViT
from datasets.dogcat import DogCatDataset

import numpy as np
from tqdm import tqdm
import os 
import argparse
import yaml

@torch.no_grad()
def evaluate(device, model, loss_func, iterator, dataset):
    # 0: Record loss during training process
    total_loss = []
    total_acc = []
    
    # Switch model to training mode
    model.eval()

    # Setup progress bar
    progress_bar = tqdm(iterator)
    for i, (img, lbl) in enumerate(progress_bar):
        # 1: Load sources, targets
        img = img.to(device)
        lbl = lbl.to(device)

        # 2: Get network outputs
        out, _ = model(img)
        
        # 3: Calculate the loss
        loss = loss_func(out, lbl)
        
        # 4: Update loss
        total_loss.append(loss.item())
        
        # 5: Update metric
        out = out.detach()
        lbl = lbl.detach()
        acc = 0.0
        acc += (out.argmax(dim=1) == lbl).sum() #float().mean()
        total_acc.append(acc.item())

    print("++++++++++++++ Evaluation result ++++++++++++++")
    loss = sum(total_loss) / len(iterator)
    print('Loss: ', loss)
    accuracy = sum(total_acc) / len(dataset)
    print('Accuracy: ', accuracy)
    return loss, accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    # Get device
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)

    # Load images
    img_dir = config['dataset']['root_dir']
    
    # Load val dataset
    val_dataset = DogCatDataset(
                          root_path=config['dataset']['root_dir'],
                          nclasses=config['model']['n_classes'],
                          phase='val'
                    )
    val_loader = DataLoader(
                            val_dataset,
                            batch_size=config['dataset']['val']['batch_size'],
                            shuffle=config['dataset']['val']['shuffle'],
                            num_workers=config['dataset']['val']['num_workers']
                    )
    
    #Load model
    model_path = os.path.join(config['pretrained_path'], config['model_filename'])
    model = ViT(image_size=config['model']['img_size'], 
                patch_size=config['model']['patch_size'], 
                n_class=config['model']['n_classes'], 
                dim=config['model']['dim'], 
                n_layer=config['model']['n_layer'], 
                n_head=config['model']['n_head'], 
                mlp_dim=config['model']['mlp_dim'],
                is_visualize=config['model']['is_visualize']
            )
    
    model = model.to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters} trainable parameters.')

    # Loss
    loss = nn.CrossEntropyLoss()
    loss = loss.to(device)

    loss_val, acc_val = evaluate(device, model, loss, val_loader, val_dataset)
    #print(len(val_dataset))
    # f = open('evaluate.txt', 'a')
    # f.write("{} \t {} \t {}\n".format(config['dataset']['val']['batch_size'], loss_val, acc_val))
    # f.close()