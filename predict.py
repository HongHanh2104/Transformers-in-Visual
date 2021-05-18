import torch
from torchvision import transforms as tvtf

from models.model import ViT
from datasets.cifar10 import CIFAR10Dataset
from datasets.dogcat import DogCatDataset

from PIL import Image
import os
import numpy as np
import random
import argparse
import yaml
import csv

CIFAR_LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', \
                'dog', 'frog', 'horse', 'ship', 'truck']


def predict(path, model, sample, device):
    lbl = int(sample[0])
    img = Image.open(os.path.join(path, sample)).convert('RGB')
    img_tensor = tvtf.ToTensor()(img)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    pred, _ = model(img_tensor)
    
    probs = torch.nn.Softmax(dim=-1)(pred)
    rank = torch.argsort(probs, dim=-1, descending=True)
    
    for idx in rank[0, :3]:
        print(f'{sample}: True label: {lbl} - {CIFAR_LABELS[lbl]}, Predicted label: {idx.item()} - {CIFAR_LABELS[idx.item()]} with {probs[0, idx.item()]:.5f}')

def save_csv(path, filename, img_list, pred_list, lbl_list):
    with open(os.path.join(path, filename), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Name', 'Predicted lbl', 'True lbl'])
        writer.writerows(zip(img_list, pred_list, lbl_list))

def set_up(config, device):
    # Load images
    img_path = os.path.join(config['dataset']['root_dir'], 'test')
    images = os.listdir(img_path)
    samples = random.sample(images, config['dataset']['test']['k'])

    # Load model
    model_path = os.path.join(config['pretrained_path'], config['model_filename'] + '.pt')
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


    return model, samples, img_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    # Get device
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)

    model, samples, img_path = set_up(config, device)

    predict(img_path, model, samples[0], device)
    # for sample in range(len(samples)):
    #     predict(sample)