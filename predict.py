import torch
import torch.nn as nn
from torchvision import transforms as tvtf
import torchvision.models as models


from models.model import ViT

from collections import defaultdict
from PIL import Image
import os
import numpy as np
import random
import argparse
import yaml
import csv


def predict(device, path, model, dataset, save_path):
    pred_list = []
    conf_list = []
    for item in dataset:
        img = Image.open(os.path.join(path, str(item) + '.jpg')).convert('RGB')
        img_tensor = tvtf.Compose(
            [
                tvtf.Resize((224, 224)),
                tvtf.ToTensor()
            ]
        )(img)

        img_tensor = img_tensor.unsqueeze(0).to(device)
        pred, _ = model(img_tensor)
        
        probs = torch.nn.Softmax(dim=-1)(pred)
        rank = torch.argsort(probs, dim=-1, descending=True)
        pred_list.append(rank[0, 0].item())
        conf_list.append(probs[0, rank[0, 0].item()].item())
    # Save file
    save_csv(save_path, 'result.csv', dataset, pred_list, conf_list)
        
def save_csv(path, filename, img_list, pred_list, conf_list):
    with open(os.path.join(path, filename), 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['Name', 'Predicted label', 'Confidence'])
        writer.writerows(zip(img_list, pred_list, conf_list))

def compare_results(resnet_csv, vit_csv):
    resnet_csv_path = os.path.join('results', 'ViT-ResNet50-DogCatDataset-lr:0.001-2021_05_21-11_38_09',
                                   'best_acc-val_acc=0.961')
    vit_csv_path = os.path.join('results', 'ViT-DogCatDataset-lr:0.0001-2021_05_19-10_02_23-with_bias',
                                   'best_acc-val_acc=0.769')
    resnet_data = defaultdict(list)
    vit_data = defaultdict(list)

    csv.register_dialect('csv_dialect',
                    delimiter='\t',
                    skipinitialspace=True,
                    quoting=csv.QUOTE_ALL)

    with open(os.path.join(resnet_csv_path, resnet_csv), 'r') as f:
        reader = csv.reader(f, dialect='csv_dialect')
        headers = next(reader)
        for row in reader:
            for h, x in zip(headers, row):
                resnet_data[h].append(x)

    with open(os.path.join(vit_csv_path, vit_csv), 'r') as f:
        reader = csv.reader(f, dialect='csv_dialect')
        headers = next(reader)
        for row in reader:
            for h, x in zip(headers, row):
                vit_data[h].append(x)
    
    diff_list = []
    for i in range(len(vit_data['Predicted label'])):
        if vit_data['Predicted label'][i] != resnet_data['Predicted label'][i]:
            diff_list.append(vit_data['Name'][i])
    print("Number of difference: ", len(diff_list))
    
    with open(os.path.join('results', 'difference.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows([[x] for x in diff_list])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()

    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    # Get device
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)

    save_path = os.path.join(config['result']['root_dir'], 
                             config['pretrained_path'].split('/')[1],
                             config['model_filename'][:-4])
    os.makedirs(save_path, exist_ok=True)   
    
    # Load images
    img_dir = config['dataset']['root_dir']
    dataset = [int(img.split('.')[0]) for img in os.listdir(img_dir)]
    dataset = sorted(dataset)
    
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
    # model = models.resnet50(pretrained=True)
    # dim_in = model.fc.in_features
    # model.fc = nn.Linear(dim_in, config['model']['n_classes'])
    
    model = model.to(device)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters} trainable parameters.')
    
    #predict(device, img_dir, model, dataset, save_path)
    
    