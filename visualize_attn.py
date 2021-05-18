import torch
from torchvision import transforms
from skimage.transform import resize

import numpy as np
import os
import argparse
import yaml
import glob
import csv
from collections import defaultdict

from PIL import Image
import matplotlib.pyplot as plt

from models.model import ViT

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', \
                'dog', 'frog', 'horse', 'ship', 'truck']

def setup(config):
    '''
    :param config: config file
    '''

    # Load model
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
    print('LOAD MODEL SUCCESSFULLY!')
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {parameters} trainable parameters.')

    # Load test file
    img_path = os.path.join(config['dataset']['root_dir'], 'test')
    img_dict = defaultdict()

    for i in range(len(LABELS)):
        img_dict[LABELS[i]] = [os.path.basename(img) \
                    for img in glob.glob(os.path.join(img_path, LABELS[i], '*.png'))]
    return model, img_dict, img_path


def attn2mask(device, attn_mat_list):
    attn_mat = torch.stack(attn_mat_list).squeeze(1) # [n_stack, h, n_q, n_k]
    
    # Mean of the attention weights across all heads
    attn_mat = torch.mean(attn_mat, dim=1)
    
    # Create an identity matrix with size of attention map
    # Add the identity matrix to the attention matrix
    # Then re-normalize the weights.
    residual_attn = torch.eye(attn_mat.shape[1]).to(device)
    aug_attn_mat = attn_mat + residual_attn
    aug_attn_mat = aug_attn_mat / aug_attn_mat.sum(dim=-1).unsqueeze(-1)
    
    joint_attns = torch.zeros(aug_attn_mat.shape).to(device)
    joint_attns[0] = aug_attn_mat[0]
    for i in range(1, aug_attn_mat.shape[0]):
        joint_attns[i] = aug_attn_mat[i] @ joint_attns[i - 1]
    
    # Attention from the output to the input
    v = joint_attns[-1]
    grid_size = int(np.sqrt(aug_attn_mat.shape[-1]))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
    mask = mask / mask.max() 
    return mask

def save_img(img, img_name, label, save_path):
    img = Image.fromarray(img)
    img.save(os.path.join(save_path, label, img_name))

def process(device, model, img_list, lbl_name, img_path, save_path):
    #pred_list = []
    for i in range(1):   #len(img_list)):
        path = os.path.join(img_path, lbl_name, img_list[i])
        img = Image.open(path).convert('RGB')
        img_tensor = transforms.ToTensor()(img)
        img_tensor = img_tensor.unsqueeze(0).to(device)
        _, attn_map = model(img_tensor)
        
        mask = attn2mask(device, attn_map)
        #plt.imshow(mask, cmap='hot', interpolation='nearest')
        #plt.savefig(os.path.join(save_path, lbl_name, 'heatmap.png'))
        
        mask = resize(mask, img.size)[:, :, np.newaxis]
        #plt.imshow(mask, cmap='hot', interpolation='nearest')
        #plt.savefig(os.path.join(save_path, lbl_name, 'heatmap_resize.png'))
        
        result = (mask * img).astype('uint8')
        #print(mask)
        save_img(result, img_list[i], lbl_name, save_path)
    print(f'Complete processing class {lbl_name}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--sav_attn')
    args = parser.parse_args()

    # Load config file
    config_path = args.config
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)

    # Create folder to save attention mask
    # and create class folders
    os.makedirs(args.sav_attn, exist_ok=True)
    for lbl in LABELS:
        os.makedirs(os.path.join(args.sav_attn, lbl), exist_ok=True)

    # Get device
    dev_id = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_id)

    model, img_dict, img_path = setup(config)

    #for item in img_dict:
        # print(img_dict[item][0])
    
    process(device, model, img_dict['truck'], 'truck', img_path, args.sav_attn)

    #print(f'Complete visualize attention for class {item}')

    # img = Image.open('test.png').convert('RGB')
    # img_size = img.size
    # img_tensor = transforms.ToTensor()(img)
    # pred, attn_map = model(img_tensor.unsqueeze(0))
    # mask = attn2mask(attn_map)
    # mask = mask / mask.max() 
    # mask = resize(mask, img.size)[:, :, np.newaxis]
    # result = (mask * img).astype('uint8')

    # print(pred)
