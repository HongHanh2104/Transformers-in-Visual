import torch
from torchvision import transforms
from torchvision import transforms as tvtf
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

# Image files that want to visualize attention
FILES = ['1', '3', '4', '7', '21']

def attn2mask(device, attn_mat_list):
    attn_mat = torch.stack(attn_mat_list).squeeze(1) # [n_stack, h, (patch_size + 1), (patch_size + 1)]
    
    # Mean of the attention weights across all heads
    attn_mat = torch.mean(attn_mat, dim=1) # [n_stack, (patch_size + 1), (patch_size + 1)]
    
    # Create an identity matrix with size of attention map
    # Add the identity matrix to the attention matrix
    # Then re-normalize the weights.
    residual_attn = torch.eye(attn_mat.shape[1]).to(device) # [(patch_size + 1), (patch_size + 1)]
    aug_attn_mat = attn_mat + residual_attn
    aug_attn_mat = aug_attn_mat / aug_attn_mat.sum(dim=-1).unsqueeze(-1)
    
    joint_attns = torch.zeros(aug_attn_mat.shape).to(device)
    joint_attns[0] = aug_attn_mat[0]
    
    for i in range(1, aug_attn_mat.shape[0]):
        joint_attns[i] = aug_attn_mat[i] @ joint_attns[i - 1]
    
    # Attention from the output to the input
    v = joint_attns[0]
    grid_size = int(np.sqrt(aug_attn_mat.shape[-1]))
    mask = v[0, 1:].reshape(grid_size, grid_size).detach().cpu().numpy()
    mask = mask / mask.max() 
    return mask

def save_img(img, img_name, save_path):
    img = Image.fromarray(img)
    img.save(os.path.join(save_path, img_name + '.png'))

def _to_img_tensor(img):
    img_tensor = tvtf.Compose(
            [
                tvtf.Resize((224, 224)),
                tvtf.ToTensor()
            ]
        )(img)

    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def process(device, model, img_path, save_path):
    #pred_list = []
    for item in FILES:   #len(img_list)):
        img = Image.open(os.path.join(img_path, item + '.jpg')).convert('RGB')
        img_size = (np.array(img).shape[:2]) 
        img_tensor = _to_img_tensor(img).to(device)
        _, attn_map = model(img_tensor)
        #print(attn_map[0][:, 0].shape)
        mask = attn2mask(device, attn_map)
        #plt.imshow(mask, cmap='hot', interpolation='nearest')
        #plt.savefig(os.path.join(save_path, item + '_heatmap.png'))
        
        mask = resize(mask, img_size)[:, :, np.newaxis]
        
        #plt.imshow(mask, cmap='hot', interpolation='nearest')
        #plt.savefig(os.path.join(save_path, item + '_heatmap_resize.png'))
        #print(mask.size, img.size)
        result = (mask * img).astype('uint8')
        #plt.imshow(mask)
        #plt.savefig(os.path.join(save_path, item + '_mask_0.png'))
        #save_img(result, item + '_and_mask', save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    args = parser.parse_args()

    # Load config file
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
    img_path = config['dataset']['root_dir']
    
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
    
    process(device, model, img_path, save_path)

    # img = Image.open('test.png').convert('RGB')
    # img_size = img.size
    # img_tensor = transforms.ToTensor()(img)
    # pred, attn_map = model(img_tensor.unsqueeze(0))
    # mask = attn2mask(attn_map)
    # mask = mask / mask.max() 
    # mask = resize(mask, img.size)[:, :, np.newaxis]
    # result = (mask * img).astype('uint8')

    # print(pred)
