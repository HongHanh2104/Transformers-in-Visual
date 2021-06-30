import os
import random
import shutil
import numpy as np
import glob
import argparse

LABELS = ['Abyssinian', 'Bengal', 'Birman', 'Bombay', \
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

def create_images_by_class(root_path, images):
    for img in images:
        namefile = img.split('_')
        if len(namefile) == 2:
            lbl = namefile[0]
        elif len(namefile) == 3:
            lbl = namefile[0] + '_' + namefile[1]
        elif len(namefile) == 4:
            lbl = namefile[0] + '_' + namefile[1] + '_' + namefile[2]
        elif len(namefile) == 5:
            lbl = namefile[0] + '_' + namefile[1] + '_' + namefile[2] + '_' + namefile[3]

        shutil.move(os.path.join(root_path, 'images', img),
                    os.path.join(root_path, lbl))

def split_train_val(root_path):
    for lbl in LABELS:
        images = [img for img in os.listdir(os.path.join(root_path, 'train', lbl))]
        n = round(len(images) / 100 * 20)
        val_images = random.sample(images, n)
        for img in val_images:
            shutil.move(os.path.join(root_path, 'train', lbl, img),
                    os.path.join(root_path, 'val', lbl))

root_path = './data/pet_data/'
os.makedirs(os.path.join(root_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(root_path, 'val'), exist_ok=True)

for i in LABELS:
    os.makedirs(os.path.join(root_path, 'train', i), exist_ok=True)
    os.makedirs(os.path.join(root_path, 'val', i), exist_ok=True)

#split_train_val(root_path)