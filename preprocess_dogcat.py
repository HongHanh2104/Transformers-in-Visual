import os
import random
import shutil
import numpy as np
import glob
import argparse

def divide_by_class(img_dir, type):
    dog_path = os.path.join(img_dir, type, 'dog')
    cat_path = os.path.join(img_dir, type, 'cat')
    os.makedirs(dog_path, exist_ok=True)
    os.makedirs(cat_path, exist_ok=True)
    
    dataset = [f for f in os.listdir(os.path.join(img_dir, type))]
    for item in dataset:
        lbl = item.split('.')[0] # get the first char
        if lbl == 'cat':
            shutil.move(os.path.join(img_dir, type, item),
                    cat_path)
        else:
            shutil.move(os.path.join(img_dir, type, item),
                    dog_path)
 
def split_train_val(img_dir):
    train_dir = os.path.join(img_dir, 'train')
    assert os.path.exists(train_dir) == True, "Missing train dir!"

    val_dir = os.path.join(img_dir, 'val')
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    train_dataset = [f for f in os.listdir(train_dir)]

    n = round(len(train_dataset) / 100 * 20)
    samples = random.sample(train_dataset, n)
    
    for i in range(len(samples)):
        shutil.move(os.path.join(train_dir, samples[i]),
                    val_dir)  

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    args = parser.parse_args()

    img_dir = args.dir
    
    #split_train_val(img_dir)
    