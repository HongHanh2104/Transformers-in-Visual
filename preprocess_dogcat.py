import os
import random
import shutil
import numpy as np
import glob
import argparse

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
    
    split_train_val(img_dir)
