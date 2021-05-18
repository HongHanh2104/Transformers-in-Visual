import pickle
import os
import random
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob

LABELS = ['airplane', 'automobile', 'bird', 'cat', 'deer', \
                'dog', 'frog', 'horse', 'ship', 'truck']

def load_images_labels(path, phase, n=0):
    if phase == 'train':
        filename = 'data_batch_' + str(n)
    elif phase == 'test':
        filename = 'test_batch'
    file = open(os.path.join(path, filename), 'rb')
    data = pickle.load(file, encoding='latin1')
    #print(data.keys())
    images = data['data']
    images = np.reshape(images, (10000, 3, 32, 32))
    labels = np.array(data['labels'])
    return images, labels

def load_filenames(path, phase, n=0):
    if phase == 'train':
        filename = 'data_batch_' + str(n)
    elif phase == 'test':
        filename = 'test_batch'
    file = open(os.path.join(path, filename), 'rb')
    data = pickle.load(file, encoding='latin1')
    filenames = data['filenames']
    return filenames

def load_cifar_categories(path):
    filename = 'batches.meta'
    file = open(os.path.join(path, filename), 'rb')
    data = pickle.load(file, encoding='latin1')
    #print(data['label_names'])
    return data['label_names']

def save_cifar_image(base_dir, save_dir, phase, n=0):
    images, labels = load_images_labels(base_dir, phase, n)
    filenames = load_filenames(base_dir, phase, n)
    
    for i in range(len(images)):
        img = images[i].transpose(1, 2, 0)
        plt.imsave(os.path.join(save_dir, phase, str(labels[i]) + '_' + filenames[i]), img)
    #print(f"Complete extract image in batch {n}")

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

def split_by_class(dir, type):
    path = os.path.join(dir, type)
    for item in LABELS:
        os.makedirs(os.path.join(path, item), exist_ok=True)
    imgs = [os.path.basename(img) for img in glob.glob(os.path.join(path, '*.png'))]
    
    for i in range(len(imgs)):
        lbl = int(imgs[i][0])
        shutil.move(os.path.join(path, imgs[i]), os.path.join(path, LABELS[lbl]))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    parser.add_argument('--img_folder')
    args = parser.parse_args()

    img_dir = os.path.join(args.dir, args.img_folder)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
        os.makedirs(os.path.join(img_dir, 'train'))
        os.makedirs(os.path.join(img_dir, 'test'))
    
    # for i in range(1, 6):
    #     save_cifar_image(args.dir, img_dir, 'train', i)
    # print("Complete extract train images.")

    # save_cifar_image(args.dir, img_dir, 'test')
    # print("Complete extract test images.")

    # split_train_val(img_dir)

    split_by_class(img_dir, 'test')