'''
Name: Berny Perez and Sonia Nath
PURPOSE: the purpose of this file is to preprocess the dataset for a pytorch implementation
'''

import os
import numpy as np
from pathlib import Path
from PIL import Image

import matplotlib.pyplot as plt

import torch
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

#image transformations that normalizes data
data_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


#image transformations that normalizes and augements data
train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    #preforming data augmentation specific to training dataset
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_dataset(ds_path,train=False):
    '''
    Purpose: The purpose of this function is to load in a dataset
    input: path to dataset and flag to identify if it is the training dataset
    output: it will return an dataset object
    '''
    print(f"Loading in dataset: {ds_path}")
    #specific to train dataset
    if train:
        ds = datasets.ImageFolder(root=ds_path,transform=train_transforms)
    else:
        ds = datasets.ImageFolder(root=ds_path,transform=data_transforms)
    class_names = ds.classes

    dataset_loader = DataLoader(
        dataset=ds,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    print("finished loading dataset") 
    return dataset_loader, class_names

def plot_images(images,labels,class_names):
    '''
    Purpose: the purpose of this function is to plot the images
    input: image label and class name
    ouput: saves png with image grid of dataset
    '''
    print("Plotting images to grid......")
    plt.figure(figsize=(12,6))
    #creates a grid of images
    grid = make_grid(images,nrow=4)
    #converts to work with matplotlib
    grid_to_show = grid.permute(1,2,0).numpy()
    plt.imshow(grid_to_show)
    plt.axis('off')
    plt.savefig("org_img.png")
    plt.close()
    print("finished plotting images")

if __name__ == "__main__": 
    
    parent_path = Path(__file__).resolve().parent
    train = parent_path / 'data/train'
    valid = parent_path / 'data/valid'
    test = parent_path / 'data/test' 

    print("loading train dataset")
    train_ds,classes = load_dataset(train,True)
    print("finished loading in trainin")

    #extract batch of images and labesl
    images,labels = next(iter(train_ds))

    #get class name for mapping dataset
    class_names = classes

    plot_images(images,labels,class_names)


    



