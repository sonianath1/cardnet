import keras
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import utils,layers
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess


#applying image transformation
data_augmentation_layers = [
layers.RandomFlip("horizontal"),
layers.RandomRotation(0.1),
]


def load_datasets(train,valid,test):
    '''
    Purpose: the purpose of this function is to load the dataset using keras
    INPUT: paths for each: train,valid,and test dataset
    OUTPUT: returns a tensor dataset object in order train,valid,test
    '''
    train_ds = utils.image_dataset_from_directory(
    train,
    batch_size=32,
    image_size=(224,224),
    label_mode="int",
    )


    valid_ds = utils.image_dataset_from_directory(
    valid,
    batch_size=32,
    image_size=(224,224),
    )


    test_ds = utils.image_dataset_from_directory(
    test,
    batch_size=32,
    image_size=(224,224),
    shuffle=False,
    )
    return train_ds,valid_ds,test_ds

def data_augmentation(train):
    '''
    Purpose: Takes training data and prefroms scaling and data augemenation
    INPUT: takes a training data loaded with keras
    OUPUT: returns a new dataset with augementation applied to all images in dataset
    '''
    for layer in data_augmentation_layers:
        new_train = layer(train)
    return new_train

def resnet_normalization(train_ds):
    '''
    Purpose: does normalizaiton of a training dataset specified for ResNet50
    INPUT: augmenteted training dataset
    OUPUT: normalized training dataset
    '''
    return train_ds.map(lambda x, y: (resnet_preprocess(x), y))
    

def efficienet_normalization(train_ds):
    '''
    Purpose: does normalization of a training dataset specified for efficNetB0
    INPUT: augmentetd training dataset
    OUTPUT: normalized dataset
    '''
    return train_ds.map(lambda x, y: (effnet_preprocess(x), y)) 

if __name__ == "__main__":
    parent_path = Path(__file__).resolve().parent
    train = parent_path / 'data/train'
    valid = parent_path / 'data/valid'
    test = parent_path / 'data/test' 

    #loading the dataset
    train_ds,valid_ds,test_ds = load_datasets(train,valid,test)
    print(f"data is loaded: \ntrain: {train_ds}\nVaild: {valid_ds}\nTest: {test_ds}")

    #applies image transformation applied to dataset
    aug_train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))


    #Normalized dataset for each network trained
    #Apply ResNet50 normalization
    resnet_dataset = resnet_normalization(aug_train_ds)

   
    #plots data before augementation 
    print("Plotting data from train dataset: ")
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.savefig('original_imgaes.png')
    plt.close()

    #ploting data after augmentation
    print("Plotting data from augemented train dataset: ")
    plt.figure(figsize=(10, 10))
    for images, _ in aug_train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.savefig('augmented_imgaes.png')
    plt.close()


    #ploting data after normalization
    print("Plotting data from normalizaed train dataset: ")
    plt.figure(figsize=(10, 10))
    for images, _ in resnet_dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(np.array(images[i]).astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")
    plt.savefig('normalized_imgaes.png')
    plt.close()

