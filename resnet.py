'''
Name: Berny Perez and Sonia Nath
Purpose: the purpose of this file is to train a resnet model for project
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from preprocess_PT import load_dataset

#choosing device type
device = torch.device('mps' if torch.backends.mps.is_available() else "cpu")
num_classes = 53

class ResNet(nn.Module):
    '''
    PURPOSE: the purpose of this class is to from a resnet model using transfer learning.
    '''
    def __init__(self):
        super(ResNet,self).__init__()
        #loading in the resnet model
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        #freezing the pretrained model
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        #fully connecting part to num of classes
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features,num_classes)
    
    def forward(self,x):
        return self.resnet(x)

print(device)
if __name__ == "__main__":
    parent_path = Path(__file__).resolve().parent
    train = parent_path / 'data/train'
    valid = parent_path / 'data/valid'
    test = parent_path / 'data/test' 

    train_ds,classes = load_dataset(train,True)
    valid_ds, _ = load_dataset(valid,False)
    test_ds, _ = load_dataset(test,False)

    #declaring model
    model = ResNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    epochs = 10
    best_val_acc = 0.0
    save_path = parent_path / "best_model.pt"

    for epoch in range(epochs):
        # ---- train ----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_ds:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ---- validate ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for images, labels in valid_ds:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  -> saved new best model (val_acc={val_acc:.4f})")