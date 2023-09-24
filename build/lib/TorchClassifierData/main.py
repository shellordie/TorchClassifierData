import numpy as np 
import cv2
import os
import matplotlib.pyplot as plt 
import random
import pandas as pd
import torch
from torch.utils.data import Dataset

def issupported(path):
    issupported=True
    filename,file_ext=os.path.splitext(path)
    support_list=['.jpg','.jpeg','.png','.JPG','.jfif']
    if file_ext in support_list:
        issupported=True
    else:
        print(f"{file_ext} is not supported" )
        issupported=False
    return issupported

class TorchDataset(Dataset):
    def __init__(self,X,y):
        self.X=X
        self.y=y
    def __len__(self):
        return len(self.X)
    def __getitem__(self,idx):
        features=self.X[idx]
        labels=self.y[idx]
        return features,labels

class ClassifierImporter(Dataset):
    X=[]
    y=[]
    def __init__(self,path, categories, size=(28, 28)):
        isDirectory = os.path.isdir(path)
        isList=type(categories)==list
        if isDirectory==False:
            raise TypeError("The 'path' argument should be a directory")
        if isList==False:
            raise TypeError("The 'categories' argument should be a list")
        dataset=[]
        for category in categories:
            dir_path=os.path.join(path, category)
            class_num=categories.index(category)
            for file_name in os.listdir(dir_path):
                if issupported(file_name):
                    img=cv2.imread(os.path.join(dir_path, file_name))
                    img=cv2.resize(img, size)
                    img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    dataset.append([img, class_num])
                else:
                    pass
        random.shuffle(dataset)
        for features, label in dataset:
            self.X.append(features)
            self.y.append(label)
        self.X=np.array(self.X)        
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self,idx):
        image=self.X[idx]
        label=self.y[idx]
        return image,label

def ClassifierTrainTestSpliter(Dataset, test_size=0.2):
    X_list=[]
    y_list=[]
    for i,(X,y) in enumerate(Dataset):
        X_list.append(X)
        y_list.append(y)
    y_list_shape=len(y_list)
    y_test_number=int(y_list_shape*test_size)
    x_test=X_list[: y_test_number]
    x_train=X_list[y_test_number :]
    y_test=y_list[: y_test_number]
    y_train=y_list[y_test_number :]
    trainset=TorchDataset(x_train,y_train)
    testset=TorchDataset(x_test,y_test)
    return trainset,testset

def ClassifierNormalizer(Dataset):
    X_list=[]
    y_list=[]
    for i,(X,y) in enumerate(Dataset):
        row=X.shape[0]
        col=X.shape[1]
        channel=X.shape[2]
        X=X.astype('float32')/255
        norm_X=X.reshape(channel,row,col)
        norm_X=torch.from_numpy(norm_X)
        y=torch.as_tensor(y)
        X_list.append(norm_X)
        y_list.append(y)
    X=X_list[::]
    y=y_list[::]
    new_dataset=TorchDataset(X,y)
    return new_dataset

def ClassifierDisplayer(Dataset,categories,batch=1):
    plt.figure(figsize=(20,10))
    for i, (X,y) in enumerate(Dataset):
        if i==batch*5:
            break
        plt.subplot(batch,5,i+1)
        plt.imshow(X)
        plt.xlabel(categories[y])
    plt.show()

