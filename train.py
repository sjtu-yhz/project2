import torch
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from torch import nn,optim
from torch.autograd import Variable
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
from numpy import random
import time
import os
import csv

from dataread import get_data
from lenet3d import LeNet3D
from test3d import test_model
from train13d import trainmodel

# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model1= LeNet3D()
    model1=model1.to(device)

    model2=LeNet3D()
    model1=model2.to(device)

    data_train,data_label,c,data_trvoxel=get_data('D:/medical-3d-voxel-classification/train_val')
    data_test,te_label,conte,data_tevoxel=get_data('D:/medical-3d-voxel-classification/test')
    model1,x_axi,loss=trainmodel(model1,data_train,data_label,c)
    filename="D:/medical-3d-voxel-classification/Submission1.csv"
    test_model(model1,data_test,conte,filename)

    X=np.linspace(0,468,x_axi,endpoint=True)
    plt.plot(X,loss)
    plt.show()
