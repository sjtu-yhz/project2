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
from train23d import trainmodel1
# preprocessing
normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":   
    model=LeNet3D()
    model=torch.load('./model_yhz.pkl',map_location=lambda storage, loc: storage)
    data_test,te_label,conte,data_tevoxel=get_data('D:/medical-3d-voxel-classification/test')
    filename="D:/medical-3d-voxel-classification/Submission.csv"
    test_model(model,data_test,conte,filename)