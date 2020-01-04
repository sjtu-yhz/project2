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
from cutout3d import Cutout

num_classes = 2
NUM_EPOCHS = 15
BATCH_SIZE=12

normalize = transforms.Normalize(mean=[.5], std=[.5])
transform = transforms.Compose([transforms.ToTensor(), normalize])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def My_mixup(x,y,alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    x1=Variable(lam*x+(1.-lam)*y)
    return x1,lam
#------------------------------------------------- train -----------------------------------------------------------------------
def trainmodel(model,data_train,data_label,c):
    print('-------------------------train seg model-----------------------------------')
    model.train()
    #optimizer = optim.SGD(model.parameters(),lr=0.003,momentum=0.9)
    #criterion = nn.MSELoss() 
    criterion = nn.BCELoss().cuda()
    #优化器
    optimizer = torch.optim.Adam(model.parameters())
    #损失函数
    #criterion = nn.BCELoss()
    all_loss=[] ; x_axi=0 ; a=[] ; brea_num=0 ; tmp_num=0.
    zhemesize=int(465/BATCH_SIZE+1)
    all_batch=1
    tmp_loss=0.
    for nu in range(zhemesize):
        a.append(nu*BATCH_SIZE)
    for epoch in range(NUM_EPOCHS): 
        position_all=np.arange(465)
        pos1=position_all.tolist(); random.shuffle(pos1)
        data_train1=[] ; data_label1=[]
        for dii in range(465):
            data_train1.append(data_train[pos1[dii]])
            data_label1.append(data_label[pos1[dii]])
        data_train=data_train1
        data_label=data_label1 
        correct=0.0
        total=0.0
        i=0
        for tmp in tqdm(a):
            fen_loss=torch.tensor(0).to(device).float()
            pos=[]
            if tmp==a[zhemesize-1]:
                obj=data_train[tmp:465]
                tmp_label=data_label[tmp:465]
            else:
                obj=data_train[tmp:tmp+BATCH_SIZE]
                tmp_label=data_label[tmp:tmp+BATCH_SIZE]
            for san in range(len(obj)):
                pos.append(san)
            random.shuffle(pos)
            #TODO:nothing
            
            for san in range(len(obj)):
                x=np.array(obj[san]) ;x=torch.tensor(x).float() ; x=x.view(-1,1,40,40,40) ; x=x.to(device)
                x_t=tmp_label[san] ; x_la=torch.tensor([[1-float(x_t),float(x_t)]]);x_la=x_la.to(device) 
                out_x=model(x);pre_x=torch.argmax(out_x,1)
                total+=1 ; correct+=(pre_x==x_t).sum().item()
                loss=criterion(out_x,x_la)
                fen_loss+=loss
            loss=fen_loss/len(obj)
            tmp_loss+=loss.float()
            if all_batch % 18==0:
                tmp_loss=tmp_loss/18
                all_loss.append(tmp_loss)
                tmp_loss=0.
                x_axi+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_batch+=1
            fen_loss=torch.tensor(0).to(device).float()
            #TODO:cutout
            for san in range(len(obj)):
                x=np.array(obj[san]) ;x=torch.tensor(x).float() ; x=x.view(-1,1,40,40,40) 
                x=Cutout(x,4,6) #TODO:Change the cutout size and batchs-------
                x=x.to(device)
                x_t=tmp_label[san] ; x_la=torch.tensor([[1-float(x_t),float(x_t)]]);x_la=x_la.to(device) 
                out_x=model(x);pre_x=torch.argmax(out_x,1)
                total+=1 ; correct+=(pre_x==x_t).sum().item()
                loss=criterion(out_x,x_la)
                fen_loss+=loss
            loss=fen_loss/len(obj)
            tmp_loss+=loss.float()
            if all_batch % 18==0:
                tmp_loss=tmp_loss/18
                all_loss.append(tmp_loss)
                tmp_loss=0.
                x_axi+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_batch+=1
            fen_loss=torch.tensor(0).to(device).float()
            
            #TODO:mixup
            for san in range(len(obj)):
                x=np.array(obj[pos[san]]) ; y=np.array(obj[(pos[san]+1)%len(obj)])
                x=torch.tensor(x).float() ; x=x.view(-1,1,40,40,40) ; x=x.to(device)
                y=torch.tensor(y).float() ; y=y.view(-1,1,40,40,40) ; y=y.to(device)

                x_t=tmp_label[pos[san]] ; y_t=tmp_label[(pos[san]+1)%len(obj)]
                vox1,lam=My_mixup(x,y)
                x_la=torch.tensor([[1-float(x_t),float(x_t)]]) ; y_la=torch.tensor([[1-float(y_t),float(y_t)]])
                x_la=x_la.to(device) ; y_la=y_la.to(device)
                out=model(vox1);out_x=model(x) ; out_y=model(y)
                pre_x=torch.argmax(out_x,1) ; pre_y=torch.argmax(out_y,1)
                total+=1
                correct+=(lam*(pre_x==x_t).sum().item()+(1-lam)*(pre_y==y_t).sum().item())
                loss=lam * criterion(out,x_la) + (1 - lam) * criterion(out, y_la)
                fen_loss+=loss
            loss=fen_loss/len(obj)
            tmp_loss+=loss.float()
            if all_batch % 18==0:
                tmp_loss=tmp_loss/18
                all_loss.append(tmp_loss)
                tmp_loss=0.
                x_axi+=1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_batch+=1
        #TODO:early stop
        print(epoch+1," accuracy:{}",correct/total)
        if (abs(tmp_num-correct/total)<0.005) | (tmp_num>correct/total):
            brea_num+=1
        else:
            brea_num=0
        if brea_num==2:
            print("-------------STOP!!!----------------")
            break
        tmp_num=correct/total
    torch.save(model,'D:/test/model2')
    return model,x_axi,all_loss

def test_model1(model1,data_test,data_tevox):
    file1=open("D:/test/Submission1.csv",'w',newline='')
    csv_write=csv.writer(file1)
    csv_write.writerow(["Id","Predicted"])
    i=0
    model1.eval()
    for data in tqdm(data_test):
        vox=np.array(data)
        vox=torch.tensor(vox).float()
        vox=vox.view(-1,1,40,40,40)
        vox=vox.to(device)
        out=model1(vox)
        a=out.tolist()
        row=[conte[i+1][0],a[0][1]/(a[0][0]+a[0][1])]
        csv_write.writerow(row)
        i+=1
    file1.close()

    file2=open("D:/test/Submission2.csv",'w',newline='')
    csv_write=csv.writer(file2)
    csv_write.writerow(["Id","Predicted"])
    i=0
    model2.eval()
    for data in tqdm(data_tevox):
        vox=np.array(data)
        vox=torch.tensor(vox).float()
        vox=vox.view(-1,1,40,40,40)
        vox=vox.to(device)
        out=model2(vox)
        a=out.tolist()
        row=[conte[i+1][0],a[0][1]/(a[0][0]+a[0][1])]
        csv_write.writerow(row)
        i+=1
    file2.close()

