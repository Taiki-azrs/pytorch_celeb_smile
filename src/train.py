import os
from os.path import join
import sys
import numpy as np
import glob
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data.dataset import Subset
import MyDataSet as mds
import simple_model as sm

def main():
    size_batch = 128
    n_epoch = 20
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = sm.Model().to(device)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    list_loss_train = []
    list_loss_test = []
    list_acc_train = []
    list_acc_test = []
    transform = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),
                                    transforms.ToTensor()])
    dataset=mds.MyDataset("../data",transform)
    #データの分割 ランダムに9:1で学習用とテスト用に分ける
    
    n_samples=len(dataset)
    train_size=int(n_samples*0.9)#学習用画像の比率
    perm=np.random.permutation(n_samples) #ランダムなlist
    train_range=perm[:train_size]
    test_range=perm[train_size:]
    data_train=Subset(dataset,train_range)
    data_test=Subset(dataset,test_range)
    train_loader=DataLoader(data_train,batch_size=size_batch,
                            shuffle=True,num_workers = os.cpu_count())
    test_loader=DataLoader(data_test,batch_size=size_batch,
                           shuffle=True,num_workers = os.cpu_count())
    # データ数の確認
    print(len(data_train), len(data_test), len(dataset))
    for epoch in range(n_epoch):
        print("-----------------------------------------")
        print('epoch: {}'.format(epoch))
        print('train')
        sum_loss = 0.
        sum_acc = 0.
        # 訓練
        model.train()
        for batch_idx,(x_batch,t_batch) in enumerate(train_loader):
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
            
            # 順伝播
            y = model(x_batch)
            loss = F.cross_entropy(y, t_batch)
            
            # 逆伝播
            opt.zero_grad()
            loss.backward()

            # パラメータ更新
            opt.step()
            
            # ロスと精度を蓄積
            sum_loss += loss.item()
            sum_acc += (y.max(1)[1] == t_batch).sum().item()
            
            #進捗の表示
            print(batch_idx,"/",len(train_loader),end="\r")
            sys.stdout.flush()
        
        mean_loss = sum_loss / len(data_train)
        mean_acc = sum_acc / len(data_train)
        list_loss_train.append(mean_loss)
        list_acc_train.append(mean_acc)
        print("- mean loss:", mean_loss)
        print("- mean accuracy:", mean_acc)    

        # Evaluate
        model.eval()
        print('test')
        sum_loss = 0.
        sum_acc = 0.
        for batch_idx,(x_batch,t_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            t_batch = t_batch.to(device)
                
            # forward
            y = model(x_batch)
            loss = F.cross_entropy(y, t_batch)
            
            sum_loss += loss.item()
            sum_acc += (y.max(1)[1] == t_batch).sum().item()

        mean_loss = sum_loss / len(data_test)
        mean_acc = sum_acc / len(data_test)
        list_loss_test.append(mean_loss)
        list_acc_test.append(mean_acc)
        print("- mean loss:", mean_loss)
        print("- mean accuracy:", mean_acc)
    torch.save(model.state_dict(),"./Celeb_Smile_dict.pth")
if __name__ == "__main__":
        main()
