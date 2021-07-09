from pathlib import Path
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from torch.utils.data.dataset import Subset
from PIL import Image
import os
class MyDataset(Dataset):
    img_list=[]
    an_list=[]
    def __init__(self,data_dir,transform=None):
        data_path=Path(data_dir)
        for i,img_dir in enumerate(sorted(data_path.glob('*'))[:2]):
            for p in img_dir.iterdir():
                self.img_list.append(p)
                self.an_list.append(i)
        self.transform = transform

        
    def __getitem__(self,index):
        path = self.img_list[index]
        img = Image.open(path)
        #img = img.transpose(2,0,1)
        if self.transform is not None:
            img = self.transform(img)
        return img , self.an_list[index]

    
    def __len__(self):
        return len(self.img_list)


# transform = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor()])
# dataset=MyDataset("data",transform)
# n_samples=len(dataset)
# train_size=int(len(dataset)*0.9)
# train_range=list(range(0,train_size))
# test_range=list(range(train_size,n_samples))
# train_data=Subset(dataset,train_range)
# test_data=Subset(dataset,test_range)
# print(len(dataset))
# print(len(train_data))
# print(len(test_data))
# dataloader = DataLoader(dataset,batch_size=64,shuffle=True,num_workers = os.cpu_count())

# for batch,(i,j) in enumerate(dataloader):
#     print(batch,i.shape)
