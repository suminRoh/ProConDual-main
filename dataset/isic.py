import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import torch


class isic2019_dataset(Dataset):
    def __init__(self,path,transform,num_classes=8,mode='train'):
        self.path = path
        self.transform = transform
        self.mode = mode
        self.num_classes=num_classes
        
        if self.mode == 'train':
            self.df = pd.read_csv(os.path.join(self.path,'ISIC2019_train.csv'))
            self.cls_num_list=self.get_cls_num_list()
        elif self.mode == 'valid':
            self.df = pd.read_csv(os.path.join(self.path,'ISIC2019_val.csv'))
        else:
            self.df = pd.read_csv(os.path.join(self.path,'ISIC2019_test.csv'))

    def get_cls_num_list(self):
        labels = pd.read_csv(os.path.join(self.path,'ISIC2019_train.csv'))
        labels=torch.Tensor(labels['label'].values)
        cls_num_list=[]

        for i in range(self.num_classes):
            count=torch.sum(torch.eq(labels,i))
            cls_num_list.append(count.item())
        
        return cls_num_list
    
    def __getitem__(self, item):
        img_path = os.path.join(self.path,'ISIC2019_Dataset',self.df.iloc[item]['category'],f"{self.df.iloc[item]['image']}.jpg")
        img = Image.open(img_path)
        if (img.mode != 'RGB'):
            img = img.convert("RGB")

        label = int(self.df.iloc[item]['label'])
        label = torch.LongTensor([label])
        if self.transform is not None:
            if self.mode == 'train':
                img1 = self.transform[0](img)
                img2 = self.transform[1](img)
                img3 = self.transform[2](img)

                return [img1,img2, img3],label
            else:
                img1 = self.transform(img)
                return img1, label
        else:
            raise Exception("Transform is None")

    def __len__(self):
        return len(list(self.df['image']))



if __name__ == '__main__':
    isic2019_dataset(path='/data/ISIC2019', transform=None,num_classes=8 ,mode='train')
    