import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
from torchvision import transforms as T


class AstroDataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        #Args:
        #    csv_file (string): Path to the csv file with annotations.
        #    root_dir (string): Directory with all the images.
        #    transform (callable, optional): Optional transform to be applied on a sample.

        self.name = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.name)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,self.name.iloc[idx, 0])
        image = Image.open(img_name)
        label = self.name['new_class'][idx]

        if self.transform:
            return (self.transform(image), label)
        
        else:
            return (self.to_tensor(image), label)


