from PIL import Image
import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np

from torchvision import transforms

class AffectDataSet(Dataset):
    def __init__(self, data_path, phase, affcls, transform=None):
        self.phase = phase
        self.transform = transform
        self.data_path = data_path
        self.cls = affcls

        if affcls == 7:
            if phase == 'train':
                df = pd.read_csv(os.path.join(self.data_path, '7cls_train.txt'), sep=' ', header=None,names=['name','label'])
            else:
                df = pd.read_csv(os.path.join(self.data_path, '7cls_val.txt'), sep=' ', header=None,names=['name','label'])
        else:
            if phase == 'train':
                df = pd.read_csv(os.path.join(self.data_path, '8cls_train.txt'), sep=' ', header=None,names=['name','label'])
            else:
                df = pd.read_csv(os.path.join(self.data_path, '8cls_val.txt'), sep=' ', header=None,names=['name','label'])

        file_names = df["name"]
        self.data = df
        self.label = self.data.loc[:, 'label'].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        print(f' distribution of {phase} samples: {self.sample_counts}')

        self.file_paths = []
        if phase == 'train':
            for f in file_names:
                path = os.path.join(self.data_path, 'train', f)
                self.file_paths.append(path)
        else:
            for f in file_names:
                path = os.path.join(self.data_path, 'valid', f)
                self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def get_labels(self):
        return self.label

if __name__ == "__main__":
    train_dataset = AffectDataSet(data_path = "/CSCI2952X/datasets/affectnet",affcls=7, phase = 'train')