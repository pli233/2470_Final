from PIL import Image
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np

class AffectDataSet(Dataset):
    def __init__(self, data_path, train=True, affcls=7, transform=None, exclude_classes=None):
        self.train = train
        self.transform = transform
        self.data_path = data_path
        self.cls = affcls
        self.exclude_classes = exclude_classes if exclude_classes else []

        # 根据分类数加载文件
        file_name = '7cls_train.txt' if train else '7cls_val.txt'
        df = pd.read_csv(os.path.join(self.data_path, file_name), sep=' ', header=None, names=['name', 'label'])

        # 过滤需要排除的类别
        if self.exclude_classes:
            df = df[~df['label'].isin(self.exclude_classes)]

        # 获取文件名和标签
        file_names = df["name"]
        self.data = df
        self.label = self.data.loc[:, 'label'].values

        # 调试：打印过滤前后的标签分布
        print(f"Original label distribution: {np.unique(self.label, return_counts=True)}")

        # 重新映射标签到连续范围
        unique_labels = np.unique(self.label)
        self.label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        self.label = np.array([self.label_map[label] for label in self.label])

        print(f"Label mapping: {self.label_map}")
        print(f"Adjusted label distribution: {np.unique(self.label, return_counts=True)}")

        # 获取样本路径
        self.file_paths = []
        subfolder = 'train' if train else 'valid'
        for f in file_names:
            path = os.path.join(self.data_path, subfolder, f)
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
