import os
import numpy as np
from PIL import Image
import cv2
import albumentations as A
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def is_jpg(file_path):
    return file_path.lower().endswith('.jpg')

# 为每个增广transform加入additional_targets参数
transform_1 = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.8,1.0), p=1),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
    A.RandomGamma(gamma_limit=(80,120), p=0.5),
    A.Solarize(threshold=128, p=0.3),
    A.ChannelShuffle(p=0.2),
    A.GaussianBlur(blur_limit=(3,7), p=0.3),
    A.JpegCompression(quality_lower=50, quality_upper=100, p=0.3),
    A.Resize(224,224),
], additional_targets={'image2': 'image'})

transform_2 = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.8,1.0), p=1),
    A.Rotate(limit=20, p=0.5),
    A.Crop(x_min=0, y_min=0, x_max=224, y_max=112, p=1),
    A.Resize(224,224),
    A.InvertImg(p=0.2),
    A.Sharpen(alpha=(0.2,0.5), p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
], additional_targets={'image2': 'image'})

transform_3 = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.8,1.0), p=1),
    A.HorizontalFlip(p=0.5),
    A.Crop(x_min=0, y_min=112, x_max=224, y_max=224, p=1),
    A.Resize(224,224),
    A.RandomGamma(gamma_limit=(80,120), p=0.5),
    A.OpticalDistortion(distort_limit=0.2, shift_limit=0.1, p=0.5),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3,7), p=0.3),
], additional_targets={'image2': 'image'})

transform_4 = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.8,1.0), p=1),
    A.Rotate(limit=15, p=0.5),
    A.Crop(x_min=0, y_min=56, x_max=224, y_max=168, p=1),
    A.Resize(224,224),
    A.Cutout(num_holes=1, max_h_size=30, max_w_size=30, fill_value=0, p=0.5),
    A.Solarize(threshold=128, p=0.3),
    A.JpegCompression(quality_lower=30, quality_upper=100, p=0.3),
    A.ChannelShuffle(p=0.2),
], additional_targets={'image2': 'image'})


basic_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
])

def make_mosaic(images):
    row1 = np.hstack([images[0], images[1]])
    row2 = np.hstack([images[2], images[3]])
    mosaic = np.vstack([row1, row2])
    return mosaic

class MultiModalLandmarksQuadAugDataset(Dataset):
    def __init__(self, rgb_root, landmarks_root):
        self.rgb_dataset = datasets.ImageFolder(root=rgb_root, is_valid_file=is_jpg)
        self.depth_dataset = datasets.ImageFolder(root=landmarks_root, is_valid_file=is_jpg)

        print("RGB 数据集长度:", len(self.rgb_dataset))
        print("Landmarks 数据集长度:", len(self.depth_dataset))
        
        self.synchronize_datasets()
        
        self.aug_list = [transform_1, transform_2, transform_3, transform_4]

    def synchronize_datasets(self):
        def get_relative_paths(dataset, is_depth=False):
            relative_paths = []
            for path, _ in dataset.samples:
                rel_path = os.path.relpath(path, dataset.root)
                if is_depth:
                    dir_name, file_name = os.path.split(rel_path)
                    base_name = file_name.replace('_landmarks', '')
                    rel_path = os.path.join(dir_name, base_name)
                relative_paths.append(rel_path)
            return relative_paths
        
        rgb_paths = get_relative_paths(self.rgb_dataset)
        depth_paths = get_relative_paths(self.depth_dataset, is_depth=True)
        
        common_paths = set(rgb_paths) & set(depth_paths)
        
        def filter_dataset(dataset, common_paths, is_depth=False):
            new_samples = []
            for path, label in dataset.samples:
                rel_path = os.path.relpath(path, dataset.root)
                if is_depth:
                    dir_name, file_name = os.path.split(rel_path)
                    base_name = file_name.replace('_landmarks', '')
                    rel_path_processed = os.path.join(dir_name, base_name)
                else:
                    rel_path_processed = rel_path
                if rel_path_processed in common_paths:
                    new_samples.append((path, label))
            dataset.samples = new_samples
            dataset.targets = [s[1] for s in new_samples]
        
        filter_dataset(self.rgb_dataset, common_paths)
        filter_dataset(self.depth_dataset, common_paths, is_depth=True)
        
        self.dataset_length = len(self.rgb_dataset)
        print("同步后的数据集长度:", self.dataset_length)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        rgb_path, label = self.rgb_dataset.samples[idx]
        landmarks_path, label_l = self.depth_dataset.samples[idx]
        assert label == label_l, "RGB和Landmarks的标签不一致"
        
        # 加载原始图像
        rgb_image_pil = Image.open(rgb_path).convert('RGB')
        landmarks_image_pil = Image.open(landmarks_path).convert('RGB') 

        # 转为numpy用于Albumentations
        rgb_np = np.array(rgb_image_pil)
        landmarks_np = np.array(landmarks_image_pil)

        # 对同一对图像应用四种不同增强
        rgb_augs = []
        landmarks_augs = []
        for aug in self.aug_list:
            out = aug(image=rgb_np, image2=landmarks_np)
            rgb_augs.append(out['image'])
            landmarks_augs.append(out['image2'])

        # 将四个增强后的图像拼接成2x2马赛克
        rgb_mosaic = make_mosaic(rgb_augs)
        landmarks_mosaic = make_mosaic(landmarks_augs)

        # 缩放回224x224
        rgb_mosaic = cv2.resize(rgb_mosaic, (224,224), interpolation=cv2.INTER_LINEAR)
        landmarks_mosaic = cv2.resize(landmarks_mosaic, (224,224), interpolation=cv2.INTER_LINEAR)

        # 转为Tensor
        rgb_mosaic_tensor = transforms.ToTensor()(Image.fromarray(rgb_mosaic))
        landmarks_mosaic_tensor = transforms.ToTensor()(Image.fromarray(landmarks_mosaic))

        # 基本版本（无增强）的图像
        rgb_basic = basic_transform(rgb_image_pil)
        landmarks_basic = basic_transform(landmarks_image_pil)

        return rgb_basic, rgb_mosaic_tensor, landmarks_basic, landmarks_mosaic_tensor, label


if __name__ == "__main__":
    rgb_root='/workspace/2470_Final/affectnet/AffectNet/train'
    landmarks_root='/workspace/2470_Final/affectnet/AffectNet_landmarks/train'

    dataset = MultiModalLandmarksQuadAugDataset(rgb_root, landmarks_root)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

    for rgb_basic, rgb_mosaic, landmarks_basic, landmarks_mosaic, label in dataloader:
        print("RGB基本版:", rgb_basic.shape)
        print("RGB 4x增强马赛克:", rgb_mosaic.shape)
        print("Landmarks基本版:", landmarks_basic.shape)
        print("Landmarks 4x增强马赛克:", landmarks_mosaic.shape)
        print("标签:", label)
        break