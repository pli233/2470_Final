import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from simclip_utils import is_photo, make_mosaic
import albumentations as A
import torchvision.transforms as transforms
import os

# --- Augmentation Pipelines ---
def get_basic_transform():
    basic_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((224, 224), scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
    ])
    return basic_transform
def get_augmentations():
    """Define augmentation pipelines."""
    # Define augmentation pipelines
    transform_1 = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=1),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(0.2, 0.2, 0.1, 0.05, p=0.5),
        A.RandomGamma((80, 120), p=0.5),
        A.Solarize(128, p=0.3),
        A.ChannelShuffle(p=0.2),
        A.GaussianBlur((3, 7), p=0.3),
        A.JpegCompression(50, 100, p=0.3),
    ])

    transform_2 = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=1),
        A.Rotate(20, p=0.5),
        A.Crop(0, 0, 224, 112, p=1),
        A.Resize(224, 224),
        A.InvertImg(p=0.2),
        A.Sharpen(alpha=(0.2, 0.5), p=0.3),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5)
    ])

    transform_3 = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=1),
        A.HorizontalFlip(p=0.5),
        A.Crop(0, 112, 224, 224, p=1),
        A.Resize(224, 224),
        A.RandomGamma((80, 120), p=0.5),
        A.OpticalDistortion(0.2, 0.1, p=0.5),
        A.ColorJitter(0.3, 0.3, 0.2, 0.1, p=0.5),
        A.GaussianBlur((3, 7), p=0.3),
    ])

    transform_4 = A.Compose([
        A.RandomResizedCrop(224, 224, scale=(0.8, 1.0), p=1),
        A.Rotate(15, p=0.5),
        A.Crop(0, 56, 224, 168, p=1),
        A.Resize(224, 224),
        A.Cutout(1, 30, 30, fill_value=0, p=0.5),
        A.Solarize(128, p=0.3),
        A.JpegCompression(30, 100, p=0.3),
        A.ChannelShuffle(p=0.2),
    ])

    return [transform_1, transform_2, transform_3, transform_4]

def get_rich_transform_rgb():
    # Define data transformations
    rich_transform_rgb = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.1),
        transforms.GaussianBlur((5, 5), (0.1, 2)),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.RandomPerspective(0.2, p=0.4),
        transforms.ToTensor(),
    ])
    return rich_transform_rgb

def get_rich_transform_gray():
    rich_transform_gray = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.GaussianBlur((5, 9), (0.1, 3)),
        transforms.RandomAdjustSharpness(2, p=0.3),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomEqualize(p=0.3),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])
    return rich_transform_gray
def get_transform_rgb():
    # Define transforms
    transform_rgb = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    return transform_rgb

def get_transform_gray():
    transform_gray = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    return transform_gray


# --- Model Definitions ---
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        """Feature extractor using ResNet18."""
        super(ResNetFeatureExtractor, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        if input_channels == 1:  # Adjust the first convolutional layer for grayscale images
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove the fully connected layer
        self.fc = nn.Linear(512, 256)  # Add a custom fully connected layer

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        """Projection head for contrastive learning."""
        super(ProjectionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)
    

# --- Dataset Definitions ---
class MultiModalDataset_Gray_Train(Dataset):

    def __init__(self, rgb_root):
        """Initialize dataset with multiple augmentations."""
        self.rgb_dataset = datasets.ImageFolder(root=rgb_root, is_valid_file=is_photo)
        print("Train RGB Dataset size:", len(self.rgb_dataset))
        self.aug_list = get_augmentations()

    def __len__(self):
        return len(self.rgb_dataset)

    def __getitem__(self, idx):
        rgb_path, label = self.rgb_dataset.samples[idx]
        rgb_image_pil = Image.open(rgb_path).convert('RGB')
        rgb_np = np.array(rgb_image_pil)

        # Apply augmentations
        rgb_augs = [aug(image=rgb_np)['image'] for aug in self.aug_list]
        rgb_mosaic = make_mosaic(rgb_augs)

        gray_augs = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in rgb_augs]
        gray_mosaic = make_mosaic(gray_augs)

        rgb_mosaic = cv2.resize(rgb_mosaic, (224, 224))
        gray_mosaic = cv2.resize(gray_mosaic, (224, 224))

        rgb_mosaic_tensor = transforms.ToTensor()(Image.fromarray(rgb_mosaic))
        gray_mosaic_tensor = transforms.ToTensor()(Image.fromarray(gray_mosaic))

        rgb_basic = get_rich_transform_rgb()(rgb_image_pil)
        gray_basic = get_rich_transform_gray()(rgb_image_pil)

        return rgb_basic, rgb_mosaic_tensor, gray_basic, gray_mosaic_tensor, label

class MultiModalDataset_Gray_Test(Dataset):
    def __init__(self, rgb_root):
        # 使用单个数据集目录
        self.rgb_dataset = datasets.ImageFolder(root=rgb_root)

        # 用户提供的transform，用于后续处理
        self.transform_rgb = get_transform_rgb()
        self.transform_gray = get_transform_gray()
        # 数据集长度
        self.dataset_length = len(self.rgb_dataset)
        print("Test RGB Dataset size:", len(self.rgb_dataset))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        # 从RGB数据集中获取图像和标签
        img_path, label = self.rgb_dataset.samples[idx]
        
        # 打开图像（RGB格式）
        image_pil = Image.open(img_path).convert('RGB')

        # 对RGB图像应用transform_rgb
        rgb_image = self.transform_rgb(image_pil)
        # 对同一张图像应用transform_gray，用于生成灰度版
        gray_image = self.transform_gray(image_pil)
        
        # 返回RGB图像、灰度图像和标签
        return rgb_image, gray_image, label


class MultiModalDataset_Landmark(Dataset):
    def __init__(self, rgb_root, landmarks_root):
        self.rgb_dataset = datasets.ImageFolder(root=rgb_root, is_valid_file=is_photo)
        self.landrmark_dataset = datasets.ImageFolder(root=landmarks_root, is_valid_file=is_photo)

        print("RGB 数据集长度:", len(self.rgb_dataset))
        print("Landmarks 数据集长度:", len(self.landrmark_dataset))
        
        self.synchronize_datasets()
        
        self.aug_list = get_augmentations()

    def synchronize_datasets(self):
        def get_relative_paths(dataset, is_landrmark=False):
            relative_paths = []
            for path, _ in dataset.samples:
                rel_path = os.path.relpath(path, dataset.root)
                if is_landrmark:
                    dir_name, file_name = os.path.split(rel_path)
                    base_name = file_name.replace('_landmarks', '')
                    rel_path = os.path.join(dir_name, base_name)
                    rel_path = os.path.splitext(rel_path)[0]  # 去掉扩展名
                else:
                    rel_path = os.path.splitext(rel_path)[0]  # 去掉扩展名
                relative_paths.append(rel_path)
            return relative_paths
        
        rgb_paths = get_relative_paths(self.rgb_dataset)
        landrmark_paths = get_relative_paths(self.landrmark_dataset, is_landrmark=True)
        
        print("RGB 路径样例:", rgb_paths[:5])
        print("Landmarks 路径样例:", landrmark_paths[:5])
        
        common_paths = set(rgb_paths) & set(landrmark_paths)
        print("公共路径数量:", len(common_paths))
        print("公共路径样例:", list(common_paths)[:5])

        if len(common_paths) == 0:
            raise ValueError("RGB 和 Landmarks 数据集没有匹配的文件，请检查路径或文件命名规则。")

        def filter_dataset(dataset, common_paths, is_landrmark=False):
            new_samples = []
            for path, label in dataset.samples:
                rel_path = os.path.relpath(path, dataset.root)
                if is_landrmark:
                    dir_name, file_name = os.path.split(rel_path)
                    base_name = file_name.replace('_landmarks', '')
                    rel_path_processed = os.path.join(dir_name, base_name)
                    rel_path_processed = os.path.splitext(rel_path_processed)[0]  # 去掉扩展名
                else:
                    rel_path_processed = os.path.splitext(rel_path)[0]  # 去掉扩展名
                if rel_path_processed in common_paths:
                    new_samples.append((path, label))
            dataset.samples = new_samples
            dataset.targets = [s[1] for s in new_samples]
        
        filter_dataset(self.rgb_dataset, common_paths)
        filter_dataset(self.landrmark_dataset, common_paths, is_landrmark=True)
        
        self.dataset_length = len(self.rgb_dataset)
        print("同步后 RGB 数据集长度:", len(self.rgb_dataset))
        print("同步后 Landmarks 数据集长度:", len(self.landrmark_dataset))

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, idx):
        rgb_path, label = self.rgb_dataset.samples[idx]
        landmarks_path, label_l = self.landrmark_dataset.samples[idx]
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
        rgb_basic = get_basic_transform()(rgb_image_pil)
        landmarks_basic = get_basic_transform()(landmarks_image_pil)

        return rgb_basic, rgb_mosaic_tensor, landmarks_basic, landmarks_mosaic_tensor, label

# --- Loss Functions ---
def contrastive_loss(features1, features2):
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)
    logits = torch.matmul(features1, features2.T)
    labels = torch.arange(features1.size(0)).to(features1.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def nt_xent_loss(features1, features2, temperature=0.5):
    features1 = F.normalize(features1, p=2, dim=1)
    features2 = F.normalize(features2, p=2, dim=1)
    features = torch.cat([features1, features2], dim=0)
    similarity_matrix = torch.matmul(features, features.T) / temperature
    batch_size = features1.size(0)
    labels = torch.arange(batch_size).to(features1.device)
    labels = torch.cat([labels, labels], dim=0)
    mask = torch.eye(2 * batch_size, dtype=bool).to(features1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
    positives = torch.cat([torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)])
    negatives = similarity_matrix[~mask].view(2 * batch_size, -1)
    logits = torch.cat([positives.unsqueeze(1), negatives], dim=1)
    labels = torch.zeros(2 * batch_size, dtype=torch.long).to(features1.device)
    loss = F.cross_entropy(logits, labels)
    return loss

