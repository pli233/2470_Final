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


# --- Augmentation Pipelines ---
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

