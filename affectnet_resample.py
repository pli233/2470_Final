import os
import random
from PIL import Image
from torchvision.utils import save_image
from collections import defaultdict
import torch
def get_class_indices(dataset):
    """
    获取每个类别对应的样本索引。
    """
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    return class_indices

def create_and_save_samples(dataset, save_dir, samples_per_class=3750):
    """
    从每个类别抽取固定数量样本并按类别保存为图片。
    """
    # 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 获取每个类别的索引
    class_indices = get_class_indices(dataset)

    for label, indices in class_indices.items():
        # 创建每个类别的文件夹
        class_folder = os.path.join(save_dir, f"class_{label}")
        os.makedirs(class_folder, exist_ok=True)
        
        # 随机抽取样本
        if len(indices) >= samples_per_class:
            selected_indices = random.sample(indices, samples_per_class)
        else:
            print(f"Warning: Not enough samples for class {label}. Found {len(indices)}, needed {samples_per_class}.")
            selected_indices = indices
        
        # 保存图片
        for i, idx in enumerate(selected_indices):
            image, _ = dataset[idx]  # 获取图像和标签
            image_path = os.path.join(class_folder, f"{i}.png")
            
            # 保存为图像文件
            if isinstance(image, torch.Tensor):
                save_image(image, image_path)  # PyTorch Tensor 保存
            elif isinstance(image, Image.Image):
                image.save(image_path)  # PIL Image 保存

        print(f"Saved {len(selected_indices)} images for class {label} to {class_folder}.")
save_dir = "./dataset/train" 
create_and_save_samples(train_dataset, save_dir, samples_per_class=3750)