import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from simclip_models import MultiModalDataset_Train, MultiModalDataset_Test, ResNetFeatureExtractor, ProjectionHead
from torch.utils.data import DataLoader
from simclip_utils import load_config
import torchvision.transforms as transforms


# --- Fine-Tuning with Model Saving ---
def simclip_finetune(train_dataloader, test_dataloader, config, device):
    """Fine-tuning with classification and model saving."""

    # Define models
    rgb_net = ResNetFeatureExtractor(input_channels=3).to(device)
    depth_net = ResNetFeatureExtractor(input_channels=1).to(device)
    projection_rgb = ProjectionHead(256, 128).to(device)
    projection_depth = ProjectionHead(256, 128).to(device)

    # Load pretrained models
    pretrain_path = os.path.join(config['save_dir'], config['exp_name'], 'pretrain', 'best_model.pth')
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"Pretrained model not found at {pretrain_path}")
    checkpoint = torch.load(pretrain_path, map_location=device)
    rgb_net.load_state_dict(checkpoint['rgb_net_state_dict'])
    depth_net.load_state_dict(checkpoint['depth_net_state_dict'])
    projection_rgb.load_state_dict(checkpoint['projection_rgb_state_dict'])
    projection_depth.load_state_dict(checkpoint['projection_depth_state_dict'])
    print(f"Loaded pretrained models from {pretrain_path}.")

    # Define classifier
    classifier = nn.Sequential(
        nn.Linear(768, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 8)  # Adjust based on your dataset
    ).to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(
        list(rgb_net.parameters()) +
        list(depth_net.parameters()) +
        list(projection_rgb.parameters()) +
        list(projection_depth.parameters()) +
        list(classifier.parameters()),
        lr=config['finetune_learning_rate']
    )
    criterion = nn.CrossEntropyLoss()

    # Create save directory for finetune
    experiment_dir = os.path.join(config['save_dir'], config['exp_name'], 'finetune')
    os.makedirs(experiment_dir, exist_ok=True)

    finetune_epochs = int(config['finetune_epochs'])
    best_accuracy = 0.0

    for epoch in range(finetune_epochs):
        # Training phase
        rgb_net.train()
        depth_net.train()
        projection_rgb.train()
        projection_depth.train()
        classifier.train()

        epoch_loss = 0.0
        for rgb_images,_, depth_images, _, label_y in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{finetune_epochs}]"):
            # 将数据移动到GPU
            rgb_images = rgb_images.to(device)
            depth_images = depth_images.to(device)
            label_y = label_y.to(device)

            # 提取投影特征并 concatenate
            rgb_features = rgb_net(rgb_images)
            depth_features = depth_net(depth_images)
            rgb_features_2 = projection_rgb(rgb_features)
            depth_features_2 = projection_depth(depth_features)


            combined_features = torch.cat((rgb_features, rgb_features_2, depth_features, depth_features_2), dim=1)

            # 通过分类器进行分类
            outputs = classifier(combined_features)

            cls_loss = criterion(outputs, label_y)
            # loss = cls_loss + recon_loss_rgb + recon_loss_depth
            loss = cls_loss
            epoch_loss = cls_loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finetune Epoch [{epoch+1}/{finetune_epochs}], Average Loss: {epoch_loss / len(train_dataloader)}')


        # Evaluation phase
        rgb_net.eval()
        depth_net.eval()
        projection_rgb.eval()
        projection_depth.eval()
        classifier.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for rgb_images, depth_images, labels in tqdm(test_dataloader, desc=f"Evaluating Epoch [{epoch+1}/{finetune_epochs}]"):
                rgb_images = rgb_images.to(device)
                depth_images = depth_images.to(device)
                labels = labels.to(device)

                # Forward pass
                rgb_features = rgb_net(rgb_images)
                depth_features = depth_net(depth_images)
                rgb_proj_features = projection_rgb(rgb_features)
                depth_proj_features = projection_depth(depth_features)

                # Concatenate features
                combined_features = torch.cat((rgb_features, rgb_proj_features, depth_features, depth_proj_features), dim=1)

                # Classification
                outputs = classifier(combined_features)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy after Finetune Epoch [{epoch+1}/{finetune_epochs}]: {accuracy:.2f}%')

        # Save the latest model
        latest_model_path = os.path.join(experiment_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'rgb_net_state_dict': rgb_net.state_dict(),
            'depth_net_state_dict': depth_net.state_dict(),
            'projection_rgb_state_dict': projection_rgb.state_dict(),
            'projection_depth_state_dict': projection_depth.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'accuracy': accuracy,
        }, latest_model_path)
        print(f'Latest finetuned model saved to {latest_model_path}')

        # Save the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_path = os.path.join(experiment_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch + 1,
                'rgb_net_state_dict': rgb_net.state_dict(),
                'depth_net_state_dict': depth_net.state_dict(),
                'projection_rgb_state_dict': projection_rgb.state_dict(),
                'projection_depth_state_dict': projection_depth.state_dict(),
                'classifier_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
            }, best_model_path)
            print(f'New best model saved with accuracy {best_accuracy:.2f}% to {best_model_path}')

    # Clean CUDA cache after training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("CUDA cache cleared.")


# --- Main Script ---
if __name__ == "__main__":
    # Load configuration
    config_path = './config.yml'
    config = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU, cant support")
        
    # Load datasets
    train_dataset = MultiModalDataset_Train(config['train_dataset_path'])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['finetune_batch_size'], 
        shuffle=True, 
        num_workers=config['finetune_num_workers']
    )
    test_dataset = MultiModalDataset_Test(config['test_dataset_path'])
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['test_batch_size'],
        shuffle=False,
        num_workers=config['test_num_workers']
    )

    # Fine-tune with model saving
    simclip_finetune(
        train_dataloader,
        test_dataloader,
        config,
        device
    )
