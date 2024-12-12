import os
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from simclip_models import MultiModalDataset_Landmark, ResNetFeatureExtractor, ProjectionHead
from torch.utils.data import DataLoader
from simclip_utils import load_config


# --- Fine-Tuning with Model Saving ---
def simclip_landmark_finetune(train_dataloader, test_dataloader, config, device):
    """Fine-tuning with classification and model saving."""

    # Define models
    rgb_net = ResNetFeatureExtractor(input_channels=3).to(device)
    landmark_net = ResNetFeatureExtractor(input_channels=3).to(device)
    projection_rgb = ProjectionHead(256, 128).to(device)
    projection_landmark = ProjectionHead(256, 128).to(device)

    # Load pretrained models
    pretrain_path = os.path.join(config['save_dir'], config['exp_name'], 'pretrain', 'best_model.pth')
    if not os.path.exists(pretrain_path):
        raise FileNotFoundError(f"Pretrained model not found at {pretrain_path}")
    checkpoint = torch.load(pretrain_path, map_location=device)
    rgb_net.load_state_dict(checkpoint['rgb_net_state_dict'])
    landmark_net.load_state_dict(checkpoint['landmark_net_state_dict'])
    projection_rgb.load_state_dict(checkpoint['projection_rgb_state_dict'])
    projection_landmark.load_state_dict(checkpoint['projection_landmark_state_dict'])
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
        list(landmark_net.parameters()) +
        list(projection_rgb.parameters()) +
        list(projection_landmark.parameters()) +
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
        landmark_net.train()
        projection_rgb.train()
        projection_landmark.train()
        classifier.train()

        epoch_loss = 0.0
        for rgb_images, _, landmark_images, _, labels in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{finetune_epochs}]"):
            # Move data to GPU
            rgb_images = rgb_images.to(device)
            landmark_images = landmark_images.to(device)
            labels = labels.to(device)
            
            # _ learning with augmented images (RGB)
            rgb_features = rgb_net(rgb_images)
            rgb_projection = projection_rgb(rgb_features)

            # Contrastive learning with augmented images (landmark)
            landmark_features = landmark_net(landmark_images)
            landmark_projection = projection_landmark(landmark_features)
            
            # Concatenate original features and projection features
            combined_features = torch.cat((rgb_features, landmark_features, rgb_projection, landmark_projection), dim=1)

            # 通过分类器进行分类
            outputs = classifier(combined_features)

            cls_loss = criterion(outputs, labels)
            loss = cls_loss
            epoch_loss = cls_loss.item()

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f'Finetune Epoch [{epoch+1}/{finetune_epochs}], Average Loss: {epoch_loss / len(train_dataloader)}')


        # Evaluation phase
        rgb_net.eval()
        landmark_net.eval()
        projection_rgb.eval()
        projection_landmark.eval()
        classifier.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for rgb_images, _, landmark_images, _, label_y  in tqdm(test_dataloader, desc=f"Evaluating Epoch [{epoch+1}/{finetune_epochs}]"):
                rgb_images = rgb_images.to(device)
                landmark_images = landmark_images.to(device)
                label_y = label_y.to(device)

                # Forward pass
                rgb_features = rgb_net(rgb_images)
                landmark_features = landmark_net(landmark_images)
                rgb_projection = projection_rgb(rgb_features)
                landmark_projection = projection_landmark(landmark_features)
                
                # Concatenate original features and projection features
                combined_features = torch.cat((rgb_features, landmark_features, rgb_projection, landmark_projection), dim=1)

                # Classification
                outputs = classifier(combined_features)
                _, predicted = torch.max(outputs.data, 1)
                total += label_y.size(0)
                correct += (predicted == label_y).sum().item()

        accuracy = 100 * correct / total
        print(f'Test Accuracy after Finetune Epoch [{epoch+1}/{finetune_epochs}]: {accuracy:.2f}%')

        # Save the latest model
        latest_model_path = os.path.join(experiment_dir, 'latest_model.pth')
        torch.save({
            'epoch': epoch + 1,
            'rgb_net_state_dict': rgb_net.state_dict(),
            'landmark_net_state_dict': landmark_net.state_dict(),
            'projection_rgb_state_dict': projection_rgb.state_dict(),
            'projection_landmark_state_dict': projection_landmark.state_dict(),
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
                'landmark_net_state_dict': landmark_net.state_dict(),
                'projection_rgb_state_dict': projection_rgb.state_dict(),
                'projection_landmark_state_dict': projection_landmark.state_dict(),
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
    rgb_root = config['train_dataset_path']
    landmarks_root= config['train_landmarks_dataset_path']
    
    train_dataset = MultiModalDataset_Landmark(rgb_root, landmarks_root)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['finetune_batch_size'], 
        shuffle=True, 
        num_workers=config['finetune_num_workers']
    )
    
    rgb_root = config['test_dataset_path']
    landmarks_root= config['test_landmarks_dataset_path']
    test_dataset = MultiModalDataset_Landmark(rgb_root, landmarks_root)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['test_batch_size'],
        shuffle=False,
        num_workers=config['test_num_workers']
    )

    # Fine-tune with model saving
    simclip_landmark_finetune(
        train_dataloader,
        test_dataloader,
        config,
        device
    )
