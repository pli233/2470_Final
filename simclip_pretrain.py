import os
import torch
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
from simclip_models import MultiModalDataset_Train, ResNetFeatureExtractor, ProjectionHead, nt_xent_loss, contrastive_loss
from torch.utils.data import DataLoader
import yaml


# --- Load Configuration ---
def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# --- Pre-Training with Model Saving ---
def simclip_pretrain(train_dataloader, config, device):
    """Pre-training with contrastive learning and model saving."""
    
    
    # Define models
    rgb_net = ResNetFeatureExtractor(input_channels=3).to(device)
    depth_net = ResNetFeatureExtractor(input_channels=1).to(device)
    projection_rgb = ProjectionHead(256, 128).to(device)
    projection_depth = ProjectionHead(256, 128).to(device)
    
    rgb_net.train()
    depth_net.train()
    projection_rgb.train()
    projection_depth.train()
        
    optimizer = optim.Adam(
        list(rgb_net.parameters()) + list(depth_net.parameters()) +
        list(projection_rgb.parameters()) + list(projection_depth.parameters()), 
        lr=config['pretrain_learning_rate']
    )
    
    # Create save directory for pretraining
    experiment_dir = os.path.join(config['save_dir'], config['exp_name'], 'pretrain')
    os.makedirs(experiment_dir, exist_ok=True)
    best_loss = float('inf')
    best_model_path = os.path.join(experiment_dir, 'best_model.pth')
    latest_model_path = os.path.join(experiment_dir, 'latest_model.pth')
    epochs = int(config['pretrain_epochs'])
    
    for epoch in range(epochs):
        epoch_loss = 0
        for rgb_images, rgb_augmented_images, depth_images, depth_augmented_images, labelq in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            # Move data to GPU
            rgb_images = rgb_images.to(device)
            rgb_augmented_images = rgb_augmented_images.to(device)
            depth_images = depth_images.to(device)
            depth_augmented_images = depth_augmented_images.to(device)

            # _ learning with augmented images (RGB)
            rgb_features_1 = rgb_net(rgb_augmented_images)
            rgb_features_2 = rgb_net(rgb_images)
            rgb_projection_1 = projection_rgb(rgb_features_1)
            rgb_projection_2 = projection_rgb(rgb_features_2)
            loss_rgb = nt_xent_loss(rgb_projection_1, rgb_projection_2)

            # Contrastive learning with augmented images (Depth)
            depth_features_1 = depth_net(depth_augmented_images)
            depth_features_2 = depth_net(depth_images)
            depth_projection_1 = projection_depth(depth_features_1)
            depth_projection_2 = projection_depth(depth_features_2)
            loss_depth = nt_xent_loss(depth_projection_1, depth_projection_2)

            # Aligning RGB and depth images through contrastive learning
            rgb_features = rgb_net(rgb_images)
            depth_features = depth_net(depth_images)
            rgb_projection = projection_rgb(rgb_features)
            depth_projection = projection_depth(depth_features)
            loss_rgb_depth = contrastive_loss(rgb_projection, depth_projection)

            loss = loss_rgb + loss_depth + loss_rgb_depth
            epoch_loss += loss.item()

            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Calculate average loss
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch [{epoch + 1}/{config['pretrain_epochs']}], Average Loss: {avg_loss:.4f}")

        # Save the latest model
        torch.save({
            'epoch': epoch + 1,
            'rgb_net_state_dict': rgb_net.state_dict(),
            'depth_net_state_dict': depth_net.state_dict(),
            'projection_rgb_state_dict': projection_rgb.state_dict(),
            'projection_depth_state_dict': projection_depth.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'avg_loss': avg_loss,
        }, latest_model_path)
        print(f"Latest model saved to {latest_model_path}")

        # Save the best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch + 1,
                'rgb_net_state_dict': rgb_net.state_dict(),
                'depth_net_state_dict': depth_net.state_dict(),
                'projection_rgb_state_dict': projection_rgb.state_dict(),
                'projection_depth_state_dict': projection_depth.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_loss': best_loss,
            }, best_model_path)
            print(f"New best model saved with loss {best_loss:.4f} to {best_model_path}")

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
    # Load dataset
    train_dataset = MultiModalDataset_Train(config['train_dataset_path'])
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['pretrain_batch_size'], 
        shuffle=True, 
        num_workers=config['pretrain_num_workers']
    )


    # Train with model saving
    simclip_pretrain(
        train_dataloader,
        config,
        device
    )
