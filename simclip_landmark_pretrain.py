import os
import torch
import torch.optim as optim
from tqdm import tqdm
from simclip_models import MultiModalDataset_Landmark, ResNetFeatureExtractor, ProjectionHead, nt_xent_loss, contrastive_loss
from simclip_utils import load_config
from torch.utils.data import DataLoader

# --- Pre-Training with Model Saving ---
def simclip_landmark_pretrain(train_dataloader, config, device):
    """Pre-training with contrastive learning and model saving."""
    
    # Define models
    rgb_net = ResNetFeatureExtractor(input_channels=3).to(device)
    landmark_net = ResNetFeatureExtractor(input_channels=3).to(device)
    projection_rgb = ProjectionHead(256, 128).to(device)
    projection_landmark = ProjectionHead(256, 128).to(device)
    
    rgb_net.train()
    landmark_net.train()
    projection_rgb.train()
    projection_landmark.train()
        
    optimizer = optim.Adam(
        list(rgb_net.parameters()) + list(landmark_net.parameters()) +
        list(projection_rgb.parameters()) + list(projection_landmark.parameters()), 
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
        for rgb_images, rgb_augmented_images, landmark_images, landmark_augmented_images, label in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            # Move data to GPU
            rgb_images = rgb_images.to(device)
            rgb_augmented_images = rgb_augmented_images.to(device)
            landmark_images = landmark_images.to(device)
            landmark_augmented_images = landmark_augmented_images.to(device)

            # _ learning with augmented images (RGB)
            rgb_features = rgb_net(rgb_images)
            rgb_features_aug = rgb_net(rgb_augmented_images)
            rgb_projection = projection_rgb(rgb_features)
            rgb_projection_aug= projection_rgb(rgb_features_aug)
            loss_rgb = nt_xent_loss(rgb_projection_aug, rgb_projection)

            # Contrastive learning with augmented images (landmark)
            landmark_features = landmark_net(landmark_images)
            landmark_features_aug = landmark_net(landmark_augmented_images)
            landmark_projection = projection_landmark(landmark_features)
            landmark_projection_aug = projection_landmark(landmark_features_aug)
            loss_landmark = nt_xent_loss(landmark_projection_aug, landmark_projection)

            # Aligning RGB and landmark images through contrastive learning
            loss_rgb_landmark = contrastive_loss(rgb_projection, landmark_projection)
            loss_rgb_landmark_aug = contrastive_loss(rgb_projection_aug, landmark_projection_aug)
            loss = loss_rgb + loss_landmark + loss_rgb_landmark + loss_rgb_landmark_aug
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
            'landmark_net_state_dict': landmark_net.state_dict(),
            'projection_rgb_state_dict': projection_rgb.state_dict(),
            'projection_landmark_state_dict': projection_landmark.state_dict(),
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
                'landmark_net_state_dict': landmark_net.state_dict(),
                'projection_rgb_state_dict': projection_rgb.state_dict(),
                'projection_landmark_state_dict': projection_landmark.state_dict(),
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
    rgb_root= config['train_dataset_path']
    landmarks_root= config['train_landmarks_dataset_path']
    
    train_dataset = MultiModalDataset_Landmark(rgb_root, landmarks_root)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['pretrain_batch_size'], 
        shuffle=True, 
        num_workers=config['pretrain_num_workers']
    )


    # Train with model saving
    simclip_landmark_pretrain(
        train_dataloader,
        config,
        device
    )