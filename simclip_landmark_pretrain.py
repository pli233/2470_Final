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
    landrmark_net = ResNetFeatureExtractor(input_channels=3).to(device)
    projection_rgb = ProjectionHead(256, 128).to(device)
    projection_landrmark = ProjectionHead(256, 128).to(device)
    
    rgb_net.train()
    landrmark_net.train()
    projection_rgb.train()
    projection_landrmark.train()
        
    optimizer = optim.Adam(
        list(rgb_net.parameters()) + list(landrmark_net.parameters()) +
        list(projection_rgb.parameters()) + list(projection_landrmark.parameters()), 
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
        for rgb_images, rgb_augmented_images, landrmark_images, landrmark_augmented_images, label in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{epochs}]"):
            # Move data to GPU
            rgb_images = rgb_images.to(device)
            rgb_augmented_images = rgb_augmented_images.to(device)
            landrmark_images = landrmark_images.to(device)
            landrmark_augmented_images = landrmark_augmented_images.to(device)

            # _ learning with augmented images (RGB)
            rgb_features_1 = rgb_net(rgb_augmented_images)
            rgb_features_2 = rgb_net(rgb_images)
            rgb_projection_1 = projection_rgb(rgb_features_1)
            rgb_projection_2 = projection_rgb(rgb_features_2)
            loss_rgb = nt_xent_loss(rgb_projection_1, rgb_projection_2)

            # Contrastive learning with augmented images (landrmark)
            landrmark_features_1 = landrmark_net(landrmark_augmented_images)
            landrmark_features_2 = landrmark_net(landrmark_images)
            landrmark_projection_1 = projection_landrmark(landrmark_features_1)
            landrmark_projection_2 = projection_landrmark(landrmark_features_2)
            loss_landrmark = nt_xent_loss(landrmark_projection_1, landrmark_projection_2)

            # Aligning RGB and landrmark images through contrastive learning
            rgb_features = rgb_net(rgb_images)
            landrmark_features = landrmark_net(landrmark_images)
            rgb_projection = projection_rgb(rgb_features)
            landrmark_projection = projection_landrmark(landrmark_features)
            loss_rgb_landrmark = contrastive_loss(rgb_projection, landrmark_projection)

            loss = loss_rgb + loss_landrmark + loss_rgb_landrmark
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
            'landrmark_net_state_dict': landrmark_net.state_dict(),
            'projection_rgb_state_dict': projection_rgb.state_dict(),
            'projection_landrmark_state_dict': projection_landrmark.state_dict(),
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
                'landrmark_net_state_dict': landrmark_net.state_dict(),
                'projection_rgb_state_dict': projection_rgb.state_dict(),
                'projection_landrmark_state_dict': projection_landrmark.state_dict(),
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
    landmarks_root= config['landmarks_output_path']
    
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
