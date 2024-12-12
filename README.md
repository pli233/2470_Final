# **SimCLIP Training and Fine-Tuning Framework**

This is our realization of pretraining and fine-tuning SimCLIP models. The overview of simclip is like below:
![alt text](<simclip method.png>)
---

## **Table of Contents**
1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Configuration File](#configuration-file)
4. [Generating Landmarks](#generating-landmarks)
5. [Pretraining](#pretraining)
6. [Fine-Tuning](#fine-tuning)


---

## **Prerequisites**
Before using this framework, use the command in env_setup.txt to get ready for env


- Dataset prepared in the specified format: 
  - Training Dataset: Located at `./affectnet/train`
  - Testing Dataset: Located at `./affectnet/test`
---

## **Project Structure** 
The directory structure for the project should look like this:

```bash
├── config.yml                     # Project Configuration
├── simclip_gray_pretrain.py       # Script for gray-scale pretraining
├── simclip_gray_finetune.py       # Script for gray-scale fine-tuning
├── simclip_generate_landmark.py   # Script to generate landmarks
├── simclip_landmark_pretrain.py   # Script for landmark-based pretraining
├── simclip_landmark_finetune.py   # Script for landmark-based fine-tuning
├── simclip_models.py              # Core model definitions
├── simclip_utils.py               # Utility functions
├── env_setup.txt                  # Command to setup environment
├── saved_models/                  # Directory for saving model checkpoints
│   ├── simclip_rn18_gray/         # Gray-scale experiment directory
│   ├── simclip_rn18_landmark/     # Landmark-based experiment directory
```

---

## **Configuration File**
The `config.yml` file contains all the parameters for training, fine-tuning, and testing. Below is an example configuration:

```yaml
# Dataset and Dataloader configuration
save_dir: './saved_models'         # Directory to save the models
exp_name: simclip_rn18_gray        # Experiment name for gray-scale
train_dataset_path: './affectnet/train'  # Path to train dataset
test_dataset_path: './affectnet/test'    # Path to test dataset
train_landmarks_dataset_path: './affectnet_landmark/train'  # Path to train landmark dataset
test_landmarks_dataset_path: './affectnet_landmark/test'    # Path to test landmark dataset

# Pretrain configuration
pretrain_batch_size: 32            # Batch size for pretraining
pretrain_num_workers: 8            # Number of workers for DataLoader
pretrain_epochs: 30                # Total pretraining epochs
pretrain_learning_rate: 0.003      # Learning rate for optimizer

# Fine-tune configuration
finetune_batch_size: 32            # Batch size for fine-tuning
finetune_num_workers: 8            # Number of workers for DataLoader
finetune_epochs: 30                # Total fine-tuning epochs
finetune_learning_rate: 0.003      # Learning rate for optimizer

# Test configuration
test_batch_size: 64                # Batch size for testing
test_num_workers: 8                # Number of workers for DataLoader
```

---

## **Generating Landmarks**
Before performing landmark-based pretraining, generate the landmark files using the following command:
```bash
python simclip_generate_landmark.py
```
This script will process the images in `train_dataset_path` and save the landmarks to `train_landmarks_dataset_path`. Similarly, it will process the test dataset for landmarks.

**Outputs**:
- Generated landmark images will be saved in the `train_landmarks_dataset_path` and `test_landmarks_dataset_path` directories.

---

## **Pretraining**
### **Gray-Scale Pretraining**
To train the SimCLIP model with gray-scale images, run the following command:
```bash
python simclip_gray_pretrain.py
```

### **Landmark-Based Pretraining**
To train the SimCLIP model with landmarks, run the following command:
```bash
python simclip_landmark_pretrain.py
```

**Outputs**:
- Checkpoints will be saved in `./saved_models/<exp_name>/pretrain/`:
  - `best_model.pth`: Model with the lowest validation loss.
  - `latest_model.pth`: Most recent checkpoint.

---

## **Fine-Tuning**
### **Gray-Scale Fine-Tuning**
To fine-tune the pretrained gray-scale SimCLIP model for classification tasks, run:
```bash
python simclip_gray_finetune.py
```

### **Landmark-Based Fine-Tuning**
To fine-tune the pretrained landmark-based SimCLIP model for classification tasks, run:
```bash
python simclip_landmark_finetune.py
```

**Outputs**:
- Checkpoints will be saved in `./saved_models/<exp_name>/finetune/`:
  - `best_model.pth`: Model with the highest classification accuracy.
  - `latest_model.pth`: Most recent checkpoint.

---