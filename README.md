# **SimCLIP Training and Fine-Tuning Framework**

This README provides step-by-step instructions for pretraining and fine-tuning SimCLIP models. It includes details on the configuration file, commands to run pretraining and fine-tuning, parameter adjustments, and troubleshooting tips.

---

## **Table of Contents**
1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Configuration File](#configuration-file)
4. [Pretraining](#pretraining)
5. [Fine-Tuning](#fine-tuning)
6. [Parameter Adjustments](#parameter-adjustments)
7. [Notes and Troubleshooting](#notes-and-troubleshooting)

---

## **Prerequisites**
Before using this framework, ensure the following:
- Python 3.8 or later
- GPU with CUDA 11.0 or later
- Required Python libraries installed:
```bash
  pip install -r requirements.txt
```
 
- Dataset prepared in the specified format: 
  - Training Dataset: Located at `./affectnet/train`
 
  - Testing Dataset: Located at `./affectnet/test`

---

## **Project Structure** 
The directory structure for the project should look like this:


```bash
├── config.yml                     # Configuration file
├── pretrain.py                    # Script for pretraining
├── finetune.py                    # Script for fine-tuning
├── test.py                        # Script for testing and evaluation
├── simclip_models.py              # Core model definitions
├── simclip_utils.py               # Utility functions
├── requirements.txt               # Required Python libraries
├── saved_models/                  # Directory for saving model checkpoints
│   ├── simclip_rn18/              # Experiment name
│       ├── pretrain/              # Pretraining checkpoints
│       ├── finetune/              # Fine-tuning checkpoints
```


---

## **Configuration File** The `config.yml` file contains all the parameters for training, fine-tuning, and testing. Below is an example configuration:

```yaml
# Dataset and Dataloader configuration
save_dir: './saved_models'         # Directory to save the models
exp_name: simclip_rn18             # Experiment name
train_dataset_path: './affectnet_3750subset/train'  # Path to train dataset
test_dataset_path: './affectnet_3750subset/test'    # Path to test dataset

# Pretrain configuration
pretrain_batch_size: 64            # Batch size for pretraining
pretrain_num_workers: 4            # Number of workers for DataLoader
pretrain_epochs: 30                # Total pretraining epochs
pretrain_learning_rate: 0.001      # Learning rate for optimizer

# Fine-tune configuration
finetune_batch_size: 64            # Batch size for fine-tuning
finetune_num_workers: 4            # Number of workers for DataLoader
finetune_epochs: 30                # Total fine-tuning epochs
finetune_learning_rate: 0.001      # Learning rate for optimizer

# Test configuration
test_batch_size: 64                # Batch size for testing
test_num_workers: 4                # Number of workers for DataLoader
```


---

**Pretraining** 
To train the SimCLIP model with contrastive learning, run the following command:


```bash
python simclip_pretrain.py --config ./config.yml
```
**Outputs** Checkpoints will be saved in `./saved_models/<exp_name>/pretrain/`: 
- `best_model.pth`: Model with the lowest validation loss
 
- `latest_model.pth`: Most recent checkpoint


---

**Fine-Tuning** 
To fine-tune the pretrained SimCLIP model for classification tasks, run:


```bash
python simclip_finetune.py --config ./config.yml
```
**Outputs** Checkpoints will be saved in `./saved_models/<exp_name>/finetune/`: 
- `best_model.pth`: Model with the highest classification accuracy
 
- `latest_model.pth`: Most recent checkpoint


---

**Outputs** 
Evaluation metrics and predictions will be logged in the terminal or saved in the specified directory.

---

**Parameter Adjustments** Modify the `config.yml` file to adjust training parameters. Common adjustments include: 
- **Batch Size:**  `pretrain_batch_size`, `finetune_batch_size`, `test_batch_size`
 
- **Learning Rate:**  `pretrain_learning_rate`, `finetune_learning_rate`
 
- **Epochs:**  `pretrain_epochs`, `finetune_epochs`

---

