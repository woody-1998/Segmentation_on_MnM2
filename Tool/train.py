import sys
sys.path.append('../')

import torch
import pandas as pd
import numpy as np
import os
from joblib import dump
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
import Model.AlexNet as AlexNet
import argparse
from torch.utils.data import DataLoader
from monai.networks.nets.resnet import ResNet, get_inplanes
from Model.ResNet18CBAM import ResNet18WithCBAM

from sympy.core.random import choice
from Dataloader.utils.dataset import MRNetDataset
from Dataloader.utils.transforms import get_mrnet_train_transforms, get_mrnet_valid_transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Model.utils.checkpoint import save_checkpoint, load_checkpoint


# input parameters
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model',
                        choices=['alexnet', 'resnet18', 'resnet18CBAM'], default='alexnet',
                        help='Choose the model to train')
    parser.add_argument('--device',
                        choices=['cpu', 'cuda', 'mps'], default='cuda',
                        help='Choose the device to train')
    parser.add_argument('--model_saving',
                        choices=['all', 'optimal'], default='optimal',
                        help='Choose which strategy to save the model')
    parser.add_argument('--dataset_path', type=str, default="../Dataset/MRNet-v1.0",)
    parser.add_argument('--checkpoint_path', type=str, default="alexnet_checkpoints")
    parser.add_argument('--slices', type=int, default=32)
    # get the input
    args = parser.parse_args()
    return args

# Define a function to set seeds for reproducibility
def set_seed(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

# Define function to load data and summarise labels
def load_data(base_path, label_type: str, data_split: str):
    """
    Loads a label CSV file and renames its columns for clarity.

    Args:
        label_type (str): one of ["abnormal", "acl", "meniscus"]
        data_split (str): "train" or "valid"
    Returns:
        pd.DataFrame: processed dataframe with columns ["Series", label_type]
    """
    filename = f"{data_split}-{label_type}.csv"
    df = pd.read_csv(os.path.join(base_path, filename), header=None)
    df.columns = ["Series", label_type]

    # Pad Series column with leading zeros to 4 digits
    df["Series"] = df["Series"].astype(str).str.zfill(4)

    return df

# get the dataloader and transformation
def dataloader(base_path, slices):

    # ===== get and merge train and valid dataset of three classes
    df_train_abnormal = load_data(base_path, "abnormal", "train")
    df_train_acl = load_data(base_path, "acl", "train")
    df_train_meniscus = load_data(base_path, "meniscus", "train")

    df_valid_abnormal = load_data(base_path, "abnormal", "valid")
    df_valid_acl = load_data(base_path, "acl", "valid")
    df_valid_meniscus = load_data(base_path, "meniscus", "valid")

    # Merge into one training dataframe
    df_train = df_train_abnormal.merge(df_train_acl, on="Series").merge(df_train_meniscus, on="Series")

    # Merge into one validation dataframe
    df_valid = df_valid_abnormal.merge(df_valid_acl, on="Series").merge(df_valid_meniscus, on="Series")


    # Set seed for reproducibility
    set_seed()

    # Define transform pipelines for training and validation
    train_transform = get_mrnet_train_transforms(target_slices=slices)
    valid_transform = get_mrnet_valid_transforms(target_slices=slices)

    # Create training dataset
    train_dataset = MRNetDataset(
        data_dir=os.path.join(base_path, "train"),
        df_labels=df_train,
        transform=train_transform
    )

    # Create validation dataset
    valid_dataset = MRNetDataset(
        data_dir=os.path.join(base_path, "valid"),
        df_labels=df_valid,
        transform=valid_transform
    )

    # Create DataLoader for training
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Create DataLoader for validation
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)

    return train_loader, valid_loader

def visualise(tr_list, va_list, tag):
    plt.plot(tr_list, label='Train ' + tag)
    plt.plot(va_list, label='Val ' + tag)
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig(tag + ".png")

# training function, train the model, save training and evaluate results
def main():
    # get all inputs
    args = argparser()

    # ==== Configuration ====
    num_epochs = args.epochs
    patience = 5
    learning_rate = args.lr
    checkpoint_dir = args.checkpoint_path
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = 0
    device = torch.device(args.device)

    # model saving tactic
    model_saving = args.model_saving

    # get the model according to needs
    print("******getting the model********")
    model_choice = args.model
    if model_choice == 'alexnet':
        model = AlexNet.AlexNetSlice()
    elif model_choice == 'resnet18':
        # Get default block_inplanes (usually [64, 128, 256, 512])
        block_inplanes = get_inplanes()

        # Define model
        model = ResNet(
            block='basic',  # ResNet18 style
            layers=(2, 2, 2, 2),  # ResNet18 config
            block_inplanes=block_inplanes,
            spatial_dims=3,
            n_input_channels=3,
            num_classes=3
        )
    elif model_choice == 'resnet18CBAM':
        model = ResNet18WithCBAM()

    print("******model ready********")
    # covert the model to the device
    model = model.to(device)
    # if device.type == 'mps':
    #     print("Device is MPS")

    # ==== DataLoader ====
    base_path = args.dataset_path
    slices = args.slices
    train_loader, valid_loader = dataloader(base_path, slices)
    print("******dataloader ready********")

    # ==== Optimizer, loss, scheduler ====
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    # ==== Metric trackers ====
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    # ==== Resume from checkpoint if available ====
    last_ckpt_path = os.path.join(checkpoint_dir, 'last_checkpoint.pth')
    if os.path.exists(last_ckpt_path):
        start_epoch, _ = load_checkpoint(
            last_ckpt_path, model, optimizer, scheduler, map_location=device
        )
        print(f"🔁 Resumed training from epoch {start_epoch + 1}")

    # ==== Training Loop ====
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        print("Training starts")

        # Wrap dataloader with tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (preds == y).float().mean().item()
            train_total += 1

        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / train_total
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)

        # ==== Validation ====
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x_val, y_val in valid_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_outputs = model(x_val)
                val_loss += criterion(val_outputs, y_val).item() * x_val.size(0)

                preds_val = (torch.sigmoid(val_outputs) > 0.5).float()
                val_correct += (preds_val == y_val).float().mean().item()
                val_total += 1

        val_loss /= len(valid_loader.dataset)
        val_acc = val_correct / val_total
        val_loss_list.append(val_loss)
        val_acc_list.append(val_acc)

        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

        # ✅ Save checkpoint after every epoch
        if model_saving == 'all':
            ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1:02d}.pth')
            save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, val_loss)
            print(f"📦 all Checkpoint saved at {ckpt_path}")
        elif model_saving == 'optimal':
            if train_acc >= max(train_acc_list):
                ckpt_path = os.path.join(checkpoint_dir, f'epoch_{epoch + 1:02d}.pth')
                save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch, val_loss)
                print(f"📦 best Checkpoint saved at {ckpt_path}")
            else:
                pass

        if scheduler.num_bad_epochs >= patience:
            print("⏹️ Early stopping triggered due to no improvement.")
            break

    visualise(train_acc_list, val_acc_list, tag="accuracy")
    visualise(train_loss_list, val_loss_list, tag="loss")


if __name__ == '__main__':
    main()



