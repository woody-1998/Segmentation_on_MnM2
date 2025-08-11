import torch
import copy
import pandas as pd
import numpy as np
import os
import logging
from joblib import dump
from torch import nn
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

import sys
sys.path.append('../')
from sympy.core.random import choice
from Dataloader.utils.dataset import CardiacMRIDataset
from Dataloader.utils.augmentation import get_eval_transform, get_train_transform
from Models.unet import BaseNet
from Models.metrics import dice_coef, iou_coef
from torch.optim.lr_scheduler import ReduceLROnPlateau


# input parameters
def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--model',
                        choices=['unet', 'fpn'], default='unet',
                        help='Choose the model to train')
    parser.add_argument('--encoder',
                        choices=['resnet50', 'efficientnet-b0'], default='resnet50',)
    parser.add_argument('--device',
                        choices=['cpu', 'cuda', 'mps'], default='mps',
                        help='Choose the device to train')
    parser.add_argument('--model_saving',
                        choices=['all', 'optimal'], default='optimal',
                        help='Choose which strategy to save the model')
    parser.add_argument('--loss_fn',
                        choices=['dice', 'jaccard', 'dice+BCE'], default='dice')
    parser.add_argument('--dataset_path', type=str, default="../Dataset/MnM2_preprocessed_2Dslices")
    parser.add_argument('--checkpoint_path', type=str, default="resnet_checkpoints")
    parser.add_argument('--output_path', type=str, default="resnet_training")

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

def get_loaders(base_path, batch_size):
    path_train_images_2d = os.path.join(base_path, 'train', 'images')
    path_train_masks_2d = os.path.join(base_path, 'train', 'masks')

    path_val_images_2d = os.path.join(base_path, 'val', 'images')
    path_val_masks_2d = os.path.join(base_path, 'val', 'masks')

    path_test_images_2d = os.path.join(base_path, 'test', 'images')
    path_test_masks_2d = os.path.join(base_path, 'test', 'masks')

    try:
        os.path.exists(path_train_images_2d)
        os.path.exists(path_train_masks_2d)
        os.path.exists(path_val_images_2d)
        os.path.exists(path_val_masks_2d)
        os.path.exists(path_test_images_2d)
        os.path.exists(path_test_masks_2d)
    except:
        print("at least one of input folder does not exist")

    set_seed()

    # get the datasets
    tra_dataset = CardiacMRIDataset(
    image_dir = path_train_images_2d,
    mask_dir = path_train_masks_2d,
    transform = get_train_transform()
)
    val_dataset = CardiacMRIDataset(
        image_dir = path_val_images_2d,
        mask_dir = path_val_masks_2d,
        transform = get_eval_transform()
    )
    test_dataset = CardiacMRIDataset(
        image_dir = path_test_images_2d,
        mask_dir = path_test_masks_2d,
        transform = get_eval_transform()
    )

    train_loader = DataLoader(tra_dataset, batch_size= batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=True)

    return train_loader, val_loader, test_loader

def visualise_loss(history, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Losses")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "train_val_losses.png"))
    plt.close()

def visualise_metrics(history, output_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(history["val_dice"], label="Validation Dice Score")
    plt.plot(history["val_iou"], label="Validation IoU Score")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Dice and IoU Score")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "val_metrics.png"))
    plt.close()


# training function, train the model, save training and evaluate results
def main():
    # get all inputs
    args = argparser()

    # set the logging config
    os.makedirs(args.output_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%H:%M:%S",
        filename= os.path.join(args.output_path, "training.log")
    )

    # ==== Configuration ====
    num_epochs = args.epochs
    learning_rate = args.lr
    checkpoint_dir = args.checkpoint_path
    os.makedirs(checkpoint_dir, exist_ok=True)
    start_epoch = 0
    device = torch.device(args.device)
    logging.info(f"device: {device}, num_epochs: {num_epochs}, learning_rate: {learning_rate}")

    # model saving tactic
    model_saving = args.model_saving

    # get the loss function choice
    lc = args.loss_fn
    if lc =='dice':
        loss_fn = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
    elif lc =='jaccard':
        loss_fn = smp.losses.JaccardLoss(mode=smp.losses.BINARY_MODE, from_logits=True)
    loss_bce = nn.BCEWithLogitsLoss()
    logging.info(f"loss function: {lc}")

    # get the model according to needs
    print("******getting the model********")
    model = BaseNet(args.model, args.encoder)

    print("******model ready********")
    # covert the model to the device
    model = model.to(device)
    # if device.type == 'mps':
    #     print("Device is MPS")

    # ==== DataLoader ====
    base_path = args.dataset_path
    train_loader, val_loader, test_loader = get_loaders(base_path, batch_size=args.batch_size)
    print("******dataloader ready********")

    # ==== Optimizer, loss, scheduler ====
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # ==== Metric trackers ====
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}

    # ==== checkpoint if available ====
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    logging.info("strat training")
    # ==== Training Loop ====
    for epoch in range(start_epoch, num_epochs):

        logging.info(f"epoch {epoch}/{num_epochs}")

        # Wrap dataloader with tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        # Training phase
        model.train()
        train_loss = 0.0
        count = 0
        for imgs, masks in pbar:
            imgs, masks = imgs.to(device), masks.to(device).float()
            optimizer.zero_grad()
            # get the outputs
            preds = model(imgs)
            loss = loss_fn(preds, masks) + loss_bce(preds, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # count += 1
            # if count >= 10:
            #     break


        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device).float()
                preds = model(imgs)
                loss = loss_fn(preds, masks) + loss_bce(preds, masks)
                val_loss += loss.item()

                preds_sigmoid = torch.sigmoid(preds)
                val_dice += dice_coef(preds_sigmoid, masks).item()
                val_iou += iou_coef(preds_sigmoid, masks).item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_iou = val_iou / len(val_loader)

        # Logging
        logging.info(
         f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Dice: {avg_val_dice:.4f}"
         f" | Val IoU: {avg_val_iou:.4f}")
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_dice'].append(avg_val_dice)
        history['val_iou'].append(avg_val_iou)

        # LR Scheduler step
        scheduler.step(avg_val_loss)
        # Save best model checkpoint (based on val loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, os.path.join(checkpoint_dir, "best_basenet.pth"))
            logging.info("best model saved")

    logging.info("training finished")
    visualise_loss(history, args.output_path)
    visualise_metrics(history, args.output_path)

    # ======= Load Best Model for Evaluation =======
    model.load_state_dict(best_model_wts)
    logging.info("evaluating starts")
    model.eval()

    test_dice, test_iou = 0.0, 0.0
    shown = 0 # pictures shown

    with torch.no_grad():
        for imgs, masks in tqdm(test_loader, desc="Testing"):
            imgs, masks = imgs.to(device), masks.to(device).float()
            preds = model(imgs)
            preds_sigmoid = torch.sigmoid(preds)
            preds_bin = (preds_sigmoid > 0.5).float()

            # plot images with gt
            if shown <= 5:
            # Filter non-zero mask indices in the batch
                non_zero_indices = [i for i in range(len(masks)) if torch.sum(masks[i]) > 0]

                if not non_zero_indices:
                    continue  # skip this batch

                for i in non_zero_indices:
                    if shown >= 5:
                        break

                    plt.figure(figsize=(12, 4))
                    plt.subplot(1, 3, 1)
                    plt.imshow(imgs[i][0].cpu(), cmap='gray')
                    plt.title("Image");
                    plt.axis('off')

                    plt.subplot(1, 3, 2)
                    plt.imshow(masks[i][0].cpu(), cmap='gray')
                    plt.title("Ground Truth");
                    plt.axis('off')

                    plt.subplot(1, 3, 3)
                    plt.imshow(preds_bin[i][0].cpu(), cmap='gray')
                    plt.title("Prediction");
                    plt.axis('off')

                    plt.savefig(os.path.join(args.output_path, f"evaluation{shown}.png"))
                    plt.close()
                    shown += 1


            # Metrics
            test_dice += dice_coef(preds_sigmoid, masks).item()
            test_iou += iou_coef(preds_sigmoid, masks).item()

    avg_test_dice = test_dice / len(test_loader)
    avg_test_iou = test_iou / len(test_loader)

    logging.info(f"\n✅ Test Dice: {avg_test_dice:.4f} | Test IoU: {avg_test_iou:.4f}")



if __name__ == '__main__':
    main()



