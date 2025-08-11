import sys
from random import choice

sys.path.append('../')

from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
import argparse
import os

from Dataloader.utils.dataset import  SDiseaseDataset
from Dataloader.utils.augmentation import get_eval_transform
from Models.unet import BaseNet
from Models.metrics import dice_coef, iou_coef
import torch



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        choices=['unet', 'fpn', 'pan', 'deeplabv3'], default='unet')
    parser.add_argument('--encoder',
                        choices=['resnet50', 'efficientnet-b0'], default='resnet50')
    parser.add_argument('--checkpoints_dir', type=str, default='resenet_checkpoints')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device',
                        choices=['cpu', 'cuda', 'mps'], default='mps',
                        help='Choose the device to train')

    args = parser.parse_args()
    return args

def eval(test_dataloader, model, device):
    model.to(device)
    model.eval()

    test_dice, test_iou = 0.0, 0.0

    with torch.no_grad():
        for imgs, masks in tqdm(test_dataloader, desc="Testing"):

            imgs, masks = imgs.to(device), masks.to(device).float()
            preds = model(imgs)
            preds_sigmoid = torch.sigmoid(preds)

            # Metrics
            test_dice += dice_coef(preds_sigmoid, masks).item()
            test_iou += iou_coef(preds_sigmoid, masks).item()

        avg_test_dice = test_dice / len(test_dataloader)
        avg_test_iou = test_iou / len(test_dataloader)

    return avg_test_dice, avg_test_iou


def main():
    args = parse_args()
    path_test_images_2d = "../Dataset/MnM2_preprocessed_2Dslices/test/images"
    path_test_masks_2d = "../Dataset/MnM2_preprocessed_2Dslices/test/masks"
    info_csv_path = "../Dataset/MnM2/dataset_information.csv"

    logging.basicConfig(level=logging.INFO,
    format = "%(asctime)s - %(message)s",
    datefmt = "%H:%M:%S",
    filename = "evaluation.log"
    )

    logging.info("loading the test dataset")

    NOR_dataset = SDiseaseDataset(
        image_dir=path_test_images_2d,
        mask_dir=path_test_masks_2d,
        transform=get_eval_transform(),
        info_csv_path=info_csv_path,
        disease_filter=["NOR"]
    )

    LV_dataset = SDiseaseDataset(
        image_dir=path_test_images_2d,
        mask_dir=path_test_masks_2d,
        transform=get_eval_transform(),
        info_csv_path=info_csv_path,
        disease_filter=["LV"]
    )

    HCM_dataset = SDiseaseDataset(
        image_dir=path_test_images_2d,
        mask_dir=path_test_masks_2d,
        transform=get_eval_transform(),
        info_csv_path=info_csv_path,
        disease_filter=["HCM"]
    )

    NOR_dataloader = DataLoader(NOR_dataset, batch_size=args.batch_size, shuffle=False)
    LV_dataloader = DataLoader(LV_dataset, batch_size=args.batch_size, shuffle=False)
    HCM_dataloader = DataLoader(HCM_dataset, batch_size=args.batch_size, shuffle=False)


    model = BaseNet(args.model, args.encoder)
    print(f"successfully build the model: {args.model} with the encoder: {args.encoder}" )
    logging.info(f"successfully build the model: {args.model} with the encoder: {args.encoder}" )
    device = torch.device(args.device)
    model.load_state_dict(torch.load(os.path.join(args.checkpoints_dir, "best_basenet.pth"), weights_only=True))
    print("successfully load the model weights")

    Nor_dice, Nor_iou = eval(NOR_dataloader, model, device)
    Lv_dice, Lv_iou = eval(LV_dataloader, model, device)
    Hcm_dice, Hcm_iou = eval(HCM_dataloader, model, device)

    logging.info("*****dice****iou")
    logging.info(f"Nor:  {Nor_iou:.4f},    {Nor_iou:.4f}")
    logging.info(f"Lv: {Lv_iou:.4f},    {Lv_iou:.4f}")
    logging.info(f"Hcm: {Hcm_iou:.4f},    {Hcm_iou:.4f}")

if __name__ == "__main__":
    main()




