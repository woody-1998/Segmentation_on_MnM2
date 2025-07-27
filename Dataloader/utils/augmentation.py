'''
Augmentation Pipeline for 2D Cardiac MRI Segmentation

Augmentation and transformations using the Albumentations library:
1) Spatial transforms (eg. flip, rotate) apply to both image and mask
2) Intensity transforms (eg. brightness, normalisation) apply to the image only
3) additional_targets={'mask': 'mask'} ensures correct transformation handling for masks

Purpose:
- Applies spatial and intensity-based data augmentations to training images and masks.
- Ensures consistent resizing and normalisation across train, validation, and test sets.
- Maintains pixel-wise alignment between MRI images and their segmentation masks.
- Converts both images and masks into PyTorch tensors for training segmentation models.
- Designed specifically for 2D slice-based cardiac MRI segmentation tasks.
'''


import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transform():
    '''
    Augmentation pipeline for the training set
    - Applies spatial and intensity-based transformations.
    - Assumes input images and masks are already preprocessed (z-score normalised and resized to 256x256).
    '''

    return A.Compose([
        A.Resize(256, 256),                 # Ensure consistent input size for the model (safe even if already resized)
        A.HorizontalFlip(p=0.5),            # Simulates anatomical left–right symmetry
        A.Affine(                 
            translate_percent=0.05,         # Simulates slight patient movement or scanner shift
            scale=(0.95, 1.05),             # Simulates zoom in/out
            rotate=(-10, 10),               # Adds slight orientation variation
            p=0.5,
            border_mode=0                   # Pad with zeros (black) when shifting/rotating outside bounds
        ),
        A.RandomBrightnessContrast(p=0.3),  # Adjusts brightness/contrast
        A.Normalize(mean=[0.0], std=[1.0]), # Z-score normalisation
        ToTensorV2()                        # Converts image and mask to PyTorch tensors [C, H, W], mask stays uint8
    ], additional_targets={'mask': 'mask'}) # Ensures the same transform is applied to the mask


def get_eval_transform():
    '''
    Minimal transform for validation and test sets
    - No augmentation.
    - Only resizing and normalisation for consistency.
    '''

    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=[0.0], std=[1.0]),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})
