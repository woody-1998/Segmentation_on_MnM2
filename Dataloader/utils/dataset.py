'''
Dataset Class for 2D Cardiac MRI Segmentation
'''


import os
import torch
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class CardiacMRIDataset(Dataset):
    '''
        Dataset for loading paired 2D cardiac MRI images and segmentation masks

        Args:
        - image_dir (str): Path to the folder containing image slices
        - mask_dir (str): Path to the folder containing mask slices
        - transform (callable, optional): Optional transform to apply to image and mask
        - return_filename (bool): If True, return the filename with each sample
        '''
    
    def __init__(self, image_dir, mask_dir, transform=None, return_filename=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.return_filename = return_filename

        # Match images and masks by filename
        self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        self.mask_filenames = sorted([f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')])

        # Ensure matched pairs
        assert len(self.image_filenames) == len(self.mask_filenames), "Mismatch between number of images and masks."


    def __len__(self):
        # Return the total number of image-mask pairs
        return len(self.image_filenames)
    

    def __getitem__(self, idx):
        # Get file paths for the image and corresponding mask
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load the image and mask volumes (2D slices in .nii.gz format)
        image = nib.load(img_path).get_fdata().astype(np.float32)
        mask = nib.load(mask_path).get_fdata().astype(np.uint8)

        # Ensure image and mask shapes match
        assert image.shape == mask.shape, f"Shape mismatch between {img_path} and {mask_path}"

        # Add channel dimension to match expected input shape: [C, H, W]
        image = np.expand_dims(image, axis=0) # Shape: [1, H, W]
        mask = np.expand_dims(mask, axis=0)   # Shape: [1, H, W]

        sample = {"image": image, "mask": mask}

        # Apply optional transformations
        if self.transform:
            sample = self.transform(sample)

        # Return image, mask, and optionally filename for reference
        if self.return_filename:
            return sample['image'], sample['mask'], self.image_filenames[idx]
        else:
            return sample['image'], sample['mask']