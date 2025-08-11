'''
Dataset Class for 2D Cardiac MRI Segmentation
'''

import os
import torch
import nibabel as nib
import numpy as np
import pandas as pd  # <-- NEW
from typing import List, Optional  # <-- NEW
from torch.utils.data import Dataset


class CardiacMRIDataset(Dataset):
    '''
        Dataset for loading paired 2D cardiac MRI images and segmentation masks

        Args:
        - image_dir (str): Path to the folder containing image slices
        - mask_dir (str): Path to the folder containing mask slices
        - transform (callable, optional): Optional transform to apply to image and mask
        - return_filename (bool): If True, return the filename with each sample

        # NEW (all optional):
        - info_csv_path (str or None): Path to MnM2 dataset_information.csv
        - disease_filter (List[str] or None): e.g., ["NOR", "LV", "HCM"] to keep only these diseases
        - subject_col (str): Column name for subject in CSV (default "SUBJECT_CODE")
        - disease_col (str): Column name for disease in CSV (default "DISEASE")
        - subject_rule (callable or None): Function mapping filename -> subject_id.
          By default assumes names like '098_SA_ES_0005.nii.gz' and returns '098'.
    '''

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform=None,
        return_filename: bool = False,
        info_csv_path: Optional[str] = None,          # <-- NEW
        disease_filter: Optional[List[str]] = None,   # <-- NEW
        subject_col: str = "SUBJECT_CODE",            # <-- NEW
        disease_col: str = "DISEASE",                 # <-- NEW
        subject_rule=None                              # <-- NEW
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.return_filename = return_filename

        # Default subject rule: for '098_SA_ES_0005.nii.gz' -> '098'
        if subject_rule is None:
            def _default_subject_rule(fname: str) -> str:
                base = os.path.basename(fname)
                if base.endswith(".nii.gz"):
                    base = base[:-7]
                return base.split("_")[0].strip()
            self.subject_rule = _default_subject_rule
        else:
            self.subject_rule = subject_rule

        # --- FAST PRE-FILTERING (optional) ---
        if info_csv_path is not None and disease_filter:
            # Normalize disease list
            disease_filter = [d.upper().strip() for d in disease_filter]

            # Read CSV once and get allowed subjects
            df = pd.read_csv(info_csv_path, dtype=str)
            df[subject_col] = df[subject_col].str.strip()
            df[disease_col] = df[disease_col].str.upper().str.strip()
            allowed_subjects = set(
                df[df[disease_col].isin(disease_filter)][subject_col].dropna().tolist()
            )

            # Scan dirs, but keep ONLY files that match allowed subjects
            all_images = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
            all_masks  = sorted([f for f in os.listdir(mask_dir)  if f.endswith('.nii.gz')])

            keep_idxs = []
            for i, fname in enumerate(all_images):
                subj = self.subject_rule(fname)
                if subj in allowed_subjects:
                    keep_idxs.append(i)

            self.image_filenames = [all_images[i] for i in keep_idxs]
            self.mask_filenames  = [all_masks[i]  for i in keep_idxs]

        else:
            # Original behavior (no filtering)
            self.image_filenames = sorted([f for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
            self.mask_filenames  = sorted([f for f in os.listdir(mask_dir)  if f.endswith('.nii.gz')])

        # Ensure matched pairs
        assert len(self.image_filenames) == len(self.mask_filenames), "Mismatch between number of images and masks."

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_filenames[idx])

        # Load (H,W) or (H,W,1)
        image = nib.load(img_path).get_fdata().astype(np.float32)
        mask  = nib.load(mask_path).get_fdata().astype(np.uint8)

        # Squeeze any trailing singleton like (H,W,1) -> (H,W)
        image = np.squeeze(image)
        mask  = np.squeeze(mask)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]   # often torch.Tensor, shape [1,H,W] from ToTensorV2
            mask  = transformed["mask"]
        # else: keep numpy

        # ---- Convert to tensors if needed
        if not torch.is_tensor(image):
            image = torch.as_tensor(image, dtype=torch.float32)
        if not torch.is_tensor(mask):
            mask = torch.as_tensor(mask, dtype=torch.float32)

        # ---- Ensure CHW, but only add channel if 2D
        if image.ndim == 2:   # [H,W] -> [1,H,W]
            image = image.unsqueeze(0)
        elif image.ndim == 3:
            pass              # already [C,H,W]
        else:
            raise ValueError(f"Unexpected image ndim: {image.ndim}")

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        elif mask.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected mask ndim: {mask.ndim}")

        # Binary mask in float for BCEWithLogits
        mask = (mask > 0).float()

        if self.return_filename:
            return image, mask, self.image_filenames[idx]
        return image, mask
