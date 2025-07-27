'''
Extraction of 2D Slices from 3D MRI Images and Masks

Steps:
1) Read 3D NIfTI files from preprocessed folder, MnM2_preprocessed_3D/.
2) Extracts 2D axial slices from 3D short-axis cardiac MRI volumes and corresponding segmentation masks. 
3) Slices are saved in train/val/test folders based on a provided subject split DataFrame.

Output:
- Slices saved as 2D NIfTI files under MnM2_preprocessed_2Dslices/{train,val,test}/{images,masks}/.
'''


import os
import nibabel as nib
import pandas as pd
from tqdm import tqdm


def extract_slices(path_data, path_out, df_csv):
    '''
    Extracts 2D slices from 3D images and masks based on subject split

    Args:
    - path_data (str): Path to folder containing preprocessed 3D volumes.
    - path_out (str): Path to output folder for 2D slices.
    - df_csv (pd.DataFrame): DataFrame with columns 'SUBJECT_CODE' and 'SET_TYPE'.

    Returns:
    - None. Slices are saved to disk.
    '''

    # Normalise set type labels: Train/Test/Validation -> train/test/val
    df_csv.loc[:, 'SET_TYPE'] = df_csv['SET_TYPE'].str.lower().replace({'validation': 'val'})
                                                             
    # Create dictionary mapping subject IDs to their set type (train/val/test)
    split_dict = dict(zip(df_csv['SUBJECT_CODE'].astype(str), df_csv['SET_TYPE']))

    # Gather all eligible files
    all_files = []
    for dirpath, _, filenames in os.walk(path_data):
        for file in filenames:
            parts = file.split('_')

            # Filter only short-axis images (skip CINE and others)
            if len(parts) < 3 or parts[1] != 'SA' or parts[2].startswith('CINE'):
                continue

            all_files.append((dirpath, file))

    # Loop through files with progress bar
    for dirpath, file in tqdm(all_files, desc="Extracting slices"):
        parts = file.split('_')

        subject_id = parts[0]
        if subject_id not in split_dict:
            continue  # Skip if not in the provided CSV
            
        # Determine split (train/val/test) and folder type (image/mask)
        SET_TYPE = split_dict[subject_id]
        folder = 'masks' if 'gt' in file else 'images'

        # Load 3D volume
        full_path = os.path.join(dirpath, file)
        volume_nifti = nib.load(full_path)
        volume = volume_nifti.get_fdata()
        affine = volume_nifti.affine
        header = volume_nifti.header

        # Extract and save each 2D slice
        for i in tqdm(range(volume.shape[2]), desc=f"Saving {file}", leave=False):
            slice_2d = volume[:, :, i] # Axial slice
            out = nib.Nifti1Image(slice_2d, affine, header=header)

            # Name format: eg. 001_SA_ED_0003.nii.gz
            slice_name = f"{file.split('.')[0]}_{i:04}.nii.gz"
            save_dir = os.path.join(path_out, SET_TYPE, folder)
            os.makedirs(save_dir, exist_ok=True) # Create dir if it doesn't exist
            nib.save(out, os.path.join(save_dir, slice_name))

    print("Extracting slices complete...")
    print("All extracted slices have been saved to:", path_out)
