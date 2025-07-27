'''
Binarisation of 3D Ground Truth Segmentation Masks

Steps to prepare 3D ground truth masks for right ventricular (RV) segmentation:
1) Load segmentation masks from MnM2 dataset.
2) Extract only the RV region by binarising label 3,
   - Set voxel = 1 if label == 3
   - Set all other voxels = 0
3) Resize each 2D slice to (256 × 256) to match preprocessed images.
4) Preserve affine metadata for spatial consistency.

Output:
- Saved as binary 3D NIfTI masks with the original affine and header preserved.
- Stored in subject-specific folders under MnM2_preprocessed_3D/masks/.
'''


import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.transform import resize


'''
Configuration settings
'''
TARGET_SHAPE = (256, 256) # Target resolution for each 2D slice


def resize_volume(volume, target_shape=TARGET_SHAPE):
    '''
    Resize each 2D slice of a binary mask using nearest-neighbor interpolation

    Args:
    - volume (ndarray): 3D NumPy array representing the binary mask volume (shape: H × W × D).
    - target_shape (tuple): Desired output shape for each 2D slice (256, 256).

    Returns:
    - ndarray: Resized 3D binary mask volume with shape (target_shape[0], target_shape[1], D), values in {0, 1}, dtype = uint8.
    '''

    resized_slices = []
    for i in range(volume.shape[2]):
        slice_2d = volume[:, :, i]

        # For binary masks, use order=0 (nearest-neighbor) to preserve discrete labels
        resized = resize(slice_2d, target_shape, order=0, preserve_range=True)
        resized_slices.append(resized)
    
    return np.stack(resized_slices, axis=2).astype(np.uint8)


def binarise_all(path_data, path_out, file_list=None):
    '''
    Binarise all RV masks 
    - Get the raw masks from the MnM2 dataset.
    - Save the binarised masks to subject folders.

    Args:
    - path_data (str): Path to raw dataset (used only if file_list is None).
    - path_out (str): Output path to save binarised 3D masks.
    - file_list (list of (dirpath, filename)): Optional list of specific files to binarise.

    Returns:
    - None: Saves binarised and resized 3D NIfTI masks to subject-specific folders under path_out.
    '''

    os.makedirs(path_out, exist_ok=True)

    # If no file list provided, collect all SA_ED/SA_ES ground truth masks
    if file_list is None:
        file_list = []
        for (dirpath, _, filenames) in os.walk(path_data):
            for file in filenames:
                parts = file.split('_')
                if (
                    len(parts) >= 3 and
                    parts[1] == 'SA' and
                    parts[2].split('.')[0] in ['ED', 'ES'] and
                    'gt' in file.lower() and
                    'CINE' not in parts[2]
                ):
                    file_list.append((dirpath, file))

    print("Total mask files to binarise:", len(file_list))

    # Binarise and save each RV masks
    for (dirpath, file) in tqdm(file_list, desc="Processing masks"):
        subject_id = file.split('_')[0]
        file_path = os.path.join(dirpath, file)

        try:
            # Load volume and metadata
            nifti_img = nib.load(file_path)
            img_3d = nifti_img.get_fdata()
            affine = nifti_img.affine
            header = nifti_img.header

            # Creates binary mask with 1 = RV, 0 = everything else
            binary_mask = (img_3d == 3).astype(np.uint8)
            resized_mask = resize_volume(binary_mask, target_shape=TARGET_SHAPE)

            # Save to subject-specific folder
            subj_out_dir = os.path.join(path_out, subject_id)
            os.makedirs(subj_out_dir, exist_ok=True)

            output_path = os.path.join(subj_out_dir, file)
            nib.save(nib.Nifti1Image(resized_mask, affine, header), output_path)

        except Exception as e:
            print(f"Failed to process {file}: {e}")

    print("Preprocessing of masks complete...")
    print("All preprocessed masks have been saved to:", path_out)
