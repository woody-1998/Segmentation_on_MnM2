'''
Preprocessing of 3D Cardiac MRI Volumes

Steps to prepare 3D cardiac MRI data for segmentation:
1) Gaussian smoothing for noise reduction.
2) Background removal by cropping to the non-zero voxel bounding box.
3) Z-score normalisation.
4) Resizing each 2D slice to a fixed resolution (256 x 256).

Output:
- Saved as 3D NIfTI images with the original affine matrix and header preserved.
- Stored in subject-specific folders under MnM2_preprocessed_3D/images/.
'''


import os
import nibabel as nib
import numpy as np

from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from skimage.transform import resize


'''
Configuration settings
'''
TARGET_SHAPE = (256, 256) # Target resolution for each 2D slice
SLICE_AXIS = 2            # Axis along which to slice (Z-axis)
SMOOTHING_SIGMA = 1.0     # Gaussian smoothing strength


def denoise(image, sigma=SMOOTHING_SIGMA):
    '''
    Apply Gaussian smoothing
    - Reduces noise in the MRI image.
    - Keeps anatomy while removing high-frequency signals.

    Args:
    - image (np.ndarray): 3D image volume to be smoothed.
    - sigma (float): Standard deviation for Gaussian kernel.

    Returns:
    - np.ndarray: Smoothed image volume.
    '''

    return gaussian_filter(image, sigma=sigma)


def remove_background(image):
    '''
    Remove background
    - Find the bounding box of all non-zero voxels.
    - Crop the image to remove empty background.

    Args:
    - image (np.ndarray): 3D image volume.

    Returns:
    - np.ndarray: Cropped image volume with only the foreground region.
    '''

    non_zero = np.argwhere(image) # Find indices of all non-zero voxels
    if non_zero.size == 0:
        return image

    # Determine the bounding box of non-zero voxels
    min_coords = non_zero.min(axis=0)
    max_coords = non_zero.max(axis=0) + 1

    # Crop the volume to the bounding box
    return image[min_coords[0]:max_coords[0],
                 min_coords[1]:max_coords[1],
                 min_coords[2]:max_coords[2]]


def normalise(image):
    '''
    Standardise voxel intensity using z-score.
    - Mean becomes 0, standard deviation becomes 1.

    Args:
    - image (np.ndarray): 3D image volume.

    Returns:
    - np.ndarray: Z-score normalised image volume.
    '''
    
    mean = np.mean(image) # Compute mean
    std = np.std(image) # Compute standard deviation

    # Apply Z-score normalisation
    if std != 0:
        return (image - mean) / (std + 1e-8) # Add small epsilon for numerical stability
    else:
        return image # Return original if standard deviation is zero


def resize_volume(volume, target_shape=TARGET_SHAPE):
    '''
    Resize each 2D slice to a fixed size
    - Ensures consistent input size across subjects.

    Args:
    - volume (np.ndarray): 3D image volume.
    - target_shape (tuple): Target resolution for each 2D slice (height, width).

    Returns:
    - np.ndarray: Resized 3D volume.
    '''

    resized_slices = []
    for i in range(volume.shape[SLICE_AXIS]):
        slice_2d = volume[:, :, i]
        resized = resize(slice_2d, target_shape, preserve_range=True) # Resize the slice to the target shape (256 x 256)
        resized_slices.append(resized)
        
    return np.stack(resized_slices, axis=SLICE_AXIS)


def preprocess_image(img_path):
    '''
    Apply full preprocessing pipeline to a single 3D image
    - Apply denoising, background removal, normalisation, and resizing.

    Args:
    - img_path (str): Path to the input NIfTI image.

    Returns:
    - tuple: (preprocessed image as np.ndarray, affine, header).
    '''
    
    img_nifti = nib.load(img_path)
    img = img_nifti.get_fdata()

    img = denoise(img)
    img = remove_background(img)
    img = normalise(img)
    img = resize_volume(img)

    return img, img_nifti.affine, img_nifti.header


def preprocess_all(input_dir, output_dir, file_list=None):
    '''
    Preprocess all images or only a selected list
    - If no file_list is provided, automatically walk through input_dir, and collect all image files (.nii.gz) excluding ground truth.
    - Each file undergoes preprocessing steps.
    - Save each subject preprocessed images into its own folder.

    Args:
    - input_dir (str): Root directory containing raw subject folders.
    - output_dir (str): Directory where preprocessed images will be saved.
    - file_list (list, optional): List of (dirpath, filename) pairs; If None, all .nii.gz files are auto-detected.

    Saves:
    - Preprocessed 3D NIfTI images in subject-specific subfolders.
    '''

    os.makedirs(output_dir, exist_ok=True)

    # If no file list provided, auto-collect all eligible image files
    if file_list is None:
        file_list = []
        for (dirpath, _, filenames) in os.walk(input_dir):
            for f in filenames:
                if f.endswith(".nii.gz") and "gt" not in f.lower():
                    file_list.append((dirpath, f))

    print("Total image files to preprocess:", len(file_list))

    # Loop through selected or discovered files
    for dirpath, file in tqdm(file_list, desc="Processing images"):
        subject_id = file.split('_')[0]
        file_path = os.path.join(dirpath, file)

        try:
            # Apply preprocessing steps
            processed_img, affine, header = preprocess_image(file_path)

            # Save into subject-specific folder under output_dir
            subj_out = os.path.join(output_dir, subject_id)
            os.makedirs(subj_out, exist_ok=True)

            out_path = os.path.join(subj_out, file)
            nib.save(nib.Nifti1Image(processed_img, affine, header), out_path)

        except Exception as e:
            print(f"Failed to process {file}: {e}")

    print("Preprocessing of 3D images complete...")
    print("All preprocessed 3D images have been saved to:", output_dir)