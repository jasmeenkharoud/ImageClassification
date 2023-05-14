import os
import numpy as np
import nibabel as nib
import cv2


def normalize_image(image):
    """Normalize the image intensity values to [0, 1]."""
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)


def extract_directory_from_path(file_path):
    """Extract the directory containing the input file."""
    return os.path.dirname(file_path)


def process_folder_recursively(input_folder):
    """Process a folder recursively and convert all NIfTI files to PNG format."""
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.nii'):
                nii_path = os.path.join(root, file)
                output_folder = extract_directory_from_path(nii_path)
                print(f'Processing: {nii_path}')
                save_slices_as_png(nii_path, output_folder)
                print(f'Saved PNG files in: {output_folder}')


def save_slices_as_png(nii_path, output_folder):
    """Save NIfTI image slices as PNG files."""
    # Load the NIfTI image
    nii_image = nib.load(nii_path)
    image_data = nii_image.get_fdata()

    # Normalize the image data
    normalized_data = normalize_image(image_data)

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Check the dimensions of the image data
    dimensions = len(image_data.shape)

    if dimensions == 2:
        cv2.imwrite(f'{output_folder}/image.png', (normalized_data * 255).astype(np.uint8))  # Change this line
    elif dimensions == 3:
        num_slices = image_data.shape[2]
        for i in range(num_slices):
            slice_data = normalized_data[:, :, i]
            cv2.imwrite(f'{output_folder}/slice_{i:04d}.png', (slice_data * 255).astype(np.uint8))  # Change this line
    elif dimensions == 4:
        num_volumes = image_data.shape[3]
        for v in range(num_volumes):
            volume_data = normalized_data[:, :, :, v]
            num_slices = volume_data.shape[2]
            for i in range(num_slices):
                slice_data = volume_data[:, :, i]
                cv2.imwrite(f'{output_folder}/volume_{v}_slice_{i:04d}.png', (slice_data * 255).astype(np.uint8))  # Change this line
    else:
        raise ValueError("Unsupported number of dimensions. Only 2D, 3D, and 4D NIfTI files are supported.")


if __name__ == '__main__':
    input_folder = './data'
    process_folder_recursively(input_folder)