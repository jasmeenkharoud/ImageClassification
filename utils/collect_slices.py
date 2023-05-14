import os
import shutil


def copy_slices_from_subfolders(input_folder, output_folder):
    """Copy specified slices from all subfolders of the input folder to the output folder."""
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for root, dirs, files in os.walk(input_folder):
        png_files = [file for file in files if file.startswith('volume_0_slice_') and file.endswith('.png')]

        if png_files:
            total_images = len(png_files)
            start_slice = 0
            end_slice = 0

            print('total images = ', total_images)

            if total_images == 256:
                start_slice = 131
                end_slice = 200
            elif total_images == 192:
                start_slice = 71
                end_slice = 140

            if start_slice and end_slice:
                for file in png_files:
                    slice_number = int(file.split('_')[-1].split('.')[0])
                    if start_slice <= slice_number <= end_slice:
                        src_folder = os.path.abspath(root)
                        src_file = os.path.join(src_folder, file)
                        file_name = f'{os.path.basename(src_folder)}_{file}'
                        dest_file = os.path.join(output_folder, file_name)
                        shutil.copy(src_file, dest_file)


if __name__ == '__main__':

    input_folder = "./data/combined"
    output_folder = "./data/selected_data_combined"

    copy_slices_from_subfolders(input_folder, output_folder)
