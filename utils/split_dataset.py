import os
import shutil
from sklearn.model_selection import train_test_split


def split_class_data(input_path, output_path, val_ratio=0.1, test_ratio=0.1):
    """
    Splits the data in the input path into train, validation, and test sets and creates train, val, and test folders in
    the output path, then copies the corresponding files into their respective folders.

    :param input_path: Path to the directory containing the class folders.
    :param output_path: Path to the directory where the train, validation, and test folders should be created.
    :param val_ratio: Fraction of data to be used for validation.
    :param test_ratio: Fraction of data to be used for testing.
    """
    # Create the train, val, and test folders inside the output folder
    train_folder = os.path.join(output_path, "train")
    val_folder = os.path.join(output_path, "val")
    test_folder = os.path.join(output_path, "test")
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(val_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    # Loop through the class folders in the input directory
    for class_folder in os.listdir(input_path):
        class_input_path = os.path.join(input_path, class_folder)
        if not os.path.isdir(class_input_path):
            continue
        # Create the class folders in the train, val, and test directories
        class_train_folder = os.path.join(train_folder, class_folder)
        class_val_folder = os.path.join(val_folder, class_folder)
        class_test_folder = os.path.join(test_folder, class_folder)
        os.makedirs(class_train_folder, exist_ok=True)
        os.makedirs(class_val_folder, exist_ok=True)
        os.makedirs(class_test_folder, exist_ok=True)

        # Get the file names from the class folder
        file_names = os.listdir(class_input_path)

        # Split the data into train, val, and test sets
        train_files, test_val_files = train_test_split(file_names, test_size=(val_ratio + test_ratio))
        test_files, val_files = train_test_split(test_val_files, test_size=(val_ratio / (val_ratio + test_ratio)))

        # Copy the files into their respective class folders in the train, val, and test directories
        for file in train_files:
            shutil.copy(os.path.join(class_input_path, file), os.path.join(class_train_folder, file))
        for file in val_files:
            shutil.copy(os.path.join(class_input_path, file), os.path.join(class_val_folder, file))
        for file in test_files:
            shutil.copy(os.path.join(class_input_path, file), os.path.join(class_test_folder, file))


def main():
    # Set the input and output paths
    input_path = ".data/selected_data_combined"
    output_path = ".data/classifier_data/combined"
    
    # Data split ratio
    val_ratio = 0.2 
    test_ratio = 0.1

    # Split the data into train, validation, and test sets
    split_class_data(input_path, output_path, val_ratio, test_ratio)


if __name__ == "__main__":
    main()
