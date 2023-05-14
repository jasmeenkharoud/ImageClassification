import os
from train_resnet import ResNet152Classifier
from train_resnext import Resnext101Classifier
from train_se_resnet import SenetResnetClassifier
from train_se_resnext import SenetResnextClassifier
from train_hybrid_se_resnext import HybridSenetResnextClassifier


def check_test_results(model, model_name, model_path):
    # Load the best model weights
    model.load_model(model_path)
    output_folder = os.path.join(output_path, dataset_name, model_name)
    if SAVE_TEST_RESULTS == True:
        os.makedirs(output_folder, exist_ok=True)
    model.test(os.path.join(test_data_path, dataset_name), save_images=SAVE_TEST_RESULTS, output_folder=output_folder)


def main():
    num_classes = 3
    
    # Test Resnet
    if 'resnet152' in model_architecture:
        model = ResNet152Classifier(num_classes=num_classes)
        best_weights_path = os.path.join(model_path, dataset_name, 'resnet152/ResNet152_best_weights.pth')
        print('\nTest results using Resnet152 architecture')
        check_test_results(model, 'resnet152', best_weights_path)

    # Test Resnext
    if 'resnext101' in model_architecture:
        model = Resnext101Classifier(num_classes=num_classes)
        best_weights_path = os.path.join(model_path, dataset_name, 'resnext101/ResNext101_best_weights.pth')
        print('\nTest results using Resnext101 architecture')
        check_test_results(model, 'resnext101', best_weights_path)

    # Test Senet_Resnet
    if 'se_resnet152' in model_architecture:
        model = SenetResnetClassifier(num_classes=num_classes)
        best_weights_path = os.path.join(model_path, dataset_name, 'se_resnet152/SE_ResNet152_best_weights.pth')
        print('\nTest results using Senet_Resnet152 architecture')
        check_test_results(model, 'se_resnet152', best_weights_path)

    # Test Senet_Resnext
    if 'se_resnext101' in model_architecture:
        model = SenetResnextClassifier(num_classes=num_classes)
        best_weights_path = os.path.join(model_path, dataset_name, 'se_resnext101/SE_ResNext101_best_weights.pth')
        print('\nTest results using Senet_Resnext101 architecture')
        check_test_results(model, 'se_resnext101', best_weights_path)

    # Test Hybrid_Senet_Resnext
    if 'hybrid_se_resnext101' in model_architecture:
        model = HybridSenetResnextClassifier(num_classes=num_classes)
        best_weights_path = os.path.join(model_path, dataset_name, 'hybrid_se_resnext101/Hybrid_SE_ResNext101_best_weights.pth')
        print('\nTest results using Hybrid Senet_Resnext101 architecture')
        check_test_results(model, 'hybrid_se_resnext101', best_weights_path)

# Set paths
dataset_name = 'combined'
output_path = 'results' # Path to save output images
test_data_path = './data/classifier_data'  # Dataset path
model_path = './weights'  # Provide model path
SAVE_TEST_RESULTS = False # Flag to save test results 
model_architecture = ['resnet152', 'resnext101', 'se_resnet152', 'se_resnext101', 'hybrid_se_resnext101'] # Add all the trained models to this list


if __name__ == '__main__':
    main()
