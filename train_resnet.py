import os
import torch.nn as nn
from torchvision.models import resnet152
import warnings
warnings.filterwarnings("ignore")
from custom_classifier import CustomClassifier


class ResNet152Classifier(CustomClassifier):
    def __init__(self, num_classes, learning_rate=1e-3, batch_size=8, num_epochs=100, output_folder='./weights'):
        model = resnet152(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        super().__init__(model, "ResNet152", num_classes, learning_rate, batch_size, num_epochs, output_folder)


def main():
    num_classes = 3
    learning_rate = 1e-3
    batch_size = 32
    num_epochs = 100
    dataset_name = 'combined'
    data_path = "./data/classifier_data/" + dataset_name
    output_folder = "./weights/" + dataset_name

    train_data_path = data_path + '/train'
    val_data_path = data_path + '/val'

    os.makedirs(output_folder + '/resnet152', exist_ok=True)

    resnet_classifier = ResNet152Classifier(num_classes, learning_rate, batch_size, num_epochs, output_folder + '/resnet152')
    resnet_train_losses, resnet_train_accuracies, resnet_val_losses, resnet_val_accuracies = resnet_classifier.train(train_data_path, val_data_path)
    resnet_classifier.plot_graphs(resnet_train_losses, resnet_train_accuracies, resnet_val_losses, resnet_val_accuracies)

    
if __name__ == '__main__':
    main()
