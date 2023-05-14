import os
import torch.nn as nn
from torchvision.models import resnext101_32x8d
import warnings
warnings.filterwarnings("ignore")
from custom_classifier import CustomClassifier


class Resnext101Classifier(CustomClassifier):
    def __init__(self, num_classes, learning_rate=1e-3, batch_size=8, num_epochs=100, output_folder='./weights'):
        model = resnext101_32x8d(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        super().__init__(model, "ResNext101", num_classes, learning_rate, batch_size, num_epochs, output_folder)


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

    os.makedirs(output_folder + '/resnext101', exist_ok=True)

    resnext_classifier = Resnext101Classifier(num_classes, learning_rate, batch_size, num_epochs, output_folder + '/resnext101')
    resnext_train_losses, resnext_train_accuracies, resnext_val_losses, resnext_val_accuracies = resnext_classifier.train(train_data_path, val_data_path)
    resnext_classifier.plot_graphs(resnext_train_losses, resnext_train_accuracies, resnext_val_losses, resnext_val_accuracies)


if __name__ == '__main__':
    main()
