import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np


class CustomClassifier:
    def __init__(self, model, model_name, num_classes, learning_rate, batch_size, num_epochs, output_folder):
        self.model = model
        self.model_name = model_name
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.output_folder = output_folder
        self.log_file = os.path.join(self.output_folder, f'{self.model_name}_metrics_log.csv')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate,
                                momentum=self.momentum, weight_decay=self.weight_decay)

    def _load_data(self, train_data_path, transform):
        train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def _calculate_class_weights(self, train_data_path, transform):
        train_dataset = torchvision.datasets.ImageFolder(root=train_data_path, transform=transform)
        class_counts = torch.zeros(self.num_classes)
        
        for _, label in train_dataset:
            class_counts[label] += 1
        
        class_weights = 1.0 / class_counts
        norm_weights = class_weights / class_weights.sum()
        
        return norm_weights.to(self.device)

    def train(self, train_data_path, val_data_path):
        # Preprocessing on train dataset
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Preprocessing on validation dataset
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        # Delete the existing log file if exists
        if os.path.exists(self.log_file):
            # Delete the existing file
            os.remove(self.log_file)
            print(f"Deleted existing log file: {self.log_file}")

        norm_weights = self._calculate_class_weights(train_data_path, train_transform)
        self.criterion = nn.CrossEntropyLoss(weight=norm_weights)

        train_loader = self._load_data(train_data_path, train_transform)
        val_loader = self._load_data(val_data_path, val_transform)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        best_accuracy = 0.0
        best_model_weights = None
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            self.adjust_learning_rate(epoch)
            self.model.train()
            print('Epoch-{0} learning rate: {1}'.format(epoch + 1, self.optimizer.param_groups[0]['lr']))
            start = datetime.now()
            for _, (data, targets) in enumerate(train_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                scores = self.model(data)
                loss = self.criterion(scores, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                _, predicted = scores.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                running_loss += loss.item()
            end = datetime.now()
            train_loss = running_loss / len(train_loader)
            train_accuracy = 100.0 * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            # Validation
            self.model.eval()
            val_running_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for data, targets in val_loader:
                    data = data.to(self.device)
                    targets = targets.to(self.device)

                    scores = self.model(data)
                    loss = self.criterion(scores, targets)

                    _, predicted = scores.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()

                    val_running_loss += loss.item()
            val_loss = val_running_loss / len(val_loader)
            val_accuracy = 100.0 * val_correct / val_total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            
            self.log_metrics(self.log_file, epoch + 1, train_loss, train_accuracy, val_loss, val_accuracy)  # Log metrics to the CSV file
            print(f'Epoch [{epoch + 1}], Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}, Time: {end-start}')

            if train_accuracies[-1] > best_accuracy:
                best_accuracy = train_accuracies[-1]
                best_model_weights = self.model.state_dict()

            torch.save(self.model.state_dict(), os.path.join(self.output_folder, f"{self.model_name}_weights.pth"))
            torch.save(best_model_weights, os.path.join(self.output_folder, f"{self.model_name}_best_weights.pth"))

        return train_losses, train_accuracies, val_losses, val_accuracies


    def plot_graphs(self, train_losses, train_accuracies, val_losses, val_accuracies):
        with PdfPages(os.path.join(self.output_folder, f'{self.model_name}_loss_accuracy_graphs.pdf')) as pdf:
            plt.figure(figsize=(10, 5))

            plt.subplot(1, 2, 1)
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'{self.model_name} Loss')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(train_accuracies, label='Train Accuracy')
            plt.plot(val_accuracies, label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'{self.model_name} Accuracy')
            plt.legend()

            pdf.savefig()
            plt.close()

    def log_metrics(self, log_file, epoch, train_loss, train_accuracy, val_loss, val_accuracy):
        if not os.path.exists(log_file):
            with open(log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch', 'Train_loss', 'train_accuracy', 'Val_loss', 'Val_accuracy'])

        with open(log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, train_loss, train_accuracy, val_loss, val_accuracy])

    # Learning rate decay
    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.learning_rate * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def load_model(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

    # Test model
    def test(self, test_data_path, batch_size=8, save_images=False, output_folder='./output'):
        self.model.eval()

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        test_dataset =  torchvision.datasets.ImageFolder(test_data_path + '/test', transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        all_targets = []
        all_predictions = []

        if save_images:
            for class_name in test_loader.dataset.classes:
                class_folder = os.path.join(output_folder, class_name)
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)

                for prediction_status in ["correct", "incorrect"]:
                    prediction_folder = os.path.join(class_folder, prediction_status)
                    if not os.path.exists(prediction_folder):
                        os.makedirs(prediction_folder)

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(test_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(data)
                _, predictions = torch.max(outputs.data, 1)

                if save_images:
                    for idx, (prediction, target) in enumerate(zip(predictions, targets)):
                        prediction_status = "correct" if prediction == target else "incorrect"
                        image_folder = os.path.join(output_folder, test_loader.dataset.classes[target], prediction_status)
                        image_path = os.path.join(image_folder, f"image_{batch_idx * batch_size + idx}.jpg")

                        # Reverse the normalization
                        img = data[idx].cpu().numpy()
                        img = (img * np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)) + np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)))
                        img = np.transpose(img, (1, 2, 0))

                        torchvision.utils.save_image(torch.tensor(img).permute(2, 0, 1), image_path)

                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)*100
        f1 = f1_score(all_targets, all_predictions, average='weighted')*100
        precision = precision_score(all_targets, all_predictions, average='weighted')*100
        recall = recall_score(all_targets, all_predictions, average='weighted')*100

        print(f'Test Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}\n')
        return accuracy, f1, precision, recall
