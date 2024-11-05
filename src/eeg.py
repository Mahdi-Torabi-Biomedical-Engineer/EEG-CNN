import csv
import os
import shutil
import requests
import zipfile
import numpy as np
import numpy.fft as fft
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torch
import torch.nn as nn
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms


# Download and unzip the dataset
def download_dataset(link, drive_path, filename):
    """Downloads and extracts a dataset from a given URL."""
    os.makedirs(drive_path, exist_ok=True)
    file_path = os.path.join(drive_path, filename)

    # Download the dataset
    response = requests.get(link, stream=True)
    with open(file_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)

    # Unzip the downloaded file
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(drive_path)

def load_eeg_data(filepath):
    dataset = pd.read_csv(filepath)
    ecg_signal = dataset['hart'].values
    return ecg_signal, dataset


# Dataset classes
class MindBigData(Dataset):
    """Custom dataset class for loading MindBigData EEG signals."""

    def __init__(self, inputs, labels, transform=None):
        self.inputs = inputs
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        input_data = self.inputs[index]
        label = self.labels[index]
        if self.transform:
            input_data = self.transform(input_data)
            label = self.transform(label)
        return input_data, label


def preprocess_eeg_data(input_file, samples_per_digit=2000):
    """Preprocesses EEG data, filtering and scaling for each channel."""
    x, y, labels_hist = load_eeg_data(input_file, samples_per_digit)
    x = preprocess(x)
    return x, y


def preprocess(x):
    """Applies filtering, standardization, and trimming to EEG signals."""
    fs = 128.0  # Sampling frequency in Hz
    x_new = np.copy(x)
    x_trimmed = np.zeros((x.shape[0], x.shape[1], 224), dtype=np.float32)

    # Define filters
    b_band, a_band = signal.butter(N=6, Wn=[0.5, 63], btype='bandpass', fs=fs)
    b_notch, a_notch = signal.iirnotch(w0=50, Q=30, fs=fs)

    for i in range(x.shape[0]):
        x_new[i] = signal.lfilter(b_band, a_band, x[i])
        x_new[i] = signal.lfilter(b_notch, a_notch, x_new[i])
        x_trimmed[i] = x_new[i][:, 32:256]

    return x_trimmed


def get_data_loaders(x, y, batch_size=64, test_split=0.25, validation_split=0.6):
    """Splits data into training, validation, and test sets, and creates DataLoader instances."""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_split, random_state=42)
    train_data = MindBigData(x_train, y_train)
    test_data = MindBigData(x_test, y_test)

    indices = list(range(len(test_data)))
    np.random.shuffle(indices)

    split = int(np.floor(validation_split * len(test_data)))
    valid_sampler = SubsetRandomSampler(indices[:split])
    test_sampler = SubsetRandomSampler(indices[split:])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(test_data, sampler=valid_sampler, batch_size=batch_size)
    test_loader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    return train_loader, valid_loader, test_loader


class Conv2dBlockELU(nn.Module):
    """Defines a convolutional block with ELU activation and batch normalization."""

    def __init__(self, in_ch, out_ch, kernel_size, padding=(0, 0), dilation=(1, 1), w_in=None):
        super(Conv2dBlockELU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ELU(inplace=True)
        )
        if w_in is not None:
            self.w_out = int(((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / 1) + 1)
        self.out_ch = out_ch

    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):
    """Convolutional Neural Network for EEG data classification."""

    def __init__(self, num_channel=10, num_classes=4, signal_length=1000,
                 filters_n1=4, kernel_window_eeg=39, kernel_window=19,
                 conv_3_dilation=4, conv_4_dilation=4):
        super().__init__()
        filters = [filters_n1, filters_n1 * 2]
        self.conv_1 = Conv2dBlockELU(1, filters[0], kernel_size=(1, kernel_window_eeg), w_in=signal_length)
        self.conv_1_1 = Conv2dBlockELU(filters[0], filters[0], kernel_size=(1, kernel_window_eeg),
                                       w_in=self.conv_1.w_out)
        self.conv_2 = Conv2dBlockELU(filters[0], filters[0], kernel_size=(num_channel, 1))
        self.conv_3 = Conv2dBlockELU(filters[0], filters[1], kernel_size=(1, kernel_window),
                                     padding=(0, conv_3_dilation - 1), dilation=(1, conv_3_dilation),
                                     w_in=self.conv_1.w_out)
        self.conv_4 = Conv2dBlockELU(filters[1], filters[1], kernel_size=(1, kernel_window),
                                     padding=(0, conv_4_dilation - 1), dilation=(1, conv_4_dilation),
                                     w_in=self.conv_3.w_out)
        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(480, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # Add channel dimension
        x = self.conv_1(x)
        x = self.conv_1_1(x)
        x = self.conv_2(x)
        x = self.dropout(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc_layer(x)
        return x


class EarlyStopping:
    """Implements early stopping for model training."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def evaluate_model(model, dataloader, device, criterion):
    """Evaluates model performance and calculates accuracy and loss on validation or test data."""
    model.eval()
    total_correct = 0
    total_images = 0
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    accuracy = total_correct / total_images * 100
    avg_loss = running_loss / len(dataloader)
    return accuracy, avg_loss


# Train and validate model
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=200):
    """Trains and validates the CNN model with early stopping."""
    train_losses, valid_losses = [], []
    early_stopping = EarlyStopping(patience=10, verbose=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_correct = 0
        total_images = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_images += labels.size(0)
            total_correct += (predicted == labels).sum().item()

        train_acc = total_correct / total_images * 100
        train_loss = running_loss / len(train_loader)
        val_acc, val_loss = evaluate_model(model, valid_loader, device, criterion)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        train_losses.append(train_loss)
        valid_losses.append(val_loss)

        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Plot training and validation losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Main code execution
link = 'http://www.mindbigdata.com/opendb/MindBigData-EP-v1.0.zip'
drive_path = './data/'
download_dataset(link, drive_path, "MindBigData-EP-v1.0.zip")

# Load, preprocess and get DataLoaders
x, y = preprocess_eeg_data(input_file="data/EP1.01.txt", samples_per_digit=6500)
train_loader, valid_loader, test_loader = get_data_loaders(x, y, batch_size=64)

# Initialize the model, criterion, optimizer, and scheduler
model = CNN(num_channel=14, num_classes=10, signal_length=224, kernel_window_eeg=32, filters_n1=8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.06, weight_decay=9e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Train the model
train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, device, num_epochs=200)
