import numpy as np
import torch
import torch.nn as nn

#Making native class loader
class Dataset(torch.utils.data.Dataset):
    # Initialization method for the dataset
    def __init__(self, X_train, y_train):
        self.data = torch.from_numpy(X_train.astype(float))
        self.labels = torch.from_numpy(y_train.astype(float)).squeeze()

    # What to do to load a single item in the dataset (read image and label)    
    def __getitem__(self, index):
        data = self.data[index]
        lbl = self.labels[index]
        data = np.transpose(data, axes=[2,0,1])
        return data,lbl
    
    # Return the number of images
    def __len__(self):
        return len(self.data)

class VGG(nn.Module):
    def __init__(self, num_classes=11):
        super(VGG, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Added convolutional layer
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(2 * 2 * 256, 256)
        self.batch1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.batch2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.batch3 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.fc4 = nn.Linear(64, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.features(x)
        out = self.flatten(out)
        out = self.fc1(out)
        out = self.batch1(out)
        out = self.fc2(out)
        out = self.batch2(out)
        out = self.fc3(out)
        out = self.batch3(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.softmax(out)
        return out
