import torch.cuda
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize
import os
import cv2
import torch.nn as nn

class MyNeuralNetwork(nn.Module):
    def __init__(self, num_classes = 10):
        super().__init__()
        self.conv1 = self._make_block(3, 16, 3)
        self.conv2 = self._make_block(16, 32, 3)
        self.conv3 = self._make_block(32, 64, 3)
        self.conv4 = self._make_block(64, 128, 3)
        self.conv5 = self._make_block(128, 128, 3)

        self.fc1 = nn.Sequential (
            nn.Dropout(p=0.5),
            nn.Linear(in_features=128 * 7 * 7, out_features=2048),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(in_features=512, out_features=num_classes)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def _make_block(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

def main():

    model = MyNeuralNetwork()
    random_input = torch.rand(8, 3, 224, 224)
    result = model(random_input)
    print(result.shape)




if __name__ == "__main__" :
    main()