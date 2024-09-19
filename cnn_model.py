import torch.nn as nn
import torch
from torchvision.ops import SqueezeExcitation

class my_cnn(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = self._makeblock(3,8,3)
        self.conv2 = self._makeblock(8, 16, 3)
        self.conv3 = self._makeblock(16, 32, 3)
        self.conv4 = self._makeblock(32, 64, 3)
        self.conv5 = self._makeblock(64, 128, 3)

        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=(128*7*7), out_features=512),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=256),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(in_features=128, out_features=num_classes)

    def _makeblock(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            SqueezeExcitation(input_channels=out_channels, squeeze_channels=max(out_channels//16, 1))
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

if __name__ == '__main__':
    model = my_cnn(num_classes=10)
    fake_data = torch.rand(16,3,224,224)
    output = model(fake_data)
    print(output.shape)