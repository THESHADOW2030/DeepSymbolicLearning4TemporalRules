import torch.nn as nn
import torch.nn.functional as F
import torch

sftmx = torch.nn.Softmax(dim=-1)

def sftmx_with_temp(x, temp):
    return sftmx(x / temp)

class CNN_mnist(nn.Module):
    def __init__(self, channels, classes, nodes_linear):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, 3, 7, stride=2)
        self.conv2 = nn.Conv2d(3, 6, 7, stride=2)
        self.fc1 = nn.Linear(nodes_linear, classes)

        self.classes = classes

    def forward(self, x, temp = 1, transformation_matrix= None, symbols_filter=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))

        if transformation_matrix != None:
            x = torch.matmul(x, transformation_matrix)
            x = x * symbols_filter
        return sftmx_with_temp(x, temp)


class CNN_minecraft(nn.Module):
    def __init__(self, channels, classes):
        super(CNN_minecraft, self).__init__()
        self.conv1 = nn.Conv2d(channels, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(30, classes)
        self.classes = classes

    def forward(self, x, temp = 1):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3))
        # x = F.relu(F.max_pool2d(self.conv2_drop(x), 3))
        x = self.flat(x)
        #x = F.relu(self.fc1(x))
        return sftmx_with_temp(x, temp)
    

class CNN_mario(nn.Module):
    def __init__(self, channels, classes):
        super(CNN_minecraft, self).__init__()
        self.conv1 = nn.Conv2d(channels, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 5, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(30, classes)
        self.classes = classes

    def forward(self, x, temp = 1):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 3))
        # x = F.relu(F.max_pool2d(self.conv2_drop(x), 3))
        x = self.flat(x)
        #x = F.relu(self.fc1(x))
        return sftmx_with_temp(x, temp)


class Linear_classifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.lin1 = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = F.softmax(self.lin1(x), dim=-1)
        return x


class MNIST_Net(torch.nn.Module):
    def __init__(self, N=10, channels=1):
        super(MNIST_Net, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(channels, 12, 5),
            torch.nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            torch.nn.ReLU(True),
            torch.nn.Conv2d(12, 16, 5),  # 6 12 12 -> 16 8 8
            torch.nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            torch.nn.ReLU(True)
        )
        self.classifier_mid = torch.nn.Sequential(
            torch.nn.Linear(16 * 4 * 4, 120),
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU())
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(84, N),
            #torch.nn.Softmax(1)
        )
        self.channels = channels

    def weights_init(self, m):
        if isinstance(m, torch.nn.Conv2d):
            print('init conv2, ', m)
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

        if isinstance(m, torch.nn.Linear):
            print('init Linear, ', m)
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x, temp = 1):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier_mid(x)
        x1 = self.classifier(x)
        return sftmx_with_temp(x1, temp)