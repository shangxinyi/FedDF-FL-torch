from torch.nn import Module, Conv2d, Linear, Flatten
from torch.nn.functional import relu, max_pool2d


class Cnn(Module):
    def __init__(self, channel: int, num_classes: int):
        super(Cnn, self).__init__()
        self.conv1 = Conv2d(channel, 32, 5)
        self.conv2 = Conv2d(32, 64, 5)
        self.fc1 = Linear(64 * 5 * 5, 512)
        self.fc2 = Linear(512, 128)
        self.fc3 = Linear(128, num_classes)
        self.flatten = Flatten()

    def forward(self, x):
        out = relu(self.conv1(x))
        out = max_pool2d(out, 2)
        out = relu(self.conv2(out))
        out = max_pool2d(out, 2)
        out = self.flatten(out)
        out = relu(self.fc1(out))
        out = relu(self.fc2(out))
        out = self.fc3(out)

        return out


class LeNet5(Module):
    def __init__(self, channel: int, num_classes: int):
        super(LeNet5, self).__init__()
        self.conv1 = Conv2d(channel, 6, 5)
        self.conv2 = Conv2d(6, 16, 5)
        self.fc1 = Linear(16 * 5 * 5, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(82, num_classes)
        self.flatten = Flatten()

    def forward(self, x):
        out = relu(self.conv1(x))
        out = max_pool2d(out, 2)
        out = relu(self.conv2(out))
        out = max_pool2d(out, 2)
        out = self.flatten(out)
        out = relu(self.fc1(out))
        out = relu(self.fc2(out))
        out = self.fc3(out)

        return out
