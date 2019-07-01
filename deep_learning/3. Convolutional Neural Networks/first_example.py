import torch.nn as nn
import torch.nn.functional as F
from ann_visualizer.visualize import ann_viz
import torch


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_1, hidden_2 = 512, 512

        self.fc1 = nn.Linear(28 * 28, hidden_1)

        self.fc2 = nn.Linear(hidden_1, hidden_2)

        # we want to output 10 class probabilities
        self.fc3 = nn.Linear(hidden_2, 10)

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        x = x.view(-1, 28 * 28)

        x = F.relu(self.fc1(x))

        x = self.dropout(x)

        x = F.relu(self.fc2(x))

        x = self.dropout(x)

        x = self.fc3(x)

        return x


model = Net()
print(model)


ann_viz(model, title="My first neural network")