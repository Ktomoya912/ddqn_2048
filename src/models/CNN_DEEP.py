import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.DIM_I = 3 * 3 * 11
        self.DIM_O = 1
        self.chw0, self.chw1, self.chw2, self.chw3, self.chw4, self.chw5 = (
            64,
            128,
            232,
            256,
            256,
            256,
        )

        # 入力のチャンネル数は11、出力のチャンネル数はchw0
        self.conv0 = nn.Conv2d(in_channels=11, out_channels=self.chw0, kernel_size=1)
        self.conv1 = nn.Conv2d(
            in_channels=self.chw0, out_channels=self.chw1, kernel_size=2, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.chw1, out_channels=self.chw2, kernel_size=2
        )
        self.conv3 = nn.Conv2d(
            in_channels=self.chw2, out_channels=self.chw3, kernel_size=2
        )
        self.conv4 = nn.Conv2d(
            in_channels=self.chw3, out_channels=self.chw4, kernel_size=2
        )
        # 以降全結合層
        self.fc1 = nn.Linear(self.chw4, self.chw5)
        self.fc2 = nn.Linear(self.chw5, self.DIM_O)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 11, 3, 3)

        # Conv layers and residual block
        output0 = F.relu(self.conv0(x))  # Conv2D (1x1)
        output1 = F.relu(self.conv1(output0))  # Conv2D (2x2) with padding
        output2 = F.relu(self.conv2(output1))  # Conv2D (2x2)
        output3 = F.relu(self.conv3(output2))  # Conv2D (2x2)
        output4 = F.relu(self.conv4(output3))

        # Reshape output for fully connected layers
        output4f = output4.view(-1, self.chw4)

        # Fully connected layers
        output5f = F.relu(self.fc1(output4f))
        output = self.fc2(output5f)

        return output
