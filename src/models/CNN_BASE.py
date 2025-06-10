import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.DIM_I = 3 * 3 * 11
        self.DIM_O = 1
        self.chw0, self.chw1, self.chw2, self.chw3 = 64, 256, 512, 256

        # 入力のチャンネル数は11、出力のチャンネル数はchw0
        self.conv0 = nn.Conv2d(in_channels=11, out_channels=self.chw0, kernel_size=1)
        self.conv1 = nn.Conv2d(
            in_channels=self.chw0, out_channels=self.chw1, kernel_size=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=self.chw1, out_channels=self.chw2, kernel_size=2
        )
        # 以降全結合層
        self.fc1 = nn.Linear(self.chw2, self.chw3)
        self.fc2 = nn.Linear(self.chw3, self.DIM_O)

    def forward(self, x: torch.Tensor):
        # Reshape input similar to TensorFlow's tf.reshape [-1, 3, 3, 11] -> [N, 11, 3, 3] for PyTorch
        x = x.view(-1, 11, 3, 3)

        # Conv layers and residual block
        output0 = F.relu(self.conv0(x))  # Conv2D (1x1)
        output1 = F.relu(self.conv1(output0))  # Conv2D (2x2) with padding
        output2 = F.relu(self.conv2(output1))  # Conv2D (2x2)

        # Reshape output for fully connected layers
        output2f = output2.view(-1, self.chw2)

        # Fully connected layers
        output3f = F.relu(self.fc1(output2f))
        output = self.fc2(output3f)

        return output
