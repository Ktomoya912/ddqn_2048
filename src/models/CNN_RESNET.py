import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.DIM_I = 3 * 3 * 11
        self.DIM_O = 1
        chw0, chw3, chw4, chw5 = 64, 256, 480, 256
        chres = 128

        # 入力のチャンネル数は11、出力のチャンネル数はchw0
        self.conv0 = nn.Conv2d(in_channels=11, out_channels=chw0, kernel_size=1)
        self.conv1 = nn.Conv2d(
            in_channels=chw0, out_channels=chres, kernel_size=2, padding=1
        )
        self.conv2 = nn.Conv2d(in_channels=chres, out_channels=chw0, kernel_size=2)
        # TODO
        # Resdial Blockを実装する必要がある。

        self.conv3 = nn.Conv2d(in_channels=chw0, out_channels=chw3, kernel_size=2)
        self.conv4 = nn.Conv2d(in_channels=chw3, out_channels=chw4, kernel_size=2)

        # 以降全結合層
        self.fc1 = nn.Linear(chw4, chw5)
        self.fc2 = nn.Linear(chw5, self.DIM_O)

    def forward(self, x: torch.Tensor):
        # Reshape input similar to TensorFlow's tf.reshape [-1, 3, 3, 11] -> [N, 11, 3, 3] for PyTorch
        x = x.view(-1, 11, 3, 3)

        # Conv layers and residual block
        output0 = F.relu(self.conv0(x))  # Conv2D (1x1)
        output1 = F.relu(self.conv1(output0))  # Conv2D (2x2) with padding
        output2 = F.relu(self.conv2(output1))  # Conv2D (2x2)

        # Residual connection
        output2all = output2 + output0  # Skip connection

        output3 = F.relu(self.conv3(output2all))  # Conv2D (2x2)
        output4 = F.relu(self.conv4(output3))  # Conv2D (2x2)

        # Reshape output for fully connected layers
        output4f = output4.view(
            -1, output4.size(1)
        )  # Flatten (similar to TensorFlow reshape [-1, chw4])

        # Fully connected layers
        output5f = F.relu(self.fc1(output4f))
        output = self.fc2(output5f)

        return output
