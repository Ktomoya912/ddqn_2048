import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.DIM_O = 1

        # -- 畳み込み層チャネル --
        c0, c1, c2, c3 = (64, 128, 256, 272)
        # ★c3 を 272 に微調整して総パラメータ ≒ 722 k

        # Conv blocks --------------------------------------------------------
        self.conv0 = nn.Conv2d(11, c0, kernel_size=1)  # 1×1
        self.conv1 = nn.Conv2d(c0, c1, kernel_size=2, padding=1)  # 2×2, P=1
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=2)  # 2×2
        self.conv3 = nn.Conv2d(c2, c3, kernel_size=2)  # 2×2
        # 出力は [N, c3, 2, 2] になる

        # FC blocks ----------------------------------------------------------
        self.fc1 = nn.Linear(c3 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, self.DIM_O)

    def forward(self, x):
        x = x.view(-1, 11, 3, 3)
        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(1)  # shape: [N, c3*4]
        x = F.relu(self.fc1(x))
        return self.fc2(x)
