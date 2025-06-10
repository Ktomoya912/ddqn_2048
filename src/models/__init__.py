from .alpha_zero import AlphaZero
from .cnn22_3x3 import CNN22_3x3
from .cnn_base import CNN_BASE
from .cnn_base_256 import CNN_BASE_256
from .cnn_base_1024 import CNN_BASE_1024
from .cnn_base_2048 import CNN_BASE_2048
from .cnn_deep import CNN_DEEP
from .cnn_deep_512 import CNN_DEEP_512
from .cnn_deep_1024 import CNN_DEEP_1024
from .cnn_four import CNN_FOUR
from .cnn_resnet import CNN_RESNET
from .cnn_resnet2 import CNN_RESNET2
from .cnn_skip_and_dense import CNN_SKIP_AND_DENSE
from .cnn_skip_and_dense2 import CNN_SKIP_AND_DENSE2

__all__ = [
    "AlphaZero",
    "CNN22_3x3",
    "CNN_BASE",
    "CNN_DEEP",
    "CNN_FOUR",
    "CNN_RESNET",
    "CNN_RESNET2",
    "CNN_SKIP_AND_DENSE",
    "CNN_SKIP_AND_DENSE2",
    "CNN_BASE_256",
    "CNN_BASE_1024",
    "CNN_BASE_2048",
    "CNN_DEEP_512",
    "CNN_DEEP_1024",
]
