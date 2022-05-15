from clvad.models.model import Model
import torch


class Encoder(Model):
    def __init__(self, net) -> None:
        super().__init__()
        self.net = net

    def forward(self, x):
        return self.net(x)

    def save(self, path):
        pass

    def load(self, path):
        pass

    # def to(self, device):
    #     self.net.to(device)

    # def multi_gpu(self):
    #     pass
