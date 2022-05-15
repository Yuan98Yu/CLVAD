from clvad.models.model import Model
from clvad.models.encoder import Encoder
import torch


class CL_Model(Model):
    def __init__(self, encoder: Encoder, proj_net) -> None:
        super().__init__()
        self.encoder = encoder
        self.proj_net = proj_net

    def forward(self, x):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    # def to(self, device):
    #     self.encoder.to(device)
    #     self.proj_net.to(device)

    # def multi_gpu(self):
    #     pass
