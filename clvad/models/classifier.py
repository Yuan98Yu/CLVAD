from clvad.models.model import Model
from clvad.models.encoder import Encoder


class Classifier(Model):
    def __init__(self, encoder: Encoder, net) -> None:
        super().__init__()
        self.encoder = encoder
        self.net = net

    def forward(self, x):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def load_encoder(self, path):
        pass

    def to(self, device):
        self.encoder.to(device)
        self.net.to(device)

    def multi_gpu(self):
        pass
