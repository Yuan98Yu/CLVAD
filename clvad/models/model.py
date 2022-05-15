import abc
from torch.nn.modules import Module


class Model(Module):
    @abc.abstractclassmethod
    def forward(self, x):
        pass

    @abc.abstractclassmethod
    def save(self, path):
        pass

    @abc.abstractclassmethod
    def load(self, path):
        pass

    # @abc.abstractclassmethod
    # def to(self, device):
    #     pass

    # @abc.abstractclassmethod
    # def multi_gpu(self):
    #     pass
