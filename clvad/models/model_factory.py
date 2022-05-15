from base64 import encode
import torch
from torch import nn

from clvad.models.classifier import Classifier
from clvad.models.cl_model import CL_Model
from clvad.models.encoder import Encoder


def make_encoder_from_hub(args) -> Encoder:
    pass
    model = torch.hub.load("facebookresearch/pytorchvideo:main",
                           model="x3d_s",
                           pretrained=True)
    layers = list(model.blocks.children())
    _layers = layers[:-1]
    feature_extractor = nn.Sequential(*_layers)
    encoder = Encoder(feature_extractor)

    return encoder


def make_classifier_from_scratch(args):
    pass
