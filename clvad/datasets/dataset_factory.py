from typing import Dict
from clvad.datasets.dataset import Dataset
from clvad.datasets.shtech_dataset import SHTechDataset
from clvad.datasets.sampler_factory import make_clip_sampler
from clvad.datasets.transform_factory import make_transform


def make_sh_cl_train_dataset(args):
    pass


def make_sh_oneclass_train_dataset(args):
    pass


def make_sh_cl_val_dataset(args):
    pass


def make_sh_oneclass_val_dataset(args):
    pass


def make_sh_feature_extract_dataset(args: Dict[str, object]) -> Dataset:
    train_transform = make_transform(num_seq=1,
                                     clip_len=args['clip_len'],
                                     img_dim=args['img_dim'])
    dataset = SHTechDataset(split_file=args['split_file'],
                            train=True,
                            sampling_type=args['sampling_type'],
                            clip_len=args['clip_len'],
                            transform=train_transform)

    return dataset
