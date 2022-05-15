from typing import Dict
import glob
from PIL import Image

import numpy as np
import pandas as pd
import torch

from clvad.datasets.sampler_factory import make_clip_sampler


class Dataset(torch.utils.data.Dataset):
    DATASET_NAME = 'ABSTRACT_DATASET'

    def __init__(self,
                 split_file: str,
                 transform,
                 train: bool = True,
                 sampling_type: str = 'random',
                 clip_len: int = 32,
                 img_suffix='jpg'
                 ):
        """[summary]
        Args:
            split_file ([type]): [description]
            transform ([type], optional): [description]. Defaults to None.
            mode (str, optional): [description]. Defaults to 'test'.
            clip_len (int, optional): [description]. Defaults to 32.
            ds (int, optional): [description]. Defaults to 1.
            window (bool, optional): [description]. Defaults to False.
        """
        self.split_file = split_file
        self.train = train
        assert transform is not None
        self.transform = transform
        self.sampling_type_name = sampling_type
        self.clip_len = clip_len
        self.img_suffix = img_suffix

        self.clip_sampler = make_clip_sampler(num_seq=1, sampling_type=sampling_type, clip_len=clip_len)
        self.video_df = pd.read_csv(split_file)
        # df_mask = self.video_df['train']
        # self.video_df = self.video_df[df_mask]
        ########
        # self.video_df = pd.DataFrame(columns=['path', 'length']) if self.train else pd.DataFrame(columns=['path', 'length', 'label_file'])
        # for line in self.video_df.iterrows():
        #     if self.train and line['train']:
        #         self.video_df.append(line)
        #     elif not self.train and not line['train']:
        #         self.video_df.append(line)

    def __repr__(self) -> str:
        return (f'{self.DATASET_NAME.center(50, "-")}\n'
                f'List File: {self.list_file}'
                f'Clip Length: {self.clip_len}')

    def __getitem__(self, index) -> Dict[object, object]:
        vpath, vlen, label_file = self.video_df.iloc[index]

        img_list = sorted(glob.glob(vpath + f'/*.{self.img_suffix}'))
        assert len(img_list) == vlen

        frame_index = self.clip_sampler(vlen) if self.train else np.arange(
            vlen)
        seq = [Image.open(img_list[i]) for i in frame_index]

        if self.transform is not None:
            seq = self.transform(seq)
        seq = torch.stack(seq, 1)

        if self.train:
            label = np.ones(self.clip_sampler.clip_len * self.clip_sampler.num_seq)
        else:
            label = np.load(label_file)
            label = label[frame_index]

        return {
            'video': seq,
            'label': label,
            'video_path': vpath,
            'label_path': label_file
        }

    def __len__(self):
        return len(self.video_df)
