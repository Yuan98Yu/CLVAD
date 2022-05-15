from abc import ABC, abstractmethod

import numpy as np


class ClipSampler(ABC):
    """Abstract class
    """
    def __init__(self, clip_len: int, ds: int) -> None:
        """
        Args:
            clip_len (int): length of video clip
            ds (int): downsample rate
        """
        super().__init__()
        self.clip_len = clip_len
        self.ds = ds
        self.num_seq = 1

    @abstractmethod
    def __call__(self, video_length: int) -> np.ndarray:
        """Sample a Clip in a given video.
        Args:
            video_length (int): the length of a given video
        Returns:
            indices [np.ndarray](num_seq, self.clip_len): An array with consequent numbers,
                representing the frame indices of a video clip.
        """
        pass


class RandomSampler(ClipSampler):
    """A sampler to get a random clip in one video
    """
    def __call__(self, video_length: int) -> np.ndarray:
        """Random select a video clip in a given video.
        Args:
            video_length (int): the length of a given video
        Returns:
            indices [np.ndarray]: An array(self.clip_len,) with consequent numbers,
                representing the frame indices of a video clip.
        """

        if video_length - self.clip_len * self.ds <= 0:  # pad left
            sequence = np.arange(self.clip_len)*self.ds + \
                np.random.choice(range(self.ds), 1)
            seq_idx = np.zeros_like(sequence)
            sequence = sequence[sequence < video_length]
            seq_idx[-len(sequence)::] = sequence
        else:
            start = np.random.choice(
                range(video_length - self.clip_len * self.ds), 1)
            seq_idx = np.arange(self.clip_len) * self.ds + start
        return seq_idx


class SlidingWindowSampler(ClipSampler):
    """A sampler to get a sequence in one video
    """
    def __call__(self, video_length: int) -> np.ndarray:
        """Random select a video clip in a given video.
        Args:
            video_length (int): the length of a given video
        Returns:
            indices [np.ndarray]: An array(self.clip_len,) with consequent numbers,
                representing the frame indices of a video clip.
        """

        if (video_length - self.clip_len * self.ds <=
                0):  # pad left, only sample once
            sequence = np.arange(self.clip_len) * self.ds
            seq_idx = np.zeros_like(sequence)
            sequence = sequence[sequence < video_length]
            seq_idx[-len(sequence)::] = sequence
        else:
            available = video_length - self.clip_len * self.ds
            start = np.expand_dims(
                np.arange(0, available + 1, self.clip_len * self.ds - 1), 1)
            # [num_sample, clip_len]
            seq_idx = np.expand_dims(np.arange(self.clip_len) * self.ds,
                                     0) + start
            print(video_length, seq_idx.shape)
            seq_idx = seq_idx.flatten(0)
        return seq_idx


class DoubleSampler(ClipSampler):
    """Sample two clips each call.
    Note: This sampler is used to bulid train_dataset for contrastive learning only.
    """
    def __init__(self,
                 clip_len: int,
                 ds: int,
                 sampling_type: str = 'random') -> None:
        super().__init__(clip_len, ds)
        self.num_seq = 2
        self.frame_sampler = self.__build_sampler(sampling_type)

    def __call__(self, total_length: int) -> np.ndarray:
        """Select two video clip in a given video.
        Args:
            video_length (int): the length of a given video
        Returns:
            indices [np.ndarray]: An array(2, self.clip_len) with consequent numbers,
                representing the frame indices of a video clip.
        """
        seq1 = self.frame_sampler(total_length)
        seq2 = self.frame_sampler(total_length)
        return np.concatenate([seq1, seq2])

    def __build_sampler(self, sampling_type):
        if sampling_type == 'random':
            sampler = RandomSampler(self.clip_len, self.ds)
        else:
            raise NotImplementedError
        return sampler
