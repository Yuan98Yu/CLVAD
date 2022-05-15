from clvad.datasets import sampler


def make_clip_sampler(num_seq: int = 1,
                      sampling_type: str = 'random',
                      clip_len: int = 32,
                      ds: int = 1) -> sampler.ClipSampler:
    if num_seq == 2:
        double = True
    elif num_seq == 1:
        double = False
    else:
        raise NotImplementedError
    if double:
        clip_sampler = sampler.DoubleSampler(clip_len=clip_len,
                                             ds=ds,
                                             sampling_type=sampling_type)
    elif sampling_type == 'random':
        clip_sampler = sampler.RandomSampler(clip_len, ds)
    elif sampling_type == 'window':
        clip_sampler = sampler.SlidingWindowSampler(clip_len, ds)
    else:
        raise NotImplementedError
    return clip_sampler
