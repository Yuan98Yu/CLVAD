import torchvision.transforms as transforms

from . import augmentation as A


def make_transform(num_seq, clip_len, img_dim, train=True):
    if num_seq == 2:
        # clip_len = clip_len * 2  # for both rgb and flow
        clip_len = clip_len  # for both rgb and flow

        null_transform = transforms.Compose([
            A.RandomSizedCrop(size=img_dim, consistent=False,
                            clip_len=clip_len, bottom_area=0.2),
            A.RandomHorizontalFlip(consistent=False, clip_len=clip_len),
            A.ToTensor(),
        ])

        base_transform = transforms.Compose([
            A.RandomSizedCrop(size=img_dim, consistent=False,
                            clip_len=clip_len, bottom_area=0.2),
            transforms.RandomApply([
                A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.0,
                            consistent=False, clip_len=clip_len)
            ], p=0.8),
            A.RandomGray(p=0.2, clip_len=clip_len),
            transforms.RandomApply(
                [A.GaussianBlur([.1, 2.], clip_len=clip_len)], p=0.5),
            A.RandomHorizontalFlip(consistent=False, clip_len=clip_len),
            A.ToTensor(),
        ])

        # oneclip: temporally take one clip, random augment twice
        # twoclip: temporally take two clips, random augment for each
        # merge oneclip & twoclip transforms with 50%/50% probability
        transform = A.TransformController(
            [A.TwoClipTransform(base_transform, null_transform, clip_len=clip_len, p=0.3),
            A.OneClipTransform(base_transform, null_transform, clip_len=clip_len)],
            weights=[0.5, 0.5])
        # print(transform)
    elif num_seq == 1:
        if train:
            transform = transforms.Compose([
                A.RandomSizedCrop(size=img_dim, consistent=True, bottom_area=0.2),
                # A.Scale(img_dim),
                A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.3, consistent=True),
                A.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                A.RandomSizedCrop(size=img_dim, consistent=True, bottom_area=0.2),
                # A.Scale(img_dim),
                A.ToTensor(),
            ])
    else:
        raise NotImplementedError

    return transform
