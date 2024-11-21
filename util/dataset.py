import os
from pathlib import Path

import torch
from torch import Tensor
from torchvision.io import read_image
from torchvision.transforms import v2 as T


class Dataset(torch.utils.data.Dataset):

    def __init__(self, root: str | Path, img_size: int = 128, lw: int = 64, lk: int = 32):
        imgs = [read_image(os.path.join(root, f)) for f in os.listdir(root)]
        transform1 = T.Compose(
            [
                T.Resize(img_size),
                T.CenterCrop(img_size),
            ]
        )
        transform2 = T.Compose(
            [
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.5], [0.5]),
            ]
        )
        self.imgs = list(map(transform1, imgs))
        self.data = list(map(transform2, self.imgs))
        self.lw = lw
        self.lk = lk

    def __len__(self) -> int:
        return len(self.data)


class TrainDataset(Dataset):

    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.data[index],
            torch.randint(0, 2, [self.lw]),
            torch.randint(0, 2, [self.lk]) - 0.5,
        )


class ValidDataset(Dataset):

    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            self.imgs[index],
            self.data[index],
            torch.randint(0, 2, [self.lw]),
            torch.randint(0, 2, [self.lk]) - 0.5,
        )


def main():
    trainset = TrainDataset('dataset/DIV2K/DIV2K_train_LR_x8')
    validset = ValidDataset('dataset/DIV2K/DIV2K_valid_LR_x8')
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=2, shuffle=True, pin_memory=True
    )
    validloader = torch.utils.data.DataLoader(validset, batch_size=2)
    for _ in range(2):
        for x, wm, key in trainloader:
            ...
        for img, x, wm, key in validloader:
            ...


if __name__ == '__main__':
    main()
