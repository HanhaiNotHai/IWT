import os
from pathlib import Path

import torch
from torch import Tensor
from torchvision.io import read_image
from torchvision.transforms import v2 as T


class Dataset(torch.utils.data.Dataset):

    def __init__(self, root: str | Path, img_size: int = 128, lw: int = 64, lk: int = 32):
        data = [read_image(os.path.join(root, f)) for f in os.listdir(root)]
        transform = T.Compose(
            [
                T.Resize(img_size),
                T.CenterCrop(img_size),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.data = list(map(transform, data))
        self.lw = lw
        self.lk = lk

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self.data[index],
            torch.randint(0, 2, [self.lw]),
            torch.randint(0, 2, [self.lk]) - 0.5,
        )


def main():
    trainset = Dataset('dataset/DIV2K/DIV2K_train_LR_x8')
    validset = Dataset('dataset/DIV2K/DIV2K_valid_LR_x8')
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=2, shuffle=True, pin_memory=True
    )
    validloader = torch.utils.data.DataLoader(validset, batch_size=2)
    for _ in range(2):
        for x, wm, key in trainloader:
            ...
        for x, wm, key in validloader:
            ...


if __name__ == '__main__':
    main()
