from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from iwt import IWT, IWTParams
from util.dataset import Dataset


@dataclass
class TrainParams:
    bs: int = 4
    epochs: int = 10
    lambda1: float = 2
    lambda2: float = 10
    lr: float = 1e-3
    eta_min = 1e-6

    img_size: int = 128
    lw: int = 64
    lk: int = 32

    train_root: str | Path = 'dataset/DIV2K/DIV2K_train_LR_x8'
    valid_root: str | Path = 'dataset/DIV2K/DIV2K_valid_LR_x8'

    def __post_init__(self):
        self.iwt_params = IWTParams(h=self.img_size, w=self.img_size, lw=self.lw, lk=self.lk)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')


def to(batch: Iterable[Tensor], device: torch.device) -> tuple[Tensor, ...]:
    return (x.to(device) for x in batch)


def main():
    params = TrainParams()

    trainset = Dataset(params.train_root, params.img_size, params.lw, params.lk)
    validset = Dataset(params.valid_root, params.img_size, params.lw, params.lk)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=params.bs, shuffle=True, pin_memory=True
    )
    validloader = torch.utils.data.DataLoader(validset, batch_size=params.bs)

    iwt = IWT(params.iwt_params).to(params.device)

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(iwt.parameters(), params.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=params.epochs * len(trainloader), eta_min=params.eta_min
    )

    for epoch in range(params.epochs):
        for batch in trainloader:
            x, wm, key = to(batch, params.device)

            optimizer.zero_grad()

            x_encoded, wm_decoded = iwt(x, wm, key)

            loss1 = mse_loss(x, x_encoded)
            loss2 = mse_loss(wm - 0.5, wm_decoded)
            loss: Tensor = params.lambda1 * loss1 + params.lambda2 * loss2

            loss.backward()
            optimizer.step()
            scheduler.step()


if __name__ == '__main__':
    main()