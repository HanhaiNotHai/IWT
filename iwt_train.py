from dataclasses import dataclass
from math import cos, pi
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from iwt import IWT, IWTParams
from util.dataset import TrainDataset, ValidDataset
from util.evaluator import Evaluator


@dataclass
class TrainParams:
    bs: int = 8
    epochs: int = 10
    lambda1: float = 2
    lambda2: float = 10
    lr: float = 1e-3
    lr_min = 1e-6

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


def get_scheduler(
    params: TrainParams, optimizer: torch.optim.Optimizer, epoch_max: int
) -> torch.optim.lr_scheduler.LRScheduler:
    coefficient = pi / epoch_max
    b = params.lr_min / params.lr
    w = (1 - b) / 2

    def lr_lambda(epoch: int) -> float:
        return (cos(epoch * coefficient) + 1) * w + b

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def to(batch: Iterable[Tensor], device: torch.device) -> tuple[Tensor, ...]:
    return (x.to(device) for x in batch)


def main():
    params = TrainParams()

    trainset = TrainDataset(params.train_root, params.img_size, params.lw, params.lk)
    validset = ValidDataset(params.valid_root, params.img_size, params.lw, params.lk)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=params.bs, shuffle=True, pin_memory=True
    )
    validloader = torch.utils.data.DataLoader(validset, batch_size=params.bs)

    iwt = IWT(params.iwt_params).to(params.device)

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(iwt.parameters(), params.lr)
    scheduler = get_scheduler(params, optimizer, params.epochs * len(trainloader))

    evaluator = Evaluator(params.device)

    for epoch in range(params.epochs):
        for batch in trainloader:
            x, wm, key = to(batch, params.device)

            optimizer.zero_grad()

            x_encoded, wm_decoded = iwt(x, wm, key)

            loss1 = mse_loss(x_encoded, x)
            loss2 = mse_loss(wm_decoded, wm - 0.5)
            loss: Tensor = params.lambda1 * loss1 + params.lambda2 * loss2

            loss.backward()
            optimizer.step()
            scheduler.step()

        iwt.eval()
        evaluator.reset()
        with torch.inference_mode():
            for batch in validloader:
                img, x, wm, key = to(batch, params.device)
                x_encoded, wm_decoded = iwt(x, wm, key)
                evaluator.update(x_encoded, img, wm_decoded, wm)
        psnr, acc = evaluator.compute()
        print(f'epoch: {epoch}, psnr: {psnr}, acc: {acc}')
        iwt.train()


if __name__ == '__main__':
    main()
