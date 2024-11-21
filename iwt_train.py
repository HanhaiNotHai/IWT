import os
from dataclasses import dataclass
from math import cos, pi
from pathlib import Path
from typing import Iterable

import torch
import wandb
from torch import Tensor, nn
from tqdm import tqdm, trange

from iwt import IWT, IWTParams
from util.dataset import TrainDataset, ValidDataset
from util.evaluator import Evaluator


@dataclass
class TrainParams:
    # Set proxy if you are using.
    proxy: str | None = None  # '127.0.0.1:7897'
    # Use wandb or not.
    WANDB: bool = False

    bs: int = 12
    valid_bs: int = 24
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
    ckpt_root: str | Path = 'checkpiont'

    def __post_init__(self):
        self.iwt_params = IWTParams(h=self.img_size, w=self.img_size, lw=self.lw, lk=self.lk)
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        if self.proxy is not None:
            os.environ['http_proxy'] = os.environ['https_proxy'] = self.proxy

    @property
    def wandb_config(self) -> dict:
        return dict(
            batch_size=self.bs,
            epochs=self.epochs,
            lambda1=self.lambda1,
            lambda2=self.lambda2,
            lr=self.lr,
            lr_min=self.lr_min,
        )


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
    validloader = torch.utils.data.DataLoader(validset, batch_size=params.valid_bs)

    iwt = IWT(params.iwt_params).to(params.device)

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.AdamW(iwt.parameters(), params.lr)
    scheduler = get_scheduler(params, optimizer, params.epochs * len(trainloader))

    evaluator = Evaluator(params.device)

    if params.WANDB:
        wandb.login()
        wandb.init(project='iwt', dir='wandb_log', config=params.wandb_config)
        wandb.watch(iwt, log='all')

    step = 0
    with trange(params.epochs, desc='epoch') as epoch_pbar:
        for epoch in epoch_pbar:
            with tqdm(trainloader, desc='train', leave=False) as train_pbar:
                for batch in train_pbar:
                    step += 1
                    x, wm, key = to(batch, params.device)

                    optimizer.zero_grad()

                    x_encoded, wm_decoded = iwt(x, wm, key)

                    loss1: Tensor = mse_loss(x_encoded, x)
                    loss2: Tensor = mse_loss(wm_decoded, wm - 0.5)
                    loss = params.lambda1 * loss1 + params.lambda2 * loss2

                    data = dict(
                        loss1=loss1.item(),
                        loss2=loss2.item(),
                        loss=loss.item(),
                        lr=scheduler.get_last_lr()[0],
                    )
                    wandb.log(data, step) if params.WANDB else None
                    train_pbar.set_postfix(data)

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

            evaluator.reset()
            with torch.inference_mode():
                iwt.eval()
                with tqdm(validloader, desc='valid', leave=False) as valid_pbar:
                    for batch in valid_pbar:
                        img, x, wm, key = to(batch, params.device)
                        x_encoded, wm_decoded = iwt(x, wm, key)
                        evaluator.update(x_encoded, img, wm_decoded, wm)
                iwt.train()
            metrics = evaluator.compute()
            wandb.log(metrics, epoch) if params.WANDB else None
            epoch_pbar.set_postfix(metrics)
            torch.save(
                iwt.state_dict(), os.path.join(params.ckpt_root, f'iwt_{epoch}_{metrics}.ckpt')
            )

    wandb.finish() if params.WANDB else None


if __name__ == '__main__':
    main()
