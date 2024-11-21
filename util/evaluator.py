from typing import Optional

import torch
from torch import Tensor
from torcheval.metrics import BinaryAccuracy, PeakSignalNoiseRatio
from torchvision.transforms import v2 as T


class Evaluator:

    def __init__(self, device: Optional[torch.device] = None):
        self.psnr = PeakSignalNoiseRatio(255.0, device=device)
        self.transform = T.Compose(
            [
                T.Normalize([-1], [2]),
                T.ToDtype(torch.uint8, scale=True),
            ]
        )

        self.acc = BinaryAccuracy(threshold=0, device=device)

    def reset(self) -> None:
        self.psnr.reset()
        self.acc.reset()

    @torch.inference_mode()
    def update(self, x_encoded: Tensor, img: Tensor, wm_decoded: Tensor, wm: Tensor) -> None:
        img_encoded = self.transform(x_encoded)
        self.psnr.update(img_encoded, img)

        self.acc.update(wm_decoded.flatten(), wm.flatten())

    @torch.inference_mode()
    def compute(self) -> tuple[Tensor, Tensor]:
        return self.psnr.compute(), self.acc.compute()


def main():
    img = torch.randint(0, 256, [2, 3, 128, 128], dtype=torch.uint8)
    t = T.Compose([T.ToDtype(torch.float32, scale=True), T.Normalize([0.5], [0.5])])
    x: Tensor = t(img)
    x_encoded = x.clone()
    x_encoded[0] *= 0.99
    x_encoded[1] *= 0.95

    wm = torch.randint(0, 2, [2, 64], dtype=torch.int32)
    wm_decoded = torch.rand(2, 64) * 2 - 1

    evaluator = Evaluator()
    evaluator.reset()
    evaluator.update(x_encoded, img, wm_decoded, wm)
    print(evaluator.compute())

    ...


if __name__ == '__main__':
    main()
