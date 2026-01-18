import math
import torch
from typing import Tuple, Dict
from torch.utils.data import IterableDataset, DataLoader


class InfiniteLinearTaskIterable(IterableDataset):
    """
    Infinite stream of episodic linear tasks (no masks, fixed Pc/Pt):
      For each episode/task:
        sample w
        sample Pc context points and Pt target points
        y = x @ w + noise
    """
    def __init__(
        self,
        x_dim: int,
        Pc: int = 16,
        Pt: int = 16,
        # x distribution
        x_dist: str = "uniform",                    # "uniform" or "normal"
        x_range: Tuple[float, float] = (-2.0, 2.0),
        # task distribution
        w_dist: str = "normal",                     # "normal" or "uniform"
        w_scale: float = 1.0,
        # noise distribution
        noise_dist: str = "uniform",                # "uniform" or "normal"
        noise_scale: float = 0.1,
        # reproducibility
        seed: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.x_dim = x_dim
        self.Pc = Pc
        self.Pt = Pt

        self.x_dist = x_dist
        self.x_range = x_range

        self.w_dist = w_dist
        self.w_scale = w_scale

        self.noise_dist = noise_dist
        self.noise_scale = noise_scale

        self.seed = seed
        self.dtype = dtype

        assert Pc > 0 and  Pt > 0

    def _sample_x(self, g: torch.Generator, n: int) -> torch.Tensor:
        if self.x_dist == "uniform":
            lo, hi = self.x_range
            return lo + (hi - lo) * torch.rand((n, self.x_dim), generator=g)
        if self.x_dist == "normal":
            return torch.randn((n, self.x_dim), generator=g)
        raise ValueError(f"Unknown x_dist: {self.x_dist}")

    def _sample_w(self, g: torch.Generator):
        if self.w_dist == "normal":
            w = self.w_scale * torch.randn((self.x_dim,), generator=g)
        elif self.w_dist == "uniform":
            w = (2.0 * torch.rand((self.x_dim,), generator=g) - 1.0) * self.w_scale
        else:
            raise ValueError(f"Unknown w_dist: {self.w_dist}")
        return w

    def _sample_noise(self, g: torch.Generator, n: int) -> torch.Tensor:
        # noise: [n, 1]
        if self.noise_scale == 0.0:
            return torch.zeros((n, 1))
        if self.noise_dist == "uniform":
            a = math.sqrt(self.noise_scale * 3.0)
            return (2.0 * torch.rand((n, 1), generator=g) - 1.0) * a
        if self.noise_dist == "normal":
            a = math.sqrt(self.noise_scale)
            return torch.randn((n, 1), generator=g) * a
        raise ValueError(f"Unknown noise_dist: {self.noise_dist}")

    def __iter__(self):
        # IMPORTANT: handle multi-worker properly
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # single-process
            g = torch.Generator()
            g.manual_seed(self.seed)
        else:
            # each worker gets a different seed stream
            g = torch.Generator()
            g.manual_seed(self.seed + worker_info.id * 10_000)

        while True:
            w = self._sample_w(g)

            x_ctx = self._sample_x(g, self.Pc)
            y_ctx = (x_ctx @ w).unsqueeze(-1) + self._sample_noise(g, self.Pc)

            x_tar = self._sample_x(g, self.Pt)
            y_tar = (x_tar @ w).unsqueeze(-1) + self._sample_noise(g, self.Pt)

            out: Dict[str, torch.Tensor] = {
                "x_ctx": x_ctx.to(self.dtype),
                "y_ctx": y_ctx.to(self.dtype),
                "x_tar": x_tar.to(self.dtype),
                "y_tar": y_tar.to(self.dtype),
            }

            yield out
def build_infinite_dataloader(
        batch_size: int = 256,
        x_dim: int = 10,
        Pc: int = 24,
        Pt: int  = 24,
        seed: int = 0,
        num_workers: int = 0,
        pin_memory: bool = False,
        **dataset_kwargs,
) -> DataLoader:
    ds = InfiniteLinearTaskIterable(
        x_dim=x_dim,
        Pc=Pc,
        Pt=Pt,
        seed=seed,
        **dataset_kwargs,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)


if __name__ == "__main__":
    dl = build_infinite_dataloader(
        batch_size=256,
        x_dim=5,
        Pc=16,
        Pt=16,
        x_dist="normal",
        x_range=(-2, 2),
        w_dist="normal",
        w_scale=1.0,
        noise_dist="uniform",
        noise_scale=0.1,
        seed=42,
        num_workers=0,
    )
    batch = next(iter(dl))
    print(batch["x_ctx"].shape, batch["y_ctx"].shape)

