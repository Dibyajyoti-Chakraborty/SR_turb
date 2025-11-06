import os
from glob import glob
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from einops import rearrange

class TurbulenceSRDataset(Dataset):
    """
    PyTorch Dataset for 2D turbulence super-resolution.
    Returns (low_res, high_res) pairs from the same timestep.
    The low-res input is created by downsampling the high-res image.
    """

    def __init__(self, root: str, upscale: int, split: str = "train", normalize: bool = False):
        """
        Args:
            root (str): Root directory containing 'train' and 'test' subfolders.
            upscale (int): The factor by which to downsample the input image (e.g., 4).
            split (str): "train" or "test" split.
        """
        super().__init__()
        if split not in ["train", "test"]:
            raise ValueError("split must be 'train' or 'test'")
        if not isinstance(upscale, int) or upscale <= 0:
            raise ValueError("upscale must be a positive integer")

        self.root = root
        self.split = split
        self.upscale = upscale
        self.normalize = normalize
        self.files = sorted(glob(os.path.join(root, split, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy files found in {os.path.join(root, split)}")

        # Normalization constants (u, v)
        self.mean = np.array([-1.65791551e-08, 2.04728862e-09]).reshape(1, 1, 2)
        self.std = np.array([0.80159812, 0.75345966]).reshape(1, 1, 2)

        # --- Load all data into memory ---
        print(f"Loading all {len(self.files)} files for split='{split}' into memory...")
        self.data = [np.load(f, mmap_mode=None) for f in self.files]  # full load

        # --- Build index of (file_id, t_idx) pairs ---
        self.samples = []
        for file_id, arr in enumerate(self.data):
            num_timesteps = arr.shape[0]
            # For SR, each timestep is a valid sample
            for i in range(num_timesteps):
                self.samples.append((file_id, i))

        print(f"Total samples in '{split}' split: {len(self.samples)}")

    @property
    def n_channels(self) -> int:
        return 2

    @property
    def img_resolution(self) -> Tuple[int, int]:
        """Returns the high-resolution (target) image resolution."""
        return (256, 256)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def _denormalize(self, x: np.ndarray) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        # Auto-detect if it's (C, H, W) and convert to (H, W, C)
        if x.shape[0] == self.n_channels and x.ndim == 3:
            x = rearrange(x, 'c h w -> h w c')
        return x * self.std + self.mean

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_id, t_idx = self.samples[idx]
        
        # Get the single, high-resolution frame (H, W, C)
        arr_hr = self.data[file_id][t_idx]

        # Create the low-resolution input by downsampling (H/s, W/s, C)
        arr_lr = arr_hr[::self.upscale, ::self.upscale, :]

        # Normalize both
        if self.normalize:
            y_np = self._normalize(arr_hr) # Target (High-Res)
            x_np = self._normalize(arr_lr) # Input (Low-Res)
        else:
            x_np = arr_lr
            y_np = arr_hr
        # Convert to PyTorch (C, H, W)
        lr = torch.from_numpy(rearrange(x_np, 'h w c -> c h w').astype(np.float32))
        hr = torch.from_numpy(rearrange(y_np, 'h w c -> c h w').astype(np.float32))

        return lr, hr


if __name__ == '__main__':
    DATA_ROOT = "/global/cfs/projectdirs/m5021/dj/data/dns_256_re_1e3_dt_0.4487989505128276"
    UPSCALE_FACTOR = 8

    print("\nInstantiating Dataset and DataLoader...")
    train_dataset = TurbulenceSRDataset(root=DATA_ROOT, upscale=UPSCALE_FACTOR, split="train")
    test_dataset = TurbulenceSRDataset(root=DATA_ROOT, upscale=UPSCALE_FACTOR, split="test")

    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)

    x_batch, y_batch = next(iter(train_dataloader))
    print(f"\n--- Batch Shapes (upscale={UPSCALE_FACTOR}) ---")
    print(f"x_batch shape (low-res):  {x_batch.shape}, dtype: {x_batch.dtype}")
    print(f"y_batch shape (high-res): {y_batch.shape}, dtype: {y_batch.dtype}")

    # Check resolution
    expected_lr_dim = train_dataset.img_resolution[0] // UPSCALE_FACTOR
    assert x_batch.shape[2] == expected_lr_dim
    assert x_batch.shape[3] == expected_lr_dim
    assert y_batch.shape[2] == train_dataset.img_resolution[0]
    assert y_batch.shape[3] == train_dataset.img_resolution[1]
    
    print("\nShape check passed!")