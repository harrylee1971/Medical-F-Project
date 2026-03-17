"""
data_preprocessing.py
=====================
載入 KKI2009 MPRAGE .nii 檔案，產生 LR/HR 3D patch pairs。

論文設定：
  - 訓練集: KKI13 ~ KKI42  (30 筆)
  - 測試集: KKI01 ~ KKI05  ( 5 筆)
  - LR 生成: 等向性下採樣 (scale factor=2) + Gaussian kernel (sigma=1)
  - patch size: 31x31x31, overlap: 16
"""

import os
import glob
import numpy as np
import nibabel as nib
from scipy.ndimage import gaussian_filter, zoom
from torch.utils.data import Dataset, DataLoader
import torch


# ─────────────────────────────────────────────
# 工具函數
# ─────────────────────────────────────────────

def load_nii(path: str) -> np.ndarray:
    """載入 .nii / .nii.gz，回傳 float32 numpy array"""
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data


def normalize(volume: np.ndarray) -> np.ndarray:
    """Min-Max 正規化到 [0, 1]"""
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin < 1e-8:
        return np.zeros_like(volume)
    return (volume - vmin) / (vmax - vmin)


def generate_lr(hr: np.ndarray, scale: int = 2, sigma: float = 1.0) -> np.ndarray:
    """
    模擬低解析度影像：
      1. 高斯模糊
      2. 等向性下採樣 (1/scale)
      3. 三次插值上採樣回原始大小 (論文輸入為 cubic upsampled LR)
    """
    blurred = gaussian_filter(hr, sigma=sigma)
    lr_small = zoom(blurred, zoom=1.0 / scale, order=3)          # 下採樣
    lr_up = zoom(lr_small, zoom=float(scale), order=3)            # cubic 上採樣
    # 確保尺寸與 HR 一致
    lr_up = _match_shape(lr_up, hr.shape)
    return lr_up.astype(np.float32)


def _match_shape(arr: np.ndarray, target_shape: tuple) -> np.ndarray:
    """裁切或填補使 arr 形狀與 target_shape 相同"""
    result = np.zeros(target_shape, dtype=arr.dtype)
    slices = tuple(slice(0, min(s, t)) for s, t in zip(arr.shape, target_shape))
    result[slices] = arr[slices]
    return result


def extract_patches(lr: np.ndarray,
                    hr: np.ndarray,
                    patch_size: int = 31,
                    overlap: int = 16) -> list:
    """
    從一對 (LR, HR) volume 中萃取所有 3D patch pairs。
    回傳 list of (lr_patch, hr_patch)，每個 patch shape=(patch_size,)*3
    """
    stride = patch_size - overlap
    D, H, W = lr.shape
    patches = []

    for d in range(0, D - patch_size + 1, stride):
        for h in range(0, H - patch_size + 1, stride):
            for w in range(0, W - patch_size + 1, stride):
                lr_p = lr[d:d+patch_size, h:h+patch_size, w:w+patch_size]
                hr_p = hr[d:d+patch_size, h:h+patch_size, w:w+patch_size]
                patches.append((lr_p.copy(), hr_p.copy()))

    return patches


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class KKI2009Dataset(Dataset):
    """
    KKI2009 MPRAGE Dataset

    資料夾結構建議：
        data_dir/
            KKI2009-01-MPRAGE.nii
            KKI2009-02-MPRAGE.nii
            ...
            KKI2009-42-MPRAGE.nii

    參數
    ----
    data_dir  : 存放所有 .nii 檔案的資料夾
    subject_ids : 要使用的受試者編號列表，e.g. list(range(13, 43)) 為訓練集
    scale     : 下採樣倍率 (論文=2)
    patch_size: patch 邊長 (論文=31)
    overlap   : patch 重疊像素 (論文=16)
    """

    def __init__(self,
                 data_dir: str,
                 subject_ids: list,
                 scale: int = 2,
                 patch_size: int = 31,
                 overlap: int = 16):
        super().__init__()
        self.patches = []

        for sid in subject_ids:
            # 尋找檔案，支援 01/1 兩種補零格式
            pattern = os.path.join(data_dir, f"KKI2009-{sid:02d}-MPRAGE.nii*")
            files = glob.glob(pattern)
            if not files:
                print(f"[Warning] 找不到 subject {sid:02d}，跳過")
                continue

            path = files[0]
            print(f"  載入 {os.path.basename(path)} ...")
            hr = normalize(load_nii(path))
            lr = generate_lr(hr, scale=scale)
            subject_patches = extract_patches(lr, hr, patch_size, overlap)
            self.patches.extend(subject_patches)
            print(f"    → {len(subject_patches)} patches")

        print(f"\n共 {len(self.patches)} 個 patch pairs")

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        lr_p, hr_p = self.patches[idx]
        # shape: (1, D, H, W) — 單通道 3D
        lr_t = torch.from_numpy(lr_p).unsqueeze(0)
        hr_t = torch.from_numpy(hr_p).unsqueeze(0)
        return lr_t, hr_t


# ─────────────────────────────────────────────
# 建立 DataLoader 的便利函數
# ─────────────────────────────────────────────

def get_dataloaders(data_dir: str,
                    batch_size: int = 32,
                    num_workers: int = 4,
                    scale: int = 2,
                    patch_size: int = 31,
                    overlap: int = 16):
    """
    回傳 (train_loader, test_loader)
    訓練集: KKI13~42, 測試集: KKI01~05
    """
    train_ids = list(range(13, 43))   # 30 筆
    test_ids  = list(range(1, 6))     #  5 筆

    print("=== 建立訓練集 ===")
    train_ds = KKI2009Dataset(data_dir, train_ids, scale, patch_size, overlap)
    print("\n=== 建立測試集 ===")
    test_ds  = KKI2009Dataset(data_dir, test_ids,  scale, patch_size, overlap)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,
                              batch_size=1,
                              shuffle=False,
                              num_workers=num_workers,
                              pin_memory=True)
    return train_loader, test_loader


if __name__ == "__main__":
    # 快速測試
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    train_loader, test_loader = get_dataloaders(data_dir, batch_size=4)
    lr_b, hr_b = next(iter(train_loader))
    print(f"LR batch: {lr_b.shape}, HR batch: {hr_b.shape}")
