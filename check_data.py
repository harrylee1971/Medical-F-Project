"""
check_data.py
=============
執行此腳本確認 LR/HR 的品質差異是否正確。
直接執行：python check_data.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')   # 不需要視窗，存成圖片
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.ndimage import gaussian_filter, zoom

import config as cfg
from data_preprocessing import normalize, generate_lr

# ── 載入一個 subject ──────────────────────────
import os, glob
candidates = (glob.glob(os.path.join(cfg.DATA_DIR, "KKI2009-13-MPRAGE.nii")) +
              glob.glob(os.path.join(cfg.DATA_DIR, "KKI2009-13-MPRAGE.nii.gz")))
if not candidates:
    raise FileNotFoundError(f"找不到 KKI2009-13，請確認 DATA_DIR={cfg.DATA_DIR}")

path = candidates[0]
print(f"載入: {path}")
hr = normalize(nib.load(path).get_fdata(dtype=np.float32))
lr = generate_lr(hr, scale=cfg.SCALE, sigma=1.0)

print(f"HR shape : {hr.shape}")
print(f"LR shape : {lr.shape}")
print(f"HR range : {hr.min():.4f} ~ {hr.max():.4f}")
print(f"LR range : {lr.min():.4f} ~ {lr.max():.4f}")

# ── 計算差異 ──────────────────────────────────
diff = np.abs(hr - lr)
print(f"\nHR vs LR 差異：")
print(f"  Mean Absolute Error : {diff.mean():.6f}")
print(f"  Max  Absolute Error : {diff.max():.6f}")
print(f"  PSNR (LR vs HR)    : ", end="")

import math
mse = np.mean((hr - lr) ** 2)
psnr = 20 * math.log10(1.0 / math.sqrt(mse)) if mse > 1e-10 else 999
print(f"{psnr:.2f} dB  ← 這個數字應該比你的訓練結果低！")

# ── 取中間切面視覺化 ──────────────────────────
mid = hr.shape[2] // 2   # axial 中間層

fig, axes = plt.subplots(1, 4, figsize=(18, 5))

axes[0].imshow(hr[:, :, mid], cmap='gray', vmin=0, vmax=1)
axes[0].set_title(f'HR (Ground Truth)\nslice {mid}')
axes[0].axis('off')

axes[1].imshow(lr[:, :, mid], cmap='gray', vmin=0, vmax=1)
axes[1].set_title(f'LR (cubic upsampled)\nslice {mid}')
axes[1].axis('off')

axes[2].imshow(diff[:, :, mid], cmap='hot', vmin=0, vmax=0.1)
axes[2].set_title(f'Difference (HR - LR)\nmax={diff[:,:,mid].max():.4f}')
axes[2].axis('off')

# 放大局部區域比較細節
h, w = hr.shape[:2]
crop = slice(h//4, 3*h//4), slice(w//4, 3*w//4)
axes[3].imshow(hr[crop[0], crop[1], mid] - lr[crop[0], crop[1], mid],
               cmap='bwr', vmin=-0.05, vmax=0.05)
axes[3].set_title('Zoomed Difference (blue=LR brighter)')
axes[3].axis('off')

plt.suptitle(f'LR/HR 品質確認  |  PSNR(LR)={psnr:.2f} dB  |  MAE={diff.mean():.4f}',
             fontsize=13)
plt.tight_layout()
out = 'C:\\Users\\user\\Downloads\\Medical-F-Project\\check_lr_hr.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\n圖片已儲存：{out}")
print("請確認 LR 比 HR 模糊/損失細節 —— 若兩者幾乎一樣則 scale 或 sigma 設定有問題")
