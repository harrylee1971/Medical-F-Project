"""
test_inference.py
=================
讀取 3 張 MRI，分別儲存：
  1. LR 影像   → lr/
  2. HR 影像   → hr/   (DCED 模型輸出)
  3. Wavelet   → wav/  (3D Haar Wavelet Fusion 輸出)

每個結果存成 .nii.gz + 中間切面 .png 方便目視比較。

直接執行：python test_inference.py
"""

import os
import numpy as np
import torch
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config as cfg
from data_preprocessing import normalize, generate_lr
from model import DCED
from wavelet_fusion import self_ensemble_inference

# ─────────────────────────────────────────────
# 設定：要測試的三個 subject（從測試集選）
# ─────────────────────────────────────────────
TEST_SUBJECTS = [1, 2, 3]   # KKI01, KKI02, KKI03

CHECKPOINT = os.path.join(cfg.SAVE_DIR, "dced_best.pth")

OUT_ROOT = os.path.join(cfg.OUTPUT_DIR, "test_results")
DIR_LR   = os.path.join(OUT_ROOT, "lr")
DIR_HR   = os.path.join(OUT_ROOT, "hr")
DIR_WAV  = os.path.join(OUT_ROOT, "wavelet")
DIR_CMP  = os.path.join(OUT_ROOT, "compare")

for d in [DIR_LR, DIR_HR, DIR_WAV, DIR_CMP]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# 儲存工具
# ─────────────────────────────────────────────

def save_nii(volume: np.ndarray, affine, header, path: str):
    nib.save(nib.Nifti1Image(volume.astype(np.float32), affine, header), path)
    print(f"  [nii] {path}")


def save_png(volume: np.ndarray, path: str, title: str = ""):
    """儲存 axial / coronal / sagittal 三個切面"""
    d, h, w = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('black')

    slices = [
        (volume[d//2, :, :], f"Axial  (z={d//2})"),
        (volume[:, h//2, :], f"Coronal (y={h//2})"),
        (volume[:, :, w//2], f"Sagittal (x={w//2})"),
    ]
    for ax, (sl, label) in zip(axes, slices):
        ax.imshow(sl.T, cmap='gray', origin='lower', vmin=0, vmax=1)
        ax.set_title(label, color='white', fontsize=10)
        ax.axis('off')

    fig.suptitle(title, color='white', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  [png] {path}")


def save_compare_png(lr, hr, wav, path: str, sid: int):
    """四格對比圖：HR(GT) | LR | DCED HR | Wavelet HR"""
    d = lr.shape[0] // 2
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor('black')

    panels = [
        (lr[d],  f"LR (degraded)\nKKI{sid:02d}"),
        (hr[d],  f"DCED HR output\nKKI{sid:02d}"),
        (wav[d], f"Wavelet fusion\nKKI{sid:02d}"),
    ]
    for i, (img, label) in enumerate(panels):
        axes[i].imshow(img, cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(label, color='white', fontsize=10)
        axes[i].axis('off')

    # Difference: Wavelet - LR
    diff = np.abs(wav[d] - lr[d])
    im = axes[3].imshow(diff, cmap='hot', vmin=0, vmax=0.15)
    axes[3].set_title(f"Diff (Wavelet - LR)\nKKI{sid:02d}", color='white', fontsize=10)
    axes[3].axis('off')
    plt.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches='tight', facecolor='black')
    plt.close()
    print(f"  [compare] {path}")


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device}")

    # 載入模型
    if not os.path.isfile(CHECKPOINT):
        raise FileNotFoundError(
            f"找不到 checkpoint：{CHECKPOINT}\n"
            "請先完成訓練，或修改 CHECKPOINT 路徑。"
        )
    model = DCED(cfg.IN_CHANNELS, cfg.NUM_EDBS, cfg.NUM_FILTERS).to(device)
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"載入 checkpoint (epoch {ckpt.get('epoch','?')},  PSNR={ckpt.get('psnr',0):.2f} dB)\n")

    for sid in TEST_SUBJECTS:
        print(f"{'='*50}")
        print(f"處理 KKI{sid:02d} ...")

        # 找 .nii 檔
        candidates = [
            os.path.join(cfg.DATA_DIR, f"KKI2009-{sid:02d}-MPRAGE.nii"),
            os.path.join(cfg.DATA_DIR, f"KKI2009-{sid:02d}-MPRAGE.nii.gz"),
        ]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            print(f"  [Skip] 找不到 KKI{sid:02d}")
            continue

        # 載入 HR，生成 LR
        img    = nib.load(path)
        hr_np  = normalize(img.get_fdata(dtype=np.float32))
        lr_np  = generate_lr(hr_np, scale=cfg.SCALE)
        affine = img.affine
        header = img.header

        print(f"  volume shape: {hr_np.shape}")

        # ── 1. 儲存 LR ──
        print("  >> 儲存 LR ...")
        save_nii(lr_np, affine, header,
                 os.path.join(DIR_LR, f"KKI{sid:02d}_LR.nii.gz"))
        save_png(lr_np,
                 os.path.join(DIR_LR, f"KKI{sid:02d}_LR.png"),
                 title=f"KKI{sid:02d} — LR (cubic upsampled, scale={cfg.SCALE})")

        # ── 2. DCED 推論 → HR ──
        print("  >> DCED 推論 ...")
        with torch.no_grad():
            lr_t   = torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0).to(device)
            hr_pred = model(lr_t).squeeze().cpu().numpy().clip(0, 1)

        s = tuple(slice(0, min(a, b)) for a, b in zip(hr_pred.shape, hr_np.shape))
        hr_pred = hr_pred[s]

        save_nii(hr_pred, affine, header,
                 os.path.join(DIR_HR, f"KKI{sid:02d}_HR.nii.gz"))
        save_png(hr_pred,
                 os.path.join(DIR_HR, f"KKI{sid:02d}_HR.png"),
                 title=f"KKI{sid:02d} — DCED HR output")

        # ── 3. Wavelet Fusion ──
        print("  >> Wavelet Fusion ...")
        wav_pred = self_ensemble_inference(
            model, lr_np, device, weights=cfg.WAVELET_WEIGHTS)
        wav_pred = wav_pred.clip(0, 1)

        s2 = tuple(slice(0, min(a, b)) for a, b in zip(wav_pred.shape, hr_np.shape))
        wav_pred = wav_pred[s2]

        save_nii(wav_pred, affine, header,
                 os.path.join(DIR_WAV, f"KKI{sid:02d}_wavelet.nii.gz"))
        save_png(wav_pred,
                 os.path.join(DIR_WAV, f"KKI{sid:02d}_wavelet.png"),
                 title=f"KKI{sid:02d} — Wavelet Fusion HR")

        # ── 4. 對比圖 ──
        print("  >> 儲存對比圖 ...")
        save_compare_png(
            lr_np, hr_pred, wav_pred,
            os.path.join(DIR_CMP, f"KKI{sid:02d}_compare.png"),
            sid=sid
        )

        print(f"  KKI{sid:02d} 完成\n")

    print(f"{'='*50}")
    print(f"全部完成！結果儲存於：{OUT_ROOT}")
    print(f"  LR 影像    → {DIR_LR}")
    print(f"  HR 影像    → {DIR_HR}")
    print(f"  Wavelet    → {DIR_WAV}")
    print(f"  對比圖     → {DIR_CMP}")


if __name__ == "__main__":
    main()
