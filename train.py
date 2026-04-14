"""
train.py
========
直接執行：python train.py
所有參數在 config.py 修改。

評估方式：每個 subject 做整張 volume 評估，再取平均（與論文 Table 3 一致）
"""

import os
import time
import math
import glob
import platform

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import nibabel as nib
from skimage.metrics import structural_similarity as ssim_fn

import config as cfg
from data_preprocessing import KKI2009Dataset, normalize, generate_lr
from model import DCED
from wavelet_fusion import self_ensemble_inference


# ─────────────────────────────────────────────
# 評估指標
# ─────────────────────────────────────────────

def psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))

def ssim(pred, target):
    return ssim_fn(pred, target, data_range=1.0)

def nrmse(pred, target):
    rmse = math.sqrt(np.mean((pred - target) ** 2))
    norm = math.sqrt(np.mean(target ** 2))
    return rmse / (norm + 1e-8)


# ─────────────────────────────────────────────
# 訓練一個 Epoch
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for i, (lr_b, hr_b) in enumerate(loader):
        lr_b = lr_b.to(device, non_blocking=True)
        hr_b = hr_b.to(device, non_blocking=True)

        optimizer.zero_grad()
        pred = model(lr_b)
        loss = criterion(pred, hr_b)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 100 == 0:
            avg = total_loss / (i + 1)
            elapsed = time.time() - t0
            print(f"  [Epoch {epoch}] step {i+1}/{len(loader)}  "
                  f"loss={avg:.6f}  elapsed={elapsed:.1f}s")

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# Volume-level 評估（每個 subject 分開，論文做法）
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_per_subject(model, device, use_wavelet=False):
    """
    對每個測試 subject 做整張 volume 的評估，最後取平均。

    為什麼不用 patch-level：
      patch 含大量腦外黑色背景，預測黑色很容易，
      PSNR 會虛高到 45+ dB，NRMSE 也會爆掉。
      整張 volume 評估才是論文 Table 3 的做法。
    """
    model.eval()
    results = []

    for sid in cfg.TEST_IDS:
        candidates = [
            os.path.join(cfg.DATA_DIR, f"KKI2009-{sid:02d}-MPRAGE.nii"),
            os.path.join(cfg.DATA_DIR, f"KKI2009-{sid:02d}-MPRAGE.nii.gz"),
        ]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            print(f"    [Skip] KKI{sid:02d} 找不到")
            continue

        hr_np = normalize(nib.load(path).get_fdata(dtype=np.float32))
        lr_np = generate_lr(hr_np, scale=cfg.SCALE)

        if use_wavelet:
            pred_np = self_ensemble_inference(
                model, lr_np, device, weights=cfg.WAVELET_WEIGHTS)
        else:
            lr_t = torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0).to(device)
            pred_np = model(lr_t).squeeze().cpu().numpy()

        s = tuple(slice(0, min(a, b)) for a, b in zip(pred_np.shape, hr_np.shape))
        pred_np = pred_np[s].clip(0, 1)
        hr_np   = hr_np[s]

        p  = psnr(pred_np, hr_np)
        s_ = ssim(pred_np, hr_np)
        n  = nrmse(pred_np, hr_np)
        results.append((sid, p, s_, n))
        print(f"    KKI{sid:02d}  PSNR={p:.2f} dB  SSIM={s_:.4f}  NRMSE={n:.4f}")

    if not results:
        return 0.0, 0.0, 0.0

    avg_p = np.mean([r[1] for r in results])
    avg_s = np.mean([r[2] for r in results])
    avg_n = np.mean([r[3] for r in results])
    print(f"    {'─'*50}")
    print(f"    平均   PSNR={avg_p:.2f} dB  SSIM={avg_s:.4f}  NRMSE={avg_n:.4f}")
    return avg_p, avg_s, avg_n


# ─────────────────────────────────────────────
# 主程式
# ─────────────────────────────────────────────

def main():
    # ── 裝置 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 訓練集（用 patch 訓練）──
    print("\n=== 建立訓練集 ===")
    train_ds = KKI2009Dataset(cfg.DATA_DIR, cfg.TRAIN_IDS,
                               cfg.SCALE, cfg.PATCH_SIZE, cfg.OVERLAP)
    if len(train_ds) == 0:
        raise RuntimeError(f"訓練集為空！請確認 DATA_DIR：{cfg.DATA_DIR}")

    nw = 0 if platform.system() == "Windows" else 4
    train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE,
                              shuffle=True, num_workers=nw)

    # ── 模型 ──
    model = DCED(cfg.IN_CHANNELS, cfg.NUM_EDBS, cfg.NUM_FILTERS).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型參數量: {total_params:,}")

    # ── 損失 & 優化器 ──
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=cfg.LR,
                           weight_decay=cfg.WEIGHT_DECAY,
                           betas=(cfg.MOMENTUM, 0.999))

    os.makedirs(cfg.SAVE_DIR, exist_ok=True)
    best_psnr = 0.0

    # ── 訓練迴圈 ──
    print(f"\n開始訓練（共 {cfg.EPOCHS} epochs）...\n")
    for epoch in range(1, cfg.EPOCHS + 1):

        avg_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch)

        # 每 EVAL_EVERY 個 epoch 做 volume-level per-subject 評估
        if epoch % cfg.EVAL_EVERY == 0 or epoch == cfg.EPOCHS:
            print(f"\nEpoch {epoch}/{cfg.EPOCHS}  train_loss={avg_loss:.6f}")
            print(f"  >> Volume-level 評估（KKI01~05 各自計算）:")
            avg_p, avg_s, avg_n = evaluate_per_subject(
                model, device, use_wavelet=False)

            if avg_p > best_psnr:
                best_psnr = avg_p
                ckpt = os.path.join(cfg.SAVE_DIR, "dced_best.pth")
                torch.save({'epoch': epoch,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'psnr': avg_p}, ckpt)
                print(f"  ★ 新最佳 PSNR={avg_p:.2f} dB → 儲存至 {ckpt}\n")
            else:
                print()
        else:
            print(f"Epoch {epoch}/{cfg.EPOCHS}  train_loss={avg_loss:.6f}")

        if epoch % cfg.SAVE_EVERY == 0:
            ckpt = os.path.join(cfg.SAVE_DIR, f"dced_epoch{epoch}.pth")
            torch.save({'epoch': epoch,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict()}, ckpt)

    # ── 最終評估 ──
    print("\n" + "="*60)
    print("最終評估結果（對應論文 Table 3）")
    print("="*60)

    print("\n[不含 Wavelet Fusion]")
    evaluate_per_subject(model, device, use_wavelet=False)

    print("\n[含 3D Wavelet Fusion]")
    evaluate_per_subject(model, device, use_wavelet=True)


# ─────────────────────────────────────────────
# 推論模式
# ─────────────────────────────────────────────

def infer(checkpoint_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCED(cfg.IN_CHANNELS, cfg.NUM_EDBS, cfg.NUM_FILTERS).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"載入 checkpoint: {checkpoint_path}  (epoch {ckpt.get('epoch','?')})")

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    nii_files = (glob.glob(os.path.join(cfg.DATA_DIR, "*.nii")) +
                 glob.glob(os.path.join(cfg.DATA_DIR, "*.nii.gz")))

    for fpath in sorted(nii_files):
        fname = os.path.basename(fpath)
        print(f"推論: {fname} ...")
        img   = nib.load(fpath)
        hr_np = normalize(img.get_fdata(dtype=np.float32))
        lr_np = generate_lr(hr_np, scale=cfg.SCALE)
        sr_np = self_ensemble_inference(
            model, lr_np, device, weights=cfg.WAVELET_WEIGHTS)
        out_path = os.path.join(cfg.OUTPUT_DIR,
                                fname.replace('.nii', '_SR.nii'))
        nib.save(nib.Nifti1Image(sr_np, img.affine, img.header), out_path)
        print(f"  → 儲存: {out_path}")


if __name__ == "__main__":
    # main()
    # 訓練完要推論時，把上面 main() 改成：
    # infer(r"C:\Users\user\Downloads\Medical-F-Project\Data\Save\dced_best.pth")
    
    # main()  # 訓練

    # 只跑最終評估
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    model = DCED(cfg.IN_CHANNELS, cfg.NUM_EDBS, cfg.NUM_FILTERS).to(device)
    ckpt = torch.load(os.path.join(cfg.SAVE_DIR, "dced_best.pth"), map_location=device)
    model.load_state_dict(ckpt['model_state'])
    print(f"載入 checkpoint (epoch {ckpt.get('epoch','?')}, PSNR={ckpt.get('psnr',0):.2f} dB)")

    print("\n" + "="*60)
    print("最終評估結果（對應論文 Table 3）")
    print("="*60)

    print("\n[不含 Wavelet Fusion]")
    evaluate_per_subject(model, device, use_wavelet=False)

    print("\n[含 3D Wavelet Fusion]")
    evaluate_per_subject(model, device, use_wavelet=True)
