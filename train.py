"""
train.py
========
DCED 完整訓練 + 評估流程

使用方式：
  python train.py --data_dir ./data --epochs 80 --batch_size 32

評估指標：PSNR, SSIM, NRMSE（對應論文）
"""

import os
import argparse
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import nibabel as nib
from skimage.metrics import structural_similarity as ssim_fn

from data_preprocessing import get_dataloaders, KKI2009Dataset, normalize, generate_lr
from model import DCED
from wavelet_fusion import self_ensemble_inference


# ─────────────────────────────────────────────
# 評估指標
# ─────────────────────────────────────────────

def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))   # 影像已正規化到 [0,1]


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return ssim_fn(pred, target, data_range=1.0)


def nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    rmse = math.sqrt(np.mean((pred - target) ** 2))
    norm = math.sqrt(np.mean(target ** 2))
    return rmse / (norm + 1e-8)


# ─────────────────────────────────────────────
# 單 Epoch 訓練
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
# Patch-based 評估（用 patch loader）
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_patches(model, loader, device):
    model.eval()
    psnr_list, ssim_list, nrmse_list = [], [], []

    for lr_b, hr_b in loader:
        lr_b = lr_b.to(device)
        pred_b = model(lr_b).cpu().numpy()
        hr_b   = hr_b.numpy()

        for pred, gt in zip(pred_b, hr_b):
            pred = pred.squeeze().clip(0, 1)
            gt   = gt.squeeze()
            psnr_list.append(psnr(pred, gt))
            ssim_list.append(ssim(pred, gt))
            nrmse_list.append(nrmse(pred, gt))

    return (np.mean(psnr_list),
            np.mean(ssim_list),
            np.mean(nrmse_list))


# ─────────────────────────────────────────────
# Volume-level 評估（完整 MRI）
# ─────────────────────────────────────────────

@torch.no_grad()
def evaluate_volume(model, data_dir, subject_ids, device,
                    scale=2, use_wavelet=False):
    """對每個受試者做整體 volume 評估"""
    import glob
    model.eval()
    results = []

    for sid in subject_ids:
        pattern = os.path.join(data_dir, f"KKI2009-{sid:02d}-MPRAGE.nii*")
        files = glob.glob(pattern)
        if not files:
            continue

        hr_np = normalize(nib.load(files[0]).get_fdata(dtype=np.float32))
        lr_np = generate_lr(hr_np, scale=scale)

        if use_wavelet:
            pred_np = self_ensemble_inference(model, lr_np, device)
        else:
            lr_t = torch.from_numpy(lr_np).unsqueeze(0).unsqueeze(0).to(device)
            pred_t = model(lr_t)
            pred_np = pred_t.squeeze().cpu().numpy()

        # 裁切到相同尺寸
        min_shape = tuple(min(a, b) for a, b in zip(pred_np.shape, hr_np.shape))
        s = tuple(slice(0, m) for m in min_shape)
        pred_np = pred_np[s].clip(0, 1)
        hr_np   = hr_np[s]

        p = psnr(pred_np, hr_np)
        s_ = ssim(pred_np, hr_np)
        n = nrmse(pred_np, hr_np)
        results.append((sid, p, s_, n))
        print(f"  KKI{sid:02d}  PSNR={p:.2f} dB  SSIM={s_:.4f}  NRMSE={n:.4f}")

    avg_p  = np.mean([r[1] for r in results])
    avg_s  = np.mean([r[2] for r in results])
    avg_n  = np.mean([r[3] for r in results])
    print(f"\n  平均   PSNR={avg_p:.2f} dB  SSIM={avg_s:.4f}  NRMSE={avg_n:.4f}")
    return avg_p, avg_s, avg_n


# ─────────────────────────────────────────────
# 主訓練流程
# ─────────────────────────────────────────────

def train(args):
    # ── 裝置 ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 資料載入 ──
    print("\n載入資料集...")
    train_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scale=args.scale,
        patch_size=args.patch_size,
        overlap=args.overlap,
    )

    # ── 模型 ──
    model = DCED(in_channels=1, num_edbs=3, num_filters=32).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型參數量: {total_params:,}")

    # ── 損失函數：MSE（EuclideanLoss）──
    criterion = nn.MSELoss()

    # ── 優化器：Adam，lr=1e-4，weight_decay=1e-5 ──
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.weight_decay,
                           betas=(0.9, 0.999))

    # ── 儲存目錄 ──
    os.makedirs(args.save_dir, exist_ok=True)
    best_psnr = 0.0

    # ── 訓練迴圈 ──
    print(f"\n開始訓練（共 {args.epochs} epochs）...\n")
    for epoch in range(1, args.epochs + 1):
        # 訓練
        avg_loss = train_one_epoch(model, train_loader,
                                   optimizer, criterion, device, epoch)

        # 每5 epoch 做 patch-level 評估
        if epoch % 5 == 0 or epoch == args.epochs:
            p, s, n = evaluate_patches(model, test_loader, device)
            print(f"\nEpoch {epoch}/{args.epochs}  "
                  f"train_loss={avg_loss:.6f}  "
                  f"PSNR={p:.2f}dB  SSIM={s:.4f}  NRMSE={n:.4f}")

            # 儲存最佳模型
            if p > best_psnr:
                best_psnr = p
                ckpt_path = os.path.join(args.save_dir, "dced_best.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'psnr': p,
                }, ckpt_path)
                print(f"  ★ 新最佳 PSNR={p:.2f} dB，儲存至 {ckpt_path}")
        else:
            print(f"Epoch {epoch}/{args.epochs}  train_loss={avg_loss:.6f}")

        # 每20 epoch 儲存 checkpoint
        if epoch % 20 == 0:
            ckpt_path = os.path.join(args.save_dir, f"dced_epoch{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }, ckpt_path)

    # ── 最終 Volume-level 評估（含 Wavelet Fusion）──
    print("\n=== 最終 Volume-level 評估（不含 Wavelet Fusion）===")
    evaluate_volume(model, args.data_dir, list(range(1, 6)), device,
                    scale=args.scale, use_wavelet=False)

    print("\n=== 最終 Volume-level 評估（含 3D Wavelet Fusion）===")
    evaluate_volume(model, args.data_dir, list(range(1, 6)), device,
                    scale=args.scale, use_wavelet=True)


# ─────────────────────────────────────────────
# 推論腳本（載入已訓練模型）
# ─────────────────────────────────────────────

def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入模型
    model = DCED(in_channels=1, num_edbs=3, num_filters=32).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"載入 checkpoint: {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")

    # 載入並推論
    import glob
    nii_files = sorted(glob.glob(os.path.join(args.input_dir, "*.nii*")))
    os.makedirs(args.output_dir, exist_ok=True)

    for fpath in nii_files:
        fname = os.path.basename(fpath)
        print(f"推論: {fname} ...")

        img = nib.load(fpath)
        hr_np = normalize(img.get_fdata(dtype=np.float32))
        lr_np = generate_lr(hr_np, scale=args.scale)

        sr_np = self_ensemble_inference(model, lr_np, device)

        # 儲存為 .nii.gz
        out_img = nib.Nifti1Image(sr_np, img.affine, img.header)
        out_path = os.path.join(args.output_dir, fname.replace('.nii', '_SR.nii'))
        nib.save(out_img, out_path)
        print(f"  儲存: {out_path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="DCED MRI Super-Resolution")
    subparsers = parser.add_subparsers(dest="mode")

    # ── train ──
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--data_dir",    type=str, required=True)
    train_p.add_argument("--save_dir",    type=str, default="./checkpoints")
    train_p.add_argument("--epochs",      type=int, default=80)
    train_p.add_argument("--batch_size",  type=int, default=32)
    train_p.add_argument("--num_workers", type=int, default=4)
    train_p.add_argument("--lr",          type=float, default=1e-4)
    train_p.add_argument("--weight_decay",type=float, default=1e-5)
    train_p.add_argument("--scale",       type=int, default=2)
    train_p.add_argument("--patch_size",  type=int, default=31)
    train_p.add_argument("--overlap",     type=int, default=16)

    # ── infer ──
    infer_p = subparsers.add_parser("infer")
    infer_p.add_argument("--checkpoint",  type=str, required=True)
    infer_p.add_argument("--input_dir",   type=str, required=True)
    infer_p.add_argument("--output_dir",  type=str, default="./sr_output")
    infer_p.add_argument("--scale",       type=int, default=2)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)
    else:
        print("請指定 mode: train 或 infer")
        print("範例：")
        print("  python train.py train --data_dir ./data --epochs 80")
        print("  python train.py infer --checkpoint ./checkpoints/dced_best.pth "
              "--input_dir ./data --output_dir ./results")
