"""
wavelet_fusion.py
=================
論文 Section III-D：Geometric Self-Ensemble + 3D Haar Wavelet Fusion

修正：
  - 旋轉邊界填充改為 reflect（消除邊緣不連續）
  - 逆旋轉後裁回原始尺寸（消除尺寸偏移）
  - Fusion 後用腦部 mask 蓋掉邊界條紋
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import rotate as scipy_rotate


# ─────────────────────────────────────────────
# 幾何旋轉
# ─────────────────────────────────────────────

def rotate_volume(vol: np.ndarray, angle: int, axes=(0, 1)) -> np.ndarray:
    if angle == 0:
        return vol.copy()
    # reflect 填充消除邊界不連續，order=1 雙線性
    return scipy_rotate(vol, angle=angle, axes=axes,
                        reshape=False, order=1, mode='reflect')


# ─────────────────────────────────────────────
# 3D Haar Wavelet Transform / Inverse
# ─────────────────────────────────────────────

def _pad_to_even(vol: np.ndarray):
    """填補使每個維度為偶數，回傳 (padded_vol, original_shape)"""
    orig = vol.shape
    pads = [(0, s % 2) for s in orig]
    if any(p[1] for p in pads):
        vol = np.pad(vol, pads, mode='reflect')
    return vol, orig


def haar_3d(vol: np.ndarray):
    """
    一階 3D Haar 小波分解。
    自動 reflect-pad 使維度為偶數，不裁切資料。
    回傳 (subbands_dict, original_shape)
    """
    vol, orig_shape = _pad_to_even(vol.astype(np.float64))

    def split(x, axis):
        slc_e = [slice(None)] * x.ndim
        slc_o = [slice(None)] * x.ndim
        slc_e[axis] = slice(None, None, 2)
        slc_o[axis] = slice(1,    None, 2)
        lo = (x[tuple(slc_e)] + x[tuple(slc_o)]) / np.sqrt(2)
        hi = (x[tuple(slc_e)] - x[tuple(slc_o)]) / np.sqrt(2)
        return lo, hi

    L0, H0 = split(vol, 0)
    LL, LH = split(L0,  1)
    HL, HH = split(H0,  1)

    LLL, LLH = split(LL, 2)
    LHL, LHH = split(LH, 2)
    HLL, HLH = split(HL, 2)
    HHL, HHH = split(HH, 2)

    subbands = {
        'LLL': LLL, 'LLH': LLH, 'LHL': LHL, 'LHH': LHH,
        'HLL': HLL, 'HLH': HLH, 'HHL': HHL, 'HHH': HHH
    }
    return subbands, orig_shape


def ihaar_3d(subbands: dict, orig_shape: tuple) -> np.ndarray:
    """
    一階 3D Haar 逆變換，裁回 orig_shape。
    """
    def merge(lo, hi, axis):
        shape = list(lo.shape)
        shape[axis] *= 2
        out = np.empty(shape, dtype=lo.dtype)
        slc_e = [slice(None)] * lo.ndim
        slc_o = [slice(None)] * lo.ndim
        slc_e[axis] = slice(None, None, 2)
        slc_o[axis] = slice(1,    None, 2)
        out[tuple(slc_e)] = (lo + hi) / np.sqrt(2)
        out[tuple(slc_o)] = (lo - hi) / np.sqrt(2)
        return out

    LL = merge(subbands['LLL'], subbands['LLH'], 2)
    LH = merge(subbands['LHL'], subbands['LHH'], 2)
    HL = merge(subbands['HLL'], subbands['HLH'], 2)
    HH = merge(subbands['HHL'], subbands['HHH'], 2)

    L0 = merge(LL, LH, 1)
    H0 = merge(HL, HH, 1)
    v  = merge(L0, H0, 0)

    # 裁回原始尺寸（去掉 pad）
    s = tuple(slice(0, n) for n in orig_shape)
    return v[s].astype(np.float32)


# ─────────────────────────────────────────────
# 融合函數
# ─────────────────────────────────────────────

def wavelet_fusion(volumes: list, weights: list = None) -> np.ndarray:
    """
    對多個 SR volume 做 3D Haar Wavelet Fusion。
    """
    n = len(volumes)
    if weights is None:
        weights = [0.6, 0.1, 0.2, 0.1][:n]
    assert len(weights) == n

    # 所有 volume 對齊到最小公共尺寸
    min_shape = tuple(min(v.shape[i] for v in volumes) for i in range(3))
    volumes = [v[:min_shape[0], :min_shape[1], :min_shape[2]] for v in volumes]

    all_subbands = []
    for v in volumes:
        sb, orig = haar_3d(v)
        all_subbands.append(sb)

    # 融合
    fused = {}
    for key in all_subbands[0]:
        # 對齊每個 subband 的尺寸
        sub_min = tuple(min(sb[key].shape[i] for sb in all_subbands) for i in range(3))
        stacked = np.stack(
            [sb[key][:sub_min[0], :sub_min[1], :sub_min[2]] for sb in all_subbands],
            axis=0
        )
        if key == 'HHH':
            # R1: max-absolute
            idx = np.argmax(np.abs(stacked), axis=0)
            fused[key] = np.take_along_axis(stacked, idx[np.newaxis], axis=0)[0]
        else:
            # R2: 加權平均
            w_arr = np.array(weights, dtype=np.float64).reshape(n, 1, 1, 1)
            fused[key] = np.sum(stacked * w_arr, axis=0)

    result = ihaar_3d(fused, min_shape)
    return result


# ─────────────────────────────────────────────
# 完整 Self-Ensemble + Wavelet Fusion 推論
# ─────────────────────────────────────────────

@torch.no_grad()
def self_ensemble_inference(model: nn.Module,
                             lr_volume: np.ndarray,
                             device: torch.device,
                             rotation_axis: tuple = (0, 1),
                             weights: list = None) -> np.ndarray:
    model.eval()
    angles = [0, 90, 180, 270]
    orig_shape = lr_volume.shape
    sr_results = []

    for angle in angles:
        # 旋轉 LR（reflect 填充）
        lr_rot = rotate_volume(lr_volume, angle, axes=rotation_axis)

        # DCED 推論
        lr_t  = torch.from_numpy(lr_rot).unsqueeze(0).unsqueeze(0).to(device)
        sr_np = model(lr_t).squeeze().cpu().numpy()

        # 逆旋轉並裁回原始尺寸
        sr_aligned = rotate_volume(sr_np, -angle, axes=rotation_axis)
        sr_aligned = sr_aligned[:orig_shape[0], :orig_shape[1], :orig_shape[2]]
        sr_results.append(sr_aligned)

    # Wavelet fusion
    fused = wavelet_fusion(sr_results, weights=weights)

    # 腦部 mask：把 LR 幾乎為 0 的背景區域還原成 LR 值，消除邊界條紋
    fused = _apply_brain_mask(fused, lr_volume, orig_shape)

    return fused


def _apply_brain_mask(fused: np.ndarray,
                      lr_volume: np.ndarray,
                      orig_shape: tuple,
                      threshold: float = 0.02) -> np.ndarray:
    """
    背景區域（LR < threshold）直接用 LR 值取代 wavelet 輸出，
    消除邊界 artifact 條紋。
    """
    s = tuple(slice(0, min(a, b)) for a, b in zip(fused.shape, orig_shape))
    out = fused.copy()
    lr_crop = lr_volume[:fused.shape[0], :fused.shape[1], :fused.shape[2]]
    mask = lr_crop < threshold          # True = 背景
    out[mask] = lr_crop[mask]           # 背景直接用 LR，不用 wavelet
    return out


# ─────────────────────────────────────────────
# 快速測試
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 測試 3D Haar Wavelet ===")
    vol = np.random.rand(65, 65, 65).astype(np.float32)   # 故意用奇數測試 pad
    subbands, orig = haar_3d(vol)
    rec = ihaar_3d(subbands, orig)
    err = np.mean(np.abs(vol - rec))
    print(f"  重建誤差（應接近 0）: {err:.2e}")

    print("\n=== 測試 Wavelet Fusion ===")
    vols = [np.random.rand(65, 65, 65).astype(np.float32) for _ in range(4)]
    fused = wavelet_fusion(vols)
    print(f"  輸入: {vols[0].shape} × 4 → 輸出: {fused.shape}")
