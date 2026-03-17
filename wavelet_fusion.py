"""
wavelet_fusion.py
=================
論文 Section III-D：Geometric Self-Ensemble + 3D Haar Wavelet Fusion

流程：
  1. 原始 LR 沿某軸旋轉 0°, 90°, 180°, 270°
  2. 分別送入 DCED → 4 張 SR 結果
  3. 各自逆旋轉
  4. 用 3D Haar Wavelet Fusion 合併：
       HHH subband → max-absolute 融合  (R1)
       其他 subband → 加權平均融合      (R2，論文 w = [0.6, 0.1, 0.2, 0.1])
  5. Inverse Haar Wavelet → 最終 HR MRI
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import rotate as scipy_rotate


# ─────────────────────────────────────────────
# 幾何旋轉（沿 Z 軸，即 axial plane）
# ─────────────────────────────────────────────

def rotate_volume(vol: np.ndarray, angle: int, axes=(0, 1)) -> np.ndarray:
    """沿指定平面旋轉 volume (角度為 90 的倍數，使用最近鄰)"""
    if angle == 0:
        return vol
    return scipy_rotate(vol, angle=angle, axes=axes,
                        reshape=False, order=1, mode='nearest')


# ─────────────────────────────────────────────
# 3D Haar Wavelet Transform / Inverse
# ─────────────────────────────────────────────

def haar_3d(vol: np.ndarray):
    """
    一階 3D Haar 小波分解。
    vol shape: (D, H, W)，需為偶數維度（自動裁切）。
    回傳 dict：鍵為 'LLL', 'LLH', 'LHL', 'LHH', 'HLL', 'HLH', 'HHL', 'HHH'
    """
    # 確保偶數
    D, H, W = vol.shape
    D = D - D % 2
    H = H - H % 2
    W = W - W % 2
    v = vol[:D, :H, :W].astype(np.float64)

    def split(x, axis):
        slc_even = [slice(None)] * x.ndim
        slc_odd  = [slice(None)] * x.ndim
        slc_even[axis] = slice(None, None, 2)
        slc_odd[axis]  = slice(1,    None, 2)
        lo = (x[tuple(slc_even)] + x[tuple(slc_odd)]) / np.sqrt(2)
        hi = (x[tuple(slc_even)] - x[tuple(slc_odd)]) / np.sqrt(2)
        return lo, hi

    L0, H0 = split(v,  0)   # along D
    LL, LH = split(L0, 1)   # along H (L branch)
    HL, HH = split(H0, 1)   # along H (H branch)

    LLL, LLH = split(LL, 2)
    LHL, LHH = split(LH, 2)
    HLL, HLH = split(HL, 2)
    HHL, HHH = split(HH, 2)

    return {
        'LLL': LLL, 'LLH': LLH, 'LHL': LHL, 'LHH': LHH,
        'HLL': HLL, 'HLH': HLH, 'HHL': HHL, 'HHH': HHH
    }


def ihaar_3d(subbands: dict) -> np.ndarray:
    """
    一階 3D Haar 小波逆變換。
    輸入 dict 同 haar_3d 輸出，回傳重建後的 volume。
    """
    def merge(lo, hi, axis):
        shape = list(lo.shape)
        shape[axis] *= 2
        out = np.empty(shape, dtype=lo.dtype)
        slc_even = [slice(None)] * lo.ndim
        slc_odd  = [slice(None)] * lo.ndim
        slc_even[axis] = slice(None, None, 2)
        slc_odd[axis]  = slice(1,    None, 2)
        out[tuple(slc_even)] = (lo + hi) / np.sqrt(2)
        out[tuple(slc_odd)]  = (lo - hi) / np.sqrt(2)
        return out

    LLL = subbands['LLL']; LLH = subbands['LLH']
    LHL = subbands['LHL']; LHH = subbands['LHH']
    HLL = subbands['HLL']; HLH = subbands['HLH']
    HHL = subbands['HHL']; HHH = subbands['HHH']

    LL = merge(LLL, LLH, 2)
    LH = merge(LHL, LHH, 2)
    HL = merge(HLL, HLH, 2)
    HH = merge(HHL, HHH, 2)

    L0 = merge(LL, LH, 1)
    H0 = merge(HL, HH, 1)
    v  = merge(L0, H0, 0)
    return v.astype(np.float32)


# ─────────────────────────────────────────────
# 融合函數
# ─────────────────────────────────────────────

def wavelet_fusion(volumes: list,
                   weights: list = None) -> np.ndarray:
    """
    對多個 SR volume 做 3D Haar Wavelet Fusion。

    volumes : list of np.ndarray (D, H, W), 已完成逆旋轉
    weights : list of float (R2 融合權重，長度需與 volumes 一致)
              預設論文值 [0.6, 0.1, 0.2, 0.1]
    回傳     : 融合後的 volume (D, H, W)
    """
    n = len(volumes)
    if weights is None:
        weights = [0.6, 0.1, 0.2, 0.1][:n]
    assert len(weights) == n, "weights 長度需與 volumes 數量相同"

    # 1. 對每個 volume 做 Haar 分解
    all_subbands = [haar_3d(v) for v in volumes]

    # 取最小公共尺寸（因旋轉/裁切可能有1-pixel差異）
    ref_shape = {k: all_subbands[0][k].shape for k in all_subbands[0]}

    fused = {}
    for key in all_subbands[0]:
        stacked = np.stack([sb[key] for sb in all_subbands], axis=0)  # (n, d/2, h/2, w/2)

        if key == 'HHH':
            # R1: max-absolute fusion（保留最強的高頻分量）
            abs_stacked = np.abs(stacked)
            idx = np.argmax(abs_stacked, axis=0)           # (d/2, h/2, w/2)
            fused[key] = np.take_along_axis(
                stacked, idx[np.newaxis], axis=0)[0]
        else:
            # R2: 加權平均
            w_arr = np.array(weights, dtype=np.float64).reshape(n, 1, 1, 1)
            fused[key] = np.sum(stacked * w_arr, axis=0)

    # 2. Inverse wavelet
    result = ihaar_3d(fused)
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
    """
    對單一 LR volume 執行 self-ensemble + 3D wavelet fusion 推論。

    步驟：
      ① 對 LR 旋轉 0/90/180/270°
      ② 各自 DCED 推論 → 4 個 SR volume
      ③ 逆旋轉對齊
      ④ 3D Haar wavelet fusion

    參數
    ----
    model        : 已訓練好的 DCED（eval mode）
    lr_volume    : (D, H, W) float32 numpy，cubic upsampled LR
    device       : torch device
    rotation_axis: 旋轉平面，預設 (0,1) 即沿 Z 軸
    weights      : R2 融合權重，預設 [0.6, 0.1, 0.2, 0.1]

    回傳
    ----
    fused : (D, H, W) float32，最終 HR MRI
    """
    model.eval()
    angles = [0, 90, 180, 270]
    sr_results = []

    for angle in angles:
        # ① 旋轉 LR
        lr_rot = rotate_volume(lr_volume, angle, axes=rotation_axis)

        # ② 送入 DCED
        lr_t = torch.from_numpy(lr_rot).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
        sr_t = model(lr_t)
        sr_np = sr_t.squeeze().cpu().numpy()

        # ③ 逆旋轉（-angle）
        sr_aligned = rotate_volume(sr_np, -angle, axes=rotation_axis)
        sr_results.append(sr_aligned)

    # ④ Wavelet fusion
    fused = wavelet_fusion(sr_results, weights=weights)
    return fused


# ─────────────────────────────────────────────
# 快速測試
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 測試 3D Haar Wavelet ===")
    vol = np.random.rand(64, 64, 64).astype(np.float32)
    subbands = haar_3d(vol)
    rec = ihaar_3d(subbands)
    err = np.mean(np.abs(vol[:rec.shape[0], :rec.shape[1], :rec.shape[2]] - rec))
    print(f"  重建誤差 (應接近 0): {err:.2e}")

    print("\n=== 測試 Wavelet Fusion ===")
    vols = [np.random.rand(64, 64, 64).astype(np.float32) for _ in range(4)]
    fused = wavelet_fusion(vols)
    print(f"  輸入: {vols[0].shape} × 4 → 輸出: {fused.shape}")

    print("\n=== 測試 Self-Ensemble Inference ===")
    from model import DCED
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = DCED().to(device).eval()
    lr_vol = np.random.rand(64, 64, 64).astype(np.float32)
    result = self_ensemble_inference(m, lr_vol, device)
    print(f"  LR: {lr_vol.shape} → Fused HR: {result.shape}")
