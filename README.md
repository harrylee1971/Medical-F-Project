# DCED — Brain MRI Super-Resolution (PyTorch 實作)

完整對應論文：
> Du et al., "Brain MRI Super-Resolution Using 3D Dilated Convolutional Encoder–Decoder Network", IEEE Access 2020

---

## 檔案結構

```
dced_mri/
├── data_preprocessing.py   # 資料載入、LR 生成、patch 萃取
├── model.py                 # DCED 模型架構
├── wavelet_fusion.py        # 3D Haar Wavelet + Self-Ensemble 推論
├── train.py                 # 訓練 & 推論主程式
└── requirements.txt
```

---

## 安裝

```bash
pip install -r requirements.txt
```

---

## 資料準備

將 KKI2009 MPRAGE .nii 檔案放入同一資料夾：

```
data/
├── KKI2009-01-MPRAGE.nii
├── KKI2009-02-MPRAGE.nii
...
└── KKI2009-42-MPRAGE.nii
```

論文分割：
- **訓練集**: KKI13 ~ KKI42 (30 筆)
- **測試集**: KKI01 ~ KKI05 ( 5 筆)

---

## 訓練

```bash
python train.py train \
  --data_dir ./data \
  --save_dir ./checkpoints \
  --epochs 80 \
  --batch_size 32 \
  --lr 1e-4 \
  --scale 2
```

| 參數 | 說明 | 論文值 |
|------|------|--------|
| `--epochs` | 訓練回合數 | 80 |
| `--batch_size` | mini-batch 大小 | 32 |
| `--lr` | Adam 學習率 | 1e-4 |
| `--weight_decay` | Adam weight decay | 1e-5 |
| `--scale` | 超解析度倍率 | 2 |
| `--patch_size` | 訓練 patch 邊長 | 31 |
| `--overlap` | patch 重疊像素 | 16 |

---

## 推論（含 3D Wavelet Fusion）

```bash
python train.py infer \
  --checkpoint ./checkpoints/dced_best.pth \
  --input_dir ./data \
  --output_dir ./results \
  --scale 2
```

輸出為 `*_SR.nii.gz`，可用 FSLeyes / ITK-SNAP 檢視。

---

## 模型架構摘要

```
Input (B, 1, 31, 31, 31)
  ↓ Layer1: Conv3d(1→64, 3x3x3, d=1) + ReLU
  ↓ Layer2: Conv3d(64→32, 3x3x3, d=1) + ReLU
  ↓ EDB1, EDB2, EDB3（各含 3 dilated encoder + 3 deconv decoder + LSC）
  ↓ LC: Concat(layer2, edb1, edb2, edb3) → Conv3d(128→1)
  ↓ Residual Add: + Input
Output (B, 1, 31, 31, 31)
```

---

## 3D Wavelet Fusion 流程

```
LR volume
  ├─ 旋轉  0° → DCED → 逆旋轉 → SR_0
  ├─ 旋轉 90° → DCED → 逆旋轉 → SR_1
  ├─ 旋轉180° → DCED → 逆旋轉 → SR_2
  └─ 旋轉270° → DCED → 逆旋轉 → SR_3
        ↓
   3D Haar DWT（每個 SR 分解為 8 個 subband）
        ↓
   HHH subband → max-absolute fusion (R1)
   其他 subband → 加權平均 w=[0.6,0.1,0.2,0.1] (R2)
        ↓
   Inverse Haar WT → Final HR MRI
```

---

## 評估指標

| 指標 | 說明 | 越高/低越好 |
|------|------|------------|
| PSNR (dB) | 峰值信噪比 | 越高越好 |
| SSIM | 結構相似度 | 越高越好 |
| NRMSE | 正規化均方根誤差 | 越低越好 |

---

## 論文結果對照（Kirby21，scale=2）

| 方法 | PSNR (dB) | SSIM | NRMSE |
|------|-----------|------|-------|
| Cubic | 33.35 | 0.9217 | 0.2383 |
| NLM | 34.19 | 0.9349 | 0.2162 |
| LRTV | 35.93 | 0.9574 | 0.1770 |
| SRCNN3D | 37.51 | 0.9735 | 0.1477 |
| ReCNN | 38.80 | 0.9797 | 0.1279 |
| DDSR | 39.00 | 0.9811 | 0.1244 |
| **DCED (ours)** | **39.28** | **0.9841** | **0.1208** |
