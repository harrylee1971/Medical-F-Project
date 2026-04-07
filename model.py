"""
model.py
========
3D Dilated Convolutional Encoder-Decoder (DCED) Network
完整對應論文 Table 2 的架構：

  Input (31x31x31x1)
  └─ Layer1: Conv3d 3x3x3, 64 filters, d=1
  └─ Layer2: Conv3d 3x3x3, 32 filters, d=1
  └─ EDB1 ~ EDB3 (各6層: 3 dilated encoder + 3 deconv decoder)
      ├─ Layer3:  Conv+ReLU   3x3x3, 32f, d=2
      ├─ Layer4:  Conv+ReLU   3x3x3, 32f, d=2
      ├─ Layer5:  Conv+ReLU   3x3x3, 32f, d=2
      ├─ Layer6:  DeConv+ReLU 3x3x3, 32f, s=1  (LSC from Layer5)
      ├─ Layer7:  DeConv+ReLU 3x3x3, 32f, s=1  (LSC from Layer4)
      └─ Layer8:  Conv+ReLU   3x3x3, 32f, d=1  (LSC from Layer3)
  └─ Layer21: Conv 1x1x1, 1 filter  (concat all EDB outputs → LC)
  └─ Addition: Layer21 + Interpolated LR  (residual)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 基礎積木
# ─────────────────────────────────────────────

def conv3d(in_ch, out_ch, dilation=1, bias=True):
    """3x3x3 dilated convolution，保持空間尺寸"""
    pad = dilation  # same padding for 3x3x3 kernel
    return nn.Conv3d(in_ch, out_ch,
                     kernel_size=3, stride=1,
                     padding=pad, dilation=dilation,
                     bias=bias)


def deconv3d(in_ch, out_ch, bias=True):
    """
    3x3x3 transposed convolution (stride=1, padding=1)
    尺寸不變，用於 degridding
    """
    return nn.ConvTranspose3d(in_ch, out_ch,
                               kernel_size=3, stride=1,
                               padding=1, bias=bias)


# ─────────────────────────────────────────────
# Encoder-Decoder Block (EDB)
# ─────────────────────────────────────────────

class EDB(nn.Module):
    """
    一個 Encoder-Decoder Block：
      3 個 2-dilated encoder  +  3 個 deconv decoder
      每對 encoder-decoder 以 LSC (element-wise add) 連接
      最後輸出經 Conv 整合後供 LC 使用
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        # ── Encoders (2-dilated) ──
        self.enc1 = nn.Sequential(conv3d(channels, channels, dilation=2), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(conv3d(channels, channels, dilation=2), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(conv3d(channels, channels, dilation=2), nn.ReLU(inplace=True))
        self.enc4 = nn.Sequential(conv3d(channels, channels, dilation=2), nn.ReLU(inplace=True))

        # ── Decoders (deconv, stride=1) ──
        self.dec1 = nn.Sequential(deconv3d(channels, channels), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(deconv3d(channels, channels), nn.ReLU(inplace=True))
        # 最後 decoder 換回普通 conv (論文 Layer8: Conv+ReLU, d=1)
        self.dec3 = nn.Sequential(conv3d(channels, channels, dilation=1), nn.ReLU(inplace=True))

    def forward(self, x):
        # ── Encoding ──
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        # e3 = self.enc4(e3)

        # ── Decoding with LSC (element-wise addition) ──
        d1 = self.dec1(e3)      # 對應 e3
        d1 = d1 + e2            # LSC: dec1 ↔ enc2

        d2 = self.dec2(d1)
        d2 = d2 + e1            # LSC: dec2 ↔ enc1

        d3 = self.dec3(d2)
        d3 = d3 + x             # LSC: dec3 ↔ 輸入

        return d3               # 供 LC 使用


# ─────────────────────────────────────────────
# 完整 DCED 網路
# ─────────────────────────────────────────────

class DCED(nn.Module):
    """
    3D Dilated Convolutional Encoder-Decoder Network

    參數
    ----
    in_channels  : 輸入通道數 (MRI = 1)
    num_edbs     : EDB 數量 (論文 L=3)
    num_filters  : 每層 filter 數 (論文 = 32，前兩層第一層=64)
    """

    def __init__(self,
                 in_channels: int = 1,
                 num_edbs: int = 3,
                 num_filters: int = 32):
        super().__init__()
        self.num_edbs = num_edbs

        # ── 前兩層特徵提取 ──
        self.layer1 = nn.Sequential(
            conv3d(in_channels, 64, dilation=1),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            conv3d(64, num_filters, dilation=1),
            nn.ReLU(inplace=True)
        )

        # ── EDB 堆疊 ──
        self.edbs = nn.ModuleList([EDB(num_filters) for _ in range(num_edbs)])

        # ── LC 整合層 (concat 所有 EDB 輸出 + layer2 輸出) ──
        # 輸入通道 = (num_edbs + 1) * num_filters
        concat_ch = (num_edbs + 1) * num_filters
        self.lc_conv = nn.Conv3d(concat_ch, in_channels,
                                 kernel_size=3, stride=1,
                                 padding=1, bias=True)

    def forward(self, x_lr_up: torch.Tensor) -> torch.Tensor:
        """
        x_lr_up : cubic upsampled LR MRI, shape (B, 1, D, H, W)
        回傳     : HR MRI prediction,     shape (B, 1, D, H, W)
        """
        # ── 特徵提取 ──
        f1 = self.layer1(x_lr_up)
        f2 = self.layer2(f1)

        # ── EDB 堆疊，收集每個 EDB 的輸出 (用於 LC) ──
        edb_outputs = [f2]
        feat = f2
        for edb in self.edbs:
            feat = edb(feat)
            edb_outputs.append(feat)

        # ── LC: concat 所有 EDB 輸出 → 1x1x1-like conv → residual image ──
        concat = torch.cat(edb_outputs, dim=1)   # (B, (L+1)*C, D, H, W)
        residual = self.lc_conv(concat)           # (B, 1, D, H, W)

        # ── 殘差加法：HR = LR_up + residual ──
        hr_pred = x_lr_up + residual

        return hr_pred


# ─────────────────────────────────────────────
# 快速驗證
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DCED(in_channels=1, num_edbs=3, num_filters=32).to(device)

    # 計算參數量
    total = sum(p.numel() for p in model.parameters())
    print(f"模型參數量: {total:,}")

    # 前向測試
    dummy = torch.randn(2, 1, 31, 31, 31, device=device)
    out = model(dummy)
    print(f"輸入: {dummy.shape} → 輸出: {out.shape}")

    # 架構摘要 (需安裝 torchinfo)
    try:
        from torchinfo import summary
        summary(model, input_size=(1, 1, 31, 31, 31), device=device)
    except ImportError:
        pass
