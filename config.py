"""
config.py
=========
所有參數集中在這裡修改，不需要 cmd 參數。
"""

# ── 路徑設定 ──────────────────────────────────
DATA_DIR    = r"C:\Users\user\Downloads\Medical-F-Project\Data\output_folder"   # .nii 檔案所在資料夾
SAVE_DIR    = r"C:\Users\user\Downloads\Medical-F-Project\Data\Save"
OUTPUT_DIR  = r"C:\Users\user\Downloads\Medical-F-Project\Data\Output"

# ── 資料集分割 ────────────────────────────────
TRAIN_IDS   = list(range(13, 43))   # KKI13 ~ KKI42 (30 筆訓練)
TEST_IDS    = list(range(1, 6))     # KKI01 ~ KKI05  (5 筆測試)

# ── 資料前處理 ────────────────────────────────
SCALE       = 2       # 超解析度倍率
PATCH_SIZE  = 31      # 3D patch 邊長
OVERLAP     = 8      # patch 重疊像素

# ── 模型架構 ──────────────────────────────────
IN_CHANNELS = 1       # MRI 單通道
NUM_EDBS    = 3       # Encoder-Decoder Block 數量
NUM_FILTERS = 32      # 每層 filter 數

# ── 訓練超參數 ────────────────────────────────
EPOCHS      = 80
BATCH_SIZE  = 64
LR          = 1e-4    # Adam 學習率
WEIGHT_DECAY= 1e-5    # Adam weight decay
MOMENTUM    = 0.9

# ── Wavelet Fusion 權重 (論文值) ──────────────
WAVELET_WEIGHTS = [0.6, 0.1, 0.2, 0.1]

# ── 其他 ──────────────────────────────────────
EVAL_EVERY  = 40       # 每幾個 epoch 做一次評估
SAVE_EVERY  = 20      # 每幾個 epoch 存一次 checkpoint
