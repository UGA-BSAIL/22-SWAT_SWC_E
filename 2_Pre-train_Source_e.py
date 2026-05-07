# ============================================================
# Source-site pretraining framework for SWAT-informed DL models
# ============================================================
# Purpose:
#   Train and evaluate LSTM, attention-enhanced LSTM, and Transformer
#   models for soil water prediction using climate-only and
#   climate +  hydrology input blocks.
#
# Notes for public/reproducible release:
#   - Path and experiment settings are grouped near the top (in 1. User settings).
# ============================================================

# ============================================================
# Pre-train LSTM variants across Feature Blocks
# (with attention options, seeds, per-config folders & exports)
# + Attention Diagnostics Integration (Entropy, STD, Gate, Pool)
# + Transformer integrated with the SAME pipeline (reproducible)
#
#
#   - SWAT-only simulation column (VSM2mm/mm) included in evaluation outputs
# ============================================================

import os, glob, shutil, random, re, math
from pathlib import Path
from typing import Tuple, Dict
import numpy as np
import pandas as pd

# ---------- Deep Learning
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# ---------- Metrics
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ---------- Plot defaults
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import seaborn as sns
import matplotlib.ticker as ticker

# keep figure style consistent across scripts (paper-ready defaults).
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
})

# ============================================================
# 1. User settings: paths, experiment name, and splits
# ============================================================
# I set the working directory explicitly so relative outputs are predictable.
# BASE_DIR = '/.../22-SWAT_SWC'
# new_directory = BASE_DIR
# os.chdir(new_directory)
# print("New Directory:", os.getcwd())

import os
from pathlib import Path
import yaml

# ============================================================
# Load path configuration
# ============================================================
CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

BASE_DIR = Path(config["base_dir"])

os.chdir(BASE_DIR)
print("New Directory:", os.getcwd())


# I use these tags to keep folder/file naming reproducible across runs + HPC/PC.
scenario_name = "output"
model = "model22"
step = "s10"
runs="R20"
N_RUNS        = 20

type="single"  # 

# I read all inputs from a single Excel workbook (sheet = training site ID).
FILE_PATH  = f"{BASE_DIR}/{scenario_name}/1-Inputs_{model}.xlsx"
SITE_TRAIN = f"DL_In_site31_Hru_116_{scenario_name}"

# I write everything into a block/model-specific folder under OUTPUT_DIR.
OUTPUT_DIR = f"1_Train_{type}_{step}_{model}_{runs}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# I hard-split train/val/test by date ranges (no random splitting).
SPLITS = {
    "train": ("2003-01-01", "2015-12-31"),  # 13 years
    "val":   ("2016-01-01", "2016-12-31"),   # 1 year
    "test":  ("2017-01-01", "2019-12-31"),    # 3 years
}

# I keep the core training hyperparameters here so it’s easy to compare experiments.
TIMESTEP      = 10
NUM_EPOCHS    = 100
BATCH_SIZE    = 32
LR            = 1e-3
NUM_LAYERS    = 2
HIDDEN_SIZE   = 64
DROPOUT       = 0.1
WEIGHT_DECAY  = 1e-6
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# I add seasonal sin/cos features (doy_sin/doy_cos) when ADD_SZN=True.
ADD_SZN       = True

# ============================================================
# --- Model selection options ---
# I can either auto-select top-k runs, or copy a fixed run list for reproducibility. and I use the second
#option 1
DO_SELECT_TOPK      = False
TOP_K               = 10

#option 2
USE_CUSTOM_RUN_LIST = True
CUSTOM_RUN_LIST     = [1, 2, 3,4,5,6,7,8,9,10,11, 12, 13,14,15,16,17,18,19,20]  # 

# ============================================================
# Reproducibility
# ============================================================
def set_global_seed(seed: int):
    # I enforce deterministic behavior so run-to-run results are traceable.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# I log seeds once at startup to confirm deterministic tracking.
for run in range(1, N_RUNS + 1):
    set_global_seed(run)
    print(f"[Seed log] Run {run}: torch = {torch.initial_seed()}, numpy = {np.random.get_state()[1][0]}")

def seed_workers(worker_id):
    # I seed dataloader workers so shuffling is consistent across GPUs/CPUs.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_loader(X, y, batch_size, shuffle, run_seed=42, num_workers=0,
                weights=None, month=None, wet=None):
    """
    Return DataLoader batches with input, target, and placeholder metadata.
    Regime/wet-dry analysis is not active in this cleaned source-site version.
    """
    g = torch.Generator()
    g.manual_seed(int(run_seed))

    X_t = torch.tensor(X)
    y_t = torch.tensor(y)

    # I default to uniform weights / dummy labels if not provided.
    if weights is None:
        weights = np.ones((len(X),), dtype=np.float32)
    if month is None:
        month = np.zeros((len(X),), dtype=np.int64)
    if wet is None:
        wet = np.zeros((len(X),), dtype=np.int64)

    w_t   = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)  # [N,1]
    m_t   = torch.tensor(month, dtype=torch.int64).unsqueeze(1)      # [N,1]
    wet_t = torch.tensor(wet, dtype=torch.int64).unsqueeze(1)        # [N,1]

    ds = TensorDataset(X_t, y_t, w_t, m_t, wet_t)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=seed_workers,
        generator=g,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

# ============================================================
# Feature Blocks
# ============================================================
# I define feature subsets ("blocks") to run ablation-style experiments.
FEATURE_BLOCKS = {
    "All": ['Rain', 'SOLARMJ/m2','TMP_MXdgC',
        'ETmm',
        'PERCmm','GW_RCHGmm',
        'SURQ_GENmm', 'LATQGENmm',
        'DAILYCN','SOL_TMPdgCS',
        'LAI','doy_sin','doy_cos',
    ],
     "Climate":  ['Rain', 'SOLARMJ/m2','TMP_MXdgC'],
}

# I keep a mapping for labels used in plots 
LABEL_MAP = {
 'PRECIPmm': 'Precipitation',
 'PETmm': 'Potential evapotranspiration',
 'ETmm': 'Evapotranspiration',
 'PERCmm': 'Percolation',
 'GW_RCHGmm': 'Groundwater recharge',
 'DA_RCHGmm': 'Deep aquifer recharge',
 'REVAPmm': 'Aquifer returning flow',
 'SURQ_GENmm': 'Surface flow',
 'LATQGENmm': 'Lateral flow',
 'WYLDmm': 'Water yield',
 'DAILYCN': 'Average curve number',
 'TMP_AVdgC': 'Avg air temperature',
 'TMP_MXdgC': 'Max air temperature',
 'SOL_TMPdgCS': 'Soil temperature',
 'SOLARMJ/m2': 'Solar radiation',
 'LAI': 'Leaf area index',
 'doy_sin': 'Seasonal (sin)',
 'doy_cos': 'Seasonal (cos)',
 'Rain':'Rainfall',
}

# 
TARGET_COL   = "mois_avg"  #target (observed soil moisture from the sensors)
SWAT_SWC_COL = "VSM2mm/mm"  # SWAT-only simulation column in the same sheet

# ============================================================
# 4. Data loading, scaling, and sequence preparation
# ============================================================
def load_and_split(file_path, sheet, add_szn=True):
    #  read one site sheet from Excel, sort by date, and create fixed date splits.
    df = pd.read_excel(file_path, sheet_name=sheet)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    if add_szn:
        # add cyclic day-of-year features so the model can learn seasonality .
        df['doy'] = df['Date'].dt.dayofyear
        df['doy_sin'] = np.sin(2*np.pi*df['doy']/365.25)
        df['doy_cos'] = np.cos(2*np.pi*df['doy']/365.25)
    dtr, dtr2 = SPLITS['train']
    dv1, dv2  = SPLITS['val']
    dte1, dte2= SPLITS['test']
    return (
        df[(df['Date']>=dtr)&(df['Date']<=dtr2)].copy(),
        df[(df['Date']>=dv1)&(df['Date']<=dv2)].copy(),
        df[(df['Date']>=dte1)&(df['Date']<=dte2)].copy()
    )

def fit_minmax(train_df, cols):
    #  fit scaling parameters ONLY on training data to avoid leakage.
    X_min = train_df[cols].min()
    X_max = train_df[cols].max()
    y_min = train_df[[TARGET_COL]].min()
    y_max = train_df[[TARGET_COL]].max()
    return X_min, X_max, y_min, y_max

def scale_xy(df, cols, X_min, X_max, y_min, y_max, clip_range=(-1.5,1.5)):
    #  scale inputs/target into [-1,1] (minmax-style), with optional clipping for stability.
    eps = 1e-12
    X = 2 * (df[cols] - X_min) / (X_max - X_min + eps) - 1
    y = 2 * (df[[TARGET_COL]] - y_min) / (y_max - y_min + eps) - 1
    if clip_range is not None:
        lo, hi = clip_range
        X = X.clip(lower=lo, upper=hi)
        y = y.clip(lower=lo, upper=hi)
    return X.values.astype(np.float32), y.values.astype(np.float32)

def make_sequences(X, y, T):
    #  convert the daily time series into supervised sequences:
    # X[t:t+T] -> y[t+T]
    Xs, ys = [], []
    for i in range(len(X)-T):
        Xs.append(X[i:i+T])
        ys.append(y[i+T])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def inverse_scale_y(scaled, y_min, y_max):
    #  invert the target scaling back to original soil moisture units.
    return (scaled + 1) * (y_max.values - y_min.values) / 2 + y_min.values

def dates_for_sequences(df_split, T):
    #  align the sequence targets with dates at index [T:].
    return df_split['Date'].iloc[T:].reset_index(drop=True)

# ============================================================
# 5. Model definitions
# ============================================================
class BaseHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        #  use a small MLP head for regression to keep the output layer consistent across models.
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
        )
    def forward(self, x):
        return self.head(x)
    
# Temporal attention over LSTM hidden states (learns which lags matter)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, temperature=0.5):
        super().__init__()
        #  compute attention scores over time steps and softmax them (temperature controls sharpness).
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.v   = nn.Linear(hidden_size, 1, bias=False)
        self.temperature = temperature

    def forward(self, H):
        scores = self.v(torch.tanh(self.W_h(H)))       # [B,T,1]
        attn_t = torch.softmax(scores / self.temperature, dim=1)
        ctx = (attn_t * H).sum(dim=1)
        return ctx, attn_t.squeeze(-1)
# Feature attention gate (learns feature-wise importance per sample)

class FeatureAttentionGate(nn.Module):
    def __init__(self, num_features, reduction=4, sigmoid_tau=0.7):
        super().__init__()
        #  implement a squeeze-and-excitation style gate to reweight input features.
        r = max(1, num_features // reduction)
        self.fc1 = nn.Sequential(
            nn.Linear(num_features, r),
            nn.BatchNorm1d(r),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(r, num_features)
        self.sigmoid_tau = sigmoid_tau

    def forward(self, x):
        z = x.mean(dim=1)
        w = self.fc2(self.fc1(z))
        w = torch.sigmoid(w / self.sigmoid_tau)
        x_scaled = x * w.unsqueeze(1)
        return x_scaled, w
    
# LSTM with temporal attention only

class LSTM_TemporalOnly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        #  apply temporal attention after the LSTM states (no feature gating here).
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.pre_attn_dropout = nn.Dropout(0.1)
        self.tattn = TemporalAttention(hidden_size, temperature=0.5)
        self.proj  = BaseHead(hidden_size)

    def forward(self, x, return_attn=False):
        H, _ = self.lstm(x)
        H = self.norm(H)
        H = self.pre_attn_dropout(H)
        ctx, attn_t = self.tattn(H)
        y = self.proj(ctx)
        if return_attn:
            return y, attn_t, None
        return y

# LSTM with feature gate + temporal attention
class LSTM_FeatureAndTemporal(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        #  combine feature gating at input + temporal attention at output (full attention model).
        self.fgate = FeatureAttentionGate(input_size, sigmoid_tau=0.7)
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.pre_attn_dropout = nn.Dropout(0.1)
        self.tattn = TemporalAttention(hidden_size, temperature=0.5)
        self.proj  = BaseHead(hidden_size)

    def forward(self, x, return_attn=False):
        x, feat_w = self.fgate(x)
        H, _ = self.lstm(x)
        H = self.norm(H)
        H = self.pre_attn_dropout(H)
        ctx, attn_t = self.tattn(H)
        y = self.proj(ctx)
        if return_attn:
            return y, attn_t, feat_w
        return y

# Plain LSTM baseline: uses last hidden state only

class LSTMPlain(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        #  keep the baseline model simple: LSTM + last hidden state + regression head.
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.proj = BaseHead(hidden_size)
    def forward(self, x, return_attn=False):
        H, _ = self.lstm(x)
        ctx = H[:, -1, :]
        y  = self.proj(ctx)
        if return_attn:
            return y, None, None
        return y

# LSTM with feature attention only: gate inputs, then last hidden
class LSTM_FeatureOnly(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        #  apply feature gating but no temporal attention (use last state).
        self.fgate = FeatureAttentionGate(input_size, sigmoid_tau=0.7)
        self.lstm  = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = BaseHead(hidden_size)

    def forward(self, x, return_attn=False):
        x, feat_w = self.fgate(x)
        H, _ = self.lstm(x)
        H = self.norm(H)
        ctx = H[:, -1, :]
        y = self.proj(ctx)
        if return_attn:
            return y, None, feat_w
        return y

# ============================================================
# 6. Transformer components
# ============================================================
# Transformer baseline uses positional encoding + TransformerEncoder + CLS pooling
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        #  use classic sinusoidal positional encoding for sequence order information.
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_cls: bool = True,
    ):
        super().__init__()
        #  keep Transformer settings aligned with LSTM defaults for fair comparisons.
        self.use_cls = use_cls
        self.seq_len = seq_len
        self.input_proj = nn.Linear(input_dim, d_model)
        self.dropout_in = nn.Dropout(dropout)

        max_len = seq_len + (1 if use_cls else 0)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        if use_cls:
            #  optionally learn a CLS token to summarize the whole sequence.
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x: torch.Tensor, return_attn: bool = False) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.dropout_in(x)
        if self.use_cls:
            B = x.size(0)
            cls = self.cls_token.expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)
        x = self.pos_enc(x)
        x = self.encoder(x)
        z = x[:, 0, :] if self.use_cls else x.mean(dim=1)
        out = self.head(z)
        if return_attn:
            return out, None, None
        return out

# ============================================================
# 7. Model registry
# ============================================================
#  keep a single registry so the experiment loop can run any architecture by name.
MODELS = {
    "LSTM_baseline":      LSTMPlain,
    "LSTM_temporalAttn":  LSTM_TemporalOnly,
    "LSTM_featureAttn":   LSTM_FeatureOnly,
    "LSTM_feat+tempAttn": LSTM_FeatureAndTemporal,
}
MODELS.update({
    "Transformer": lambda input_size, hidden_size, num_layers, dropout_rate:
        TransformerRegressor(
            input_dim=input_size,
            seq_len=TIMESTEP,
            d_model=HIDDEN_SIZE,
            n_heads=4,
            n_layers=NUM_LAYERS,
            d_ff=4*HIDDEN_SIZE,
            dropout=DROPOUT,
            use_cls=True,
        )
})

# ============================================================
# 8. Metrics and training helpers
# ============================================================
def compute_metrics(y_true, y_pred):
    #  compute common hydrology + ML metrics 
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    r = np.corrcoef(y_true, y_pred)[0,1] if len(y_true) > 1 else np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    nse  = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    pbias= 100 * np.sum(y_pred - y_true) / np.sum(y_true)
    std_true = np.std(y_true); mean_true = np.mean(y_true)
    alpha = np.std(y_pred)/std_true if std_true!=0 else np.nan
    beta  = np.mean(y_pred)/mean_true if mean_true!=0 else np.nan
    kge   = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    return {"R": r, "RMSE": rmse, "MAE": mae, "R2": r2, "NSE": nse, "PBIAS": pbias, "KGE": kge}

def train_one(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=20, lr=LR):
    #  use reduction='none' for compatibility with earlier versions; standard mean loss is used here.
    base_criterion = nn.SmoothL1Loss(beta=1.0, reduction="none")
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    #  use LR scheduler on EMA-smoothed val loss to reduce noise.
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, threshold=1e-4,
        threshold_mode='rel', cooldown=3, min_lr=1e-5
    )
    ema = None; alpha=0.2
    best_loss = float('inf'); best_state=None; wait=0
    train_losses=[]; val_losses=[]

    for epoch in range(num_epochs):
        model.train()
        run_train=0.0
        nb=0
        for xb, yb, wb, _, _ in train_loader:
            xb, yb, wb = xb.to(DEVICE), yb.to(DEVICE), wb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss_vec = base_criterion(pred, yb)  # [B,1]
            #  Standard SmoothL1 loss used in the submitted analysis.
            loss = loss_vec.mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run_train += loss.item()
            nb += 1
        train_losses.append(run_train / max(1, nb))

        model.eval(); v=0.0; nbv=0
        with torch.no_grad():
            for xb, yb, _, _, _ in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                v += base_criterion(pred, yb).mean().item()
                nbv += 1
        v = v / max(1, nbv)
        val_losses.append(v)

        #  smooth validation loss and use it to drive LR schedule.
        ema = v if ema is None else alpha*v + (1-alpha)*ema
        sched.step(ema)

        #  keep the best model based on raw val loss and stop early with patience.
        if v < best_loss:
            best_loss = v
            best_state = {k:p.cpu().clone() for k,p in model.state_dict().items()}
            wait=0
        else:
            wait+=1
            if wait>=patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses, ema

# ============================================================
# 9. Diagnostics and attention utilities
# ============================================================
def gather_attention_over_loader(model, loader):
    #  compute run-level mean attention for a quick overview per checkpoint.
    model.eval()
    all_temp, all_feat = [], []
    with torch.no_grad():
        for xb, _, _, _, _ in loader:
            xb = xb.to(DEVICE)
            _, attn_t, feat_w = model(xb, return_attn=True)
            if attn_t is not None:
                all_temp.append(attn_t.detach().cpu().numpy())
            if feat_w is not None:
                all_feat.append(feat_w.detach().cpu().numpy())
    avg_temp = None if not all_temp else np.concatenate(all_temp, axis=0).mean(axis=0)
    avg_feat = None if not all_feat else np.concatenate(all_feat, axis=0).mean(axis=0)
    return avg_temp, avg_feat

def diagnostics_attention(model, loader, variant_dir, timestep, cols):
    #  export interpretability diagnostics:
    #  - temporal attention entropy (sharp vs diffuse)
    #  - temporal attention std (peakedness)
    #  - feature gate distribution + raw values
    model.eval()
    entropy_vals, std_vals, gate_vals = [], [], []
    with torch.no_grad():
        for xb, _, _, _, _ in loader:
            xb = xb.to(DEVICE)
            _, attn_t, feat_w = model(xb, return_attn=True)
            if attn_t is not None:
                a = torch.clamp(attn_t, 1e-8, 1)
                H = -(a * a.log()).sum(dim=1)
                entropy_vals.extend(H.cpu().numpy())
                std_vals.extend(attn_t.std(dim=1).cpu().numpy())
            if feat_w is not None:
                gate_vals.extend(feat_w.cpu().numpy())

    results = {}
    if entropy_vals:
        results["MeanEntropy"] = float(np.mean(entropy_vals))
        results["MedianEntropy"] = float(np.median(entropy_vals))
        results["logT"] = float(np.log(timestep))
        results["MeanAttnSTD"] = float(np.mean(std_vals))
        pd.DataFrame({"Entropy": entropy_vals, "STD": std_vals})\
          .to_csv(os.path.join(variant_dir, "temporal_attention_diagnostics.csv"), index=False)

    if gate_vals:
        W = np.vstack(gate_vals)
        plt.figure(figsize=(6,4))
        plt.hist(W.flatten(), bins=30, color="gray", edgecolor="k")
        plt.axvline(0.5, ls="--", c="r"); plt.title("Feature Gate Weights")
        plt.xlabel("weight"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(variant_dir, "feature_gate_hist.png"), dpi=220)
        plt.close()
        pd.DataFrame(W, columns=[LABEL_MAP.get(c, c) for c in cols])\
          .to_csv(os.path.join(variant_dir, "feature_gate_values.csv"), index=False)
    return results

def compare_pooling_modes(model, loader, y_min, y_max, variant_dir, timestep):
    #  compare different pooling choices on LSTM states:
    # last vs mean vs attention pooling (helps justify architectural choices).
    if not hasattr(model, "lstm"):
        return {}
    device = next(model.parameters()).device
    def inverse(scaled): return (scaled + 1)*(y_max.values - y_min.values)/2 + y_min.values
    metrics_all = {}
    model.eval()
    with torch.no_grad():
        for mode in ["last","mean","attn"]:
            preds_all, obs_all = [], []
            for xb, yb, _, _, _ in loader:
                xb, yb = xb.to(device), yb.to(device)
                H, _ = model.lstm(xb)
                if mode == "last":
                    ctx = H[:,-1,:]
                elif mode == "mean":
                    ctx = H.mean(dim=1)
                elif mode == "attn" and hasattr(model, "tattn"):
                    ctx, _ = model.tattn(H)
                else:
                    ctx = H.mean(dim=1)
                yhat = model.proj(ctx).squeeze(-1).cpu().numpy()
                y_true = yb.cpu().numpy()
                preds_all.append(inverse(yhat[:,None])); obs_all.append(inverse(y_true))
            y_true = np.concatenate(obs_all).ravel(); y_pred = np.concatenate(preds_all).ravel()
            r = np.corrcoef(y_true, y_pred)[0,1] if len(y_true) > 1 else np.nan
            rmse = np.sqrt(mean_squared_error(y_true,y_pred))
            mae = mean_absolute_error(y_true,y_pred)
            r2 = r2_score(y_true,y_pred)
            metrics_all[mode] = {"R":float(r),"RMSE":float(rmse),"MAE":float(mae),"R2":float(r2)}
    pd.DataFrame(metrics_all).to_csv(os.path.join(variant_dir, f"pooling_compare_T{timestep}.csv"))
    return metrics_all


# ============================================================
# 10. Main runner for one feature-block × model configuration
# ============================================================
def run_config(block_name, cols, model_name):
    print(f"\n==== Block: {block_name} | Model: {model_name} ====")

    #  create the experiment folder structure as:
    # OUTPUT_DIR / block_name / model_name /
    block_dir   = os.path.join(OUTPUT_DIR, block_name)
    variant_dir = os.path.join(block_dir, model_name)
    os.makedirs(variant_dir, exist_ok=True)

    #  load and date-split data from the Excel sheet for the training site.
    df_tr, df_va, df_te = load_and_split(FILE_PATH, SITE_TRAIN, add_szn=ADD_SZN)

    #wet_threshold = None

    #  confirm feature availability early so errors are obvious.
    missing = [c for c in cols if c not in df_tr.columns]
    if missing:
        raise ValueError(f"Missing columns in data for block {block_name}: {missing}")

    #  fit scaling parameters on training only.
    Xmin, Xmax, ymin, ymax = fit_minmax(df_tr, cols)

    #  scale features and target consistently across train/val/test.
    Xtr, ytr = scale_xy(df_tr, cols, Xmin, Xmax, ymin, ymax)
    Xva, yva = scale_xy(df_va, cols, Xmin, Xmax, ymin, ymax)
    Xte, yte = scale_xy(df_te, cols, Xmin, Xmax, ymin, ymax)

    #  build supervised sequences using TIMESTEP.
    Xtr_s, ytr_s = make_sequences(Xtr, ytr, TIMESTEP)
    Xva_s, yva_s = make_sequences(Xva, yva, TIMESTEP)
    Xte_s, yte_s = make_sequences(Xte, yte, TIMESTEP)

    #  align dates with sequence targets.
    dates_train = dates_for_sequences(df_tr, TIMESTEP)
    dates_val   = dates_for_sequences(df_va, TIMESTEP)
    dates_test  = dates_for_sequences(df_te, TIMESTEP)

    # pull SWAT soil moisture (baseline simulation) aligned to the same target dates.
    if SWAT_SWC_COL not in df_tr.columns:
        raise ValueError(f"SWAT_SWC_COL='{SWAT_SWC_COL}' not found in sheet columns.")
    swat_train = df_tr[SWAT_SWC_COL].iloc[TIMESTEP:].reset_index(drop=True).values
    swat_val   = df_va[SWAT_SWC_COL].iloc[TIMESTEP:].reset_index(drop=True).values
    swat_test  = df_te[SWAT_SWC_COL].iloc[TIMESTEP:].reset_index(drop=True).values

    # Dummy metadata keeps the DataLoader output structure unchanged.
    month_tr = month_va = month_te = None
    wet_tr = wet_va = wet_te = None
    w_tr = w_va = w_te = None

    #  prepare dicts to store predictions for all runs + SWAT baseline in one workbook.
    metrics_all = []
    train_preds_all = {
        "Date": dates_train,
        "Observed": inverse_scale_y(ytr_s, ymin, ymax).flatten(),
        f"SWAT_{SWAT_SWC_COL}": swat_train
    }
    val_preds_all = {
        "Date": dates_val,
        "Observed": inverse_scale_y(yva_s, ymin, ymax).flatten(),
        f"SWAT_{SWAT_SWC_COL}": swat_val
    }
    test_preds_all = {
        "Date": dates_test,
        "Observed": inverse_scale_y(yte_s, ymin, ymax).flatten(),
        f"SWAT_{SWAT_SWC_COL}": swat_test
    }

    temporal_by_run = {}
    feature_by_run  = {}

    ModelClass = MODELS[model_name]

    # ============================
    # RUN LOOP
    # ============================
    for run in range(1, N_RUNS+1):
        #  reseed per run so each run is reproducible and distinct.
        set_global_seed(run)

        train_loader = make_loader(Xtr_s, ytr_s, BATCH_SIZE, shuffle=True,  run_seed=run)
        val_loader   = make_loader(Xva_s, yva_s, BATCH_SIZE, shuffle=False, run_seed=run)
        test_loader  = make_loader(Xte_s, yte_s, BATCH_SIZE, shuffle=False, run_seed=run)

        #  instantiate the chosen architecture with a consistent hidden size/layers config.
        model = ModelClass(input_size=len(cols), hidden_size=HIDDEN_SIZE,
                           num_layers=NUM_LAYERS, dropout_rate=DROPOUT).to(DEVICE)

        #  train with early stopping on validation loss and keep the best checkpoint.
        model, tr_losses, va_losses, _ = train_one(model, train_loader, val_loader)

        #  save per-epoch loss curves to an Excel file for debugging/appendix.
        try:
            xlsxpath = os.path.join(variant_dir, f"loss_{model_name}_run_{run:03d}.xlsx")
            max_len = max(len(tr_losses), len(va_losses))
            df_loss = pd.DataFrame({
                "Epoch": np.arange(1, max_len+1),
                "Train_Loss": tr_losses + [None]*(max_len-len(tr_losses)),
                "Val_Loss":   va_losses + [None]*(max_len-len(va_losses)),
            })
            df_loss.to_excel(xlsxpath, index=False)
        except Exception as e:
            print("Could not save loss table:", e)

        #  save the best checkpoint per run so I can reuse it later (TL/forecasting/SHAP).
        ckpt = os.path.join(variant_dir, f"best_{model_name}_Run{run:03d}.pt")
        torch.save(model.state_dict(), ckpt)
        print(f"[{block_name}/{model_name}] Saved best model -> {ckpt}")

        #  evaluate by running inference on full train/val/test sequences.
        model.eval()
        with torch.no_grad():
            yhat_tr = model(torch.tensor(Xtr_s).to(DEVICE)).cpu().numpy()
            yhat_va = model(torch.tensor(Xva_s).to(DEVICE)).cpu().numpy()
            yhat_te = model(torch.tensor(Xte_s).to(DEVICE)).cpu().numpy()

        #  invert scaling to report performance in original units.
        tr_pred = inverse_scale_y(yhat_tr, ymin, ymax).flatten()
        va_pred = inverse_scale_y(yhat_va, ymin, ymax).flatten()
        te_pred = inverse_scale_y(yhat_te, ymin, ymax).flatten()

        #  store each run’s predictions as a separate column in the output workbook.
        train_preds_all[f"Pred_Run{run:03d}"] = tr_pred
        val_preds_all[f"Pred_Run{run:03d}"]   = va_pred
        test_preds_all[f"Pred_Run{run:03d}"]  = te_pred

        # Metrics vs observed (I compute for each split).
        ytr_obs = train_preds_all["Observed"]
        yva_obs = val_preds_all["Observed"]
        yte_obs = test_preds_all["Observed"]

        metrics_all += [
            {"Run": run, "Set": "Train", **compute_metrics(ytr_obs, tr_pred)},
            {"Run": run, "Set": "Val",   **compute_metrics(yva_obs, va_pred)},
            {"Run": run, "Set": "Test",  **compute_metrics(yte_obs, te_pred)},
        ]

        #  also keep a simple mean attention per run for quick summaries/plots.
        try:
            _ = model(torch.tensor(Xte_s[:1]).to(DEVICE), return_attn=True)  # 
            avg_temp, avg_feat = gather_attention_over_loader(model, test_loader)
            if avg_temp is not None:
                temporal_by_run[f"Run{run:03d}"] = avg_temp
                pd.DataFrame({"LagIndex": np.arange(len(avg_temp)), "Weight": avg_temp})\
                  .to_csv(os.path.join(variant_dir, f"temporal_attn_Run{run:03d}.csv"), index=False)
            if avg_feat is not None:
                feat_cols = [LABEL_MAP.get(c, c) for c in cols]
                df_feat = pd.DataFrame({"Feature": feat_cols, "Weight": avg_feat})
                feature_by_run[f"Run{run:03d}"] = avg_feat
                df_feat.to_csv(os.path.join(variant_dir, f"feature_attn_Run{run:03d}.csv"), index=False)
        except Exception:
            #  silently skip if the model doesn’t support return_attn().
            pass

    # ============================
    # ADD SWAT METRICS ONCE (OUTSIDE run loop)
    # ============================
    #  compute SWAT baseline metrics once per split (not per run).
    metrics_all += [
        {"Run": "SWAT", "Set": "Train",
         **compute_metrics(train_preds_all["Observed"], train_preds_all[f"SWAT_{SWAT_SWC_COL}"])},
        {"Run": "SWAT", "Set": "Val",
         **compute_metrics(val_preds_all["Observed"],   val_preds_all[f"SWAT_{SWAT_SWC_COL}"])},
        {"Run": "SWAT", "Set": "Test",
         **compute_metrics(test_preds_all["Observed"],  test_preds_all[f"SWAT_{SWAT_SWC_COL}"])},
    ]


    #  save a single Excel workbook containing predictions + metrics for this (block, model).
    pretrain_xlsx = os.path.join(variant_dir, f"pre_trained_{model_name}.xlsx")
    with pd.ExcelWriter(pretrain_xlsx, engine="openpyxl", mode="w") as writer:
        pd.DataFrame(train_preds_all).to_excel(writer, sheet_name="train_pretrained", index=False)
        pd.DataFrame(val_preds_all).to_excel(writer,   sheet_name="val_pretrained",   index=False)
        pd.DataFrame(test_preds_all).to_excel(writer,  sheet_name="test_pretrained",  index=False)
        pd.DataFrame(metrics_all).to_excel(writer,     sheet_name="metrics_pretrained", index=False)
    print(f"• Saved workbook: {pretrain_xlsx}")

    #  aggregate attention across runs into a compact table for plotting (mean across runs).
    try:
        if temporal_by_run:
            T = len(next(iter(temporal_by_run.values())))
            df_temp = pd.DataFrame({"LagIndex": np.arange(T)})
            for k,v in temporal_by_run.items(): df_temp[k]=v
            df_temp["MeanAcrossRuns"] = df_temp[[c for c in df_temp.columns if c.startswith("Run")]].mean(axis=1)
            df_temp.to_excel(os.path.join(variant_dir, f"temporal_attention_all_runs_{model_name}.xlsx"), index=False)
        if feature_by_run:
            feat_cols = [LABEL_MAP.get(c, c) for c in cols]
            df_feat = pd.DataFrame({"Feature": feat_cols})
            for k,v in feature_by_run.items(): df_feat[k]=v
            df_feat["MeanAcrossRuns"] = df_feat[[c for c in df_feat.columns if c.startswith("Run")]].mean(axis=1)
            df_feat.to_excel(os.path.join(variant_dir, f"feature_attention_all_runs_{model_name}.xlsx"), index=False)
    except Exception as e:
        print("Could not aggregate attention tables:", e)

    #  aggregate loss tables across runs into one Excel file (each run = a sheet).
    try:
        loss_files = sorted(glob.glob(os.path.join(variant_dir, f"loss_{model_name}_run_*.xlsx")))
        if loss_files:
            combined_path = os.path.join(variant_dir, f"loss_all_runs_{model_name}.xlsx")
            with pd.ExcelWriter(combined_path, engine="openpyxl", mode="w") as writer:
                for i, fp in enumerate(loss_files, start=1):
                    df = pd.read_excel(fp)
                    df.to_excel(writer, sheet_name=f"Run{i:03d}", index=False)
            print(f"• Aggregated loss tables: {combined_path}")
    except Exception as e:
        print("Loss aggregation error:", e)

    #  run diagnostics and pooling comparisons once at the end using the last trained model object.
    # Regime metadata is inactive; dummy metadata is created inside make_loader.
    # diag_loader = make_loader(Xte_s, yte_s, BATCH_SIZE, shuffle=False, run_seed=999,
    #                           weights=w_te, month=month_te, wet=wet_te)
    diag_loader = make_loader(Xte_s, yte_s, BATCH_SIZE, shuffle=False, run_seed=999)
    diag_res = diagnostics_attention(model, diag_loader, variant_dir, TIMESTEP, cols)

    try:
        # pool_loader = make_loader(Xte_s, yte_s, BATCH_SIZE, shuffle=False, run_seed=999,
        #                           weights=w_te, month=month_te, wet=wet_te)
        pool_loader = make_loader(Xte_s, yte_s, BATCH_SIZE, shuffle=False, run_seed=999)
        pool_res = compare_pooling_modes(model, pool_loader, ymin, ymax, variant_dir, TIMESTEP)
    except Exception as e:
        print("Pooling compare failed:", e)
        pool_res = {}

    #  write a short summary csv for diagnostics.
    if diag_res or pool_res:
        rows = []
        if diag_res:
            rows.append({"Key":"MeanEntropy","Value":diag_res.get("MeanEntropy", np.nan)})
            rows.append({"Key":"MedianEntropy","Value":diag_res.get("MedianEntropy", np.nan)})
            rows.append({"Key":"logT","Value":diag_res.get("logT", np.nan)})
            rows.append({"Key":"MeanAttnSTD","Value":diag_res.get("MeanAttnSTD", np.nan)})
        for mode, m in pool_res.items():
            rows.append({"Key":f"Pool_{mode}_R",   "Value":m.get("R",np.nan)})
            rows.append({"Key":f"Pool_{mode}_RMSE","Value":m.get("RMSE",np.nan)})
            rows.append({"Key":f"Pool_{mode}_MAE", "Value":m.get("MAE",np.nan)})
            rows.append({"Key":f"Pool_{mode}_R2",  "Value":m.get("R2",np.nan)})
        pd.DataFrame(rows).to_csv(os.path.join(variant_dir, "diagnostics_summary.csv"), index=False)

    return variant_dir, pretrain_xlsx

# ============================================================
# 11. Optional checkpoint selection
# ============================================================
def select_topk(variant_dir, model_name, top_k=20):
    #  load metrics from the workbook and rank runs by Test NSE/KGE/RMSE.
    wb = os.path.join(variant_dir, f"pre_trained_{model_name}.xlsx")
    if not os.path.isfile(wb):
        print("No workbook found for selection:", wb)
        return

    dfm = pd.read_excel(wb, sheet_name="metrics_pretrained")
    test_df = dfm[dfm["Set"].astype(str).str.strip().str.lower() == "test"].copy()

    for col, bad in [("NSE", -np.inf), ("KGE", -np.inf), ("RMSE", np.inf)]:
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce").fillna(bad)

    test_df["Run_int"] = pd.to_numeric(test_df["Run"], errors="coerce").fillna(-1).astype(int)
    test_df["Run_id"]  = test_df["Run_int"].astype(str).str.zfill(3)

    leaderboard = test_df.sort_values(
        by=["NSE", "KGE", "RMSE"], ascending=[False, False, True]
    ).reset_index(drop=True)

    leaderboard_path = os.path.join(variant_dir, "leaderboard_pretrained_Test.csv")
    leaderboard.to_csv(leaderboard_path, index=False)
    print("Saved leaderboard ->", leaderboard_path)

    selected_ids = leaderboard.head(top_k)["Run_id"].tolist()
    sel_txt = os.path.join(variant_dir, "selected_runs_test.txt")
    with open(sel_txt, "w") as f:
        for rid in selected_ids:
            f.write(f"{rid}\n")
    print("Saved selected run IDs ->", sel_txt)

    #  copy selected checkpoints into a dedicated folder for later TL/forecasting.
    sel_dir = os.path.join(variant_dir, "selected_pretrained")
    os.makedirs(sel_dir, exist_ok=True)
    copied, missing = 0, []
    for rid in selected_ids:
        src = os.path.join(variant_dir, f"best_{model_name}_Run{rid}.pt")
        dst = os.path.join(sel_dir, f"best_{model_name}_Run{rid}.pt")
        if os.path.isfile(src):
            shutil.copy2(src, dst); copied += 1   #  copy (not move) to preserve the original folder.
        else:
            missing.append(rid)
    print(f"copied {copied} checkpoints -> {sel_dir}")
    if missing:
        print("Missing:", ", ".join(missing))

def select_run_list(variant_dir, model_name, run_list):
    #  manually copy a fixed list of runs (useful when I want consistent ensembles).
    if not run_list:
        print("No custom runs provided, skipping manual selection.")
        return

    selected_ids = [str(int(r)).zfill(3) for r in run_list]

    sel_txt = os.path.join(variant_dir, "selected_runs_custom.txt")
    with open(sel_txt, "w") as f:
        for rid in selected_ids:
            f.write(f"{rid}\n")
    print("Saved custom selected run IDs ->", sel_txt)

    sel_dir = os.path.join(variant_dir, "selected_pretrained_custom")
    os.makedirs(sel_dir, exist_ok=True)

    copied, missing = 0, []
    for rid in selected_ids:
        src = os.path.join(variant_dir, f"best_{model_name}_Run{rid}.pt")
        dst = os.path.join(sel_dir, f"best_{model_name}_Run{rid}.pt")
        if os.path.isfile(src):
            shutil.copy2(src, dst)
            copied += 1
        else:
            missing.append(rid)

    print(f"copied {copied} custom checkpoints -> {sel_dir}")
    if missing:
        print(" Missing (not found):", ", ".join(missing))

# ============================
# 12. Execute experiment
# ============================
#  loop over feature blocks × model architectures and run the full pretraining pipeline.
BLOCKS_TO_RUN = ["All","Climate"]

MODELS_TO_RUN = [
    "LSTM_baseline",
    "LSTM_featureAttn",
    "LSTM_temporalAttn",
    "LSTM_feat+tempAttn",
    "Transformer",
]

for block in BLOCKS_TO_RUN:
    cols = FEATURE_BLOCKS[block]
    for model_name in MODELS_TO_RUN:
        variant_dir, pretrain_xlsx = run_config(block, cols, model_name)

        # I optionally copy top-k runs based on test leaderboard.
        if DO_SELECT_TOPK:
            select_topk(variant_dir, model_name, TOP_K)

        # I also optionally copy a custom run list for fixed ensembles.
        if USE_CUSTOM_RUN_LIST:
            select_run_list(variant_dir, model_name, CUSTOM_RUN_LIST)
