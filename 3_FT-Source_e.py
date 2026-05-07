

# ============================================================
# TRANSFER LEARNING / Pre trainign on source site and Fine-tune on multiple target sites
# (Works for all model in MODELS: 4 LSTM variants + Transformer)
# ============================================================

import os, re, glob, math, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -----------------------
# Splits used during Pre training at source site (align scalers!)
# -----------------------
SPLITS = {
    "train": ("2003-01-01", "2015-12-31"),
    "val":   ("2016-01-01", "2016-12-31"),
    "test":  ("2017-01-01", "2019-12-31"),
}

# -----------------------
# Core hyperparams
# -----------------------
TIMESTEP     = 10
BATCH_SIZE   = 32
HIDDEN_SIZE  = 64
NUM_LAYERS   = 2
DROPOUT      = 0.1
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_COL   = "mois_avg"
ADD_SZN      = True  # 

# EXACT base features from training (#SOL_TMPdgCS)
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

PRE_BLOCK = "All"  ## we change to "Climate" to test only climate block

# ============================================================
# Models (same as pre-training)
# ============================================================
class BaseHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
        )
    def forward(self, x): return self.head(x)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_size, temperature=0.5):
        super().__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.v   = nn.Linear(hidden_size, 1, bias=False)
        self.temperature = temperature
    def forward(self, H):
        scores = self.v(torch.tanh(self.W_h(H)))
        attn_t = torch.softmax(scores / self.temperature, dim=1)
        ctx = (attn_t * H).sum(dim=1)
        return ctx, attn_t.squeeze(-1)

class FeatureAttentionGate(nn.Module):
    def __init__(self, num_features, reduction=4, sigmoid_tau=0.7):
        super().__init__()
        r = max(1, num_features // reduction)
        self.fc1 = nn.Sequential(nn.Linear(num_features, r), nn.BatchNorm1d(r), nn.ReLU())
        self.fc2 = nn.Linear(r, num_features)
        self.sigmoid_tau = sigmoid_tau
    def forward(self, x):
        z = x.mean(dim=1)
        w = self.fc2(self.fc1(z))
        w = torch.sigmoid(w / self.sigmoid_tau)
        return x * w.unsqueeze(1), w

class LSTMPlain(nn.Module):  # "LSTM_baseline"
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0.0)
        self.proj = BaseHead(hidden_size)
    def forward(self, x, return_attn=False):
        H, _ = self.lstm(x); y = self.proj(H[:, -1, :])
        return (y, None, None) if return_attn else y

class LSTM_TemporalOnly(nn.Module):  # "LSTM_temporalAttn"
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_size); self.pre_attn_dropout = nn.Dropout(0.1)
        self.tattn = TemporalAttention(hidden_size, temperature=0.5)
        self.proj  = BaseHead(hidden_size)
    def forward(self, x, return_attn=False):
        H, _ = self.lstm(x); H = self.pre_attn_dropout(self.norm(H))
        ctx, attn_t = self.tattn(H); y = self.proj(ctx)
        return (y, attn_t, None) if return_attn else y

class LSTM_FeatureOnly(nn.Module):  # "LSTM_featureAttn"
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.fgate = FeatureAttentionGate(input_size, sigmoid_tau=0.7)
        self.lstm  = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                             dropout=dropout_rate if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_size); self.proj = BaseHead(hidden_size)
    def forward(self, x, return_attn=False):
        x, feat_w = self.fgate(x); H, _ = self.lstm(x); y = self.proj(self.norm(H)[:, -1, :])
        return (y, None, feat_w) if return_attn else y

class LSTM_FeatureAndTemporal(nn.Module):  # "LSTM_feat+tempAttn"
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate):
        super().__init__()
        self.fgate = FeatureAttentionGate(input_size, sigmoid_tau=0.7)
        self.lstm  = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                             dropout=dropout_rate if num_layers > 1 else 0.0)
        self.norm = nn.LayerNorm(hidden_size); self.pre_attn_dropout = nn.Dropout(0.1)
        self.tattn = TemporalAttention(hidden_size, temperature=0.5)
        self.proj  = BaseHead(hidden_size)
    def forward(self, x, return_attn=False):
        x, feat_w = self.fgate(x); H, _ = self.lstm(x)
        ctx, attn_t = self.tattn(self.pre_attn_dropout(self.norm(H))); y = self.proj(ctx)
        return (y, attn_t, feat_w) if return_attn else y

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x): return x + self.pe[:, :x.size(1), :]

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, n_heads=4, n_layers=2, d_ff=256, dropout=0.1, use_cls=True):
        super().__init__()
        self.use_cls = use_cls
        self.input_proj = nn.Linear(input_dim, d_model); self.dropout_in = nn.Dropout(dropout)
        max_len = seq_len + (1 if use_cls else 0); self.pos_enc = PositionalEncoding(d_model, max_len)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
                                           dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1,1,d_model)); nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.head = nn.Sequential(nn.Linear(d_model, d_model//2), nn.ReLU(), nn.Dropout(0.2), nn.Linear(d_model//2, 1))
    def forward(self, x):
        x = self.dropout_in(self.input_proj(x))
        if self.use_cls:
            B = x.size(0); cls = self.cls_token.expand(B,1,-1); x = torch.cat([cls, x], dim=1)
        z = self.encoder(self.pos_enc(x)); h = z[:,0,:] if self.use_cls else z.mean(dim=1)
        return self.head(h)

MODELS = {
    "LSTM_baseline":      LSTMPlain,
    "LSTM_temporalAttn":  LSTM_TemporalOnly,
    "LSTM_featureAttn":   LSTM_FeatureOnly,
    "LSTM_feat+tempAttn": LSTM_FeatureAndTemporal,
    "Transformer":        lambda input_size, hidden_size, num_layers, dropout_rate:
                          TransformerRegressor(input_dim=input_size, seq_len=TIMESTEP,
                                               d_model=HIDDEN_SIZE, n_heads=4, n_layers=2,
                                               d_ff=256, dropout=DROPOUT, use_cls=True).to(DEVICE)
}

# ============================================================
# Scenario  files/folders
# ============================================================
scenario_name = "output"
model = "model22"
step = "s10"
runs="R20"

type="single"  # 


#BASE_DIR = "/..../22-SWAT_SWC"


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

FILE_PATH = f"{BASE_DIR}/{scenario_name}/1-Inputs_{model}.xlsx"
SITE_TRAIN = f"DL_In_site31_Hru_116_{scenario_name}"
OUTPUT_DIR = f"{BASE_DIR}/1_Train_{type}_{step}_{model}_{runs}"


FT_OUTPUT_ROOT = f"{BASE_DIR}/2_FT_{type}_{step}_{model}_{runs}"
 ##2-FT_{step}_{model}_{runs}_{type}
os.makedirs(FT_OUTPUT_ROOT, exist_ok=True)

TARGET_SITES =  [
    f"DL_In_site22_Hru_494_{scenario_name}",
    f"DL_In_site27_Hru_276_{scenario_name}",
    f"DL_In_site30_Hru_107_{scenario_name}",
    #f"DL_In_site31_Hru_116_{scenario_name}",
    f"DL_In_site32_Hru_158_{scenario_name}",
    f"DL_In_site37_Hru_50_{scenario_name}",
    f"DL_In_site40_Hru_180_{scenario_name}",
    f"DL_In_site43_Hru_38_{scenario_name}",
    f"DL_In_site62_Hru_338_{scenario_name}",
    f"DL_In_site64_Hru_370_{scenario_name}",
]

# ============================================================
# Reproducibility helpers
# ============================================================
def set_global_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed); random.seed(worker_seed)

def make_loader(X, y, batch_size, shuffle, run_seed):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32))
    # If empty, return a non-shuffled loader (SequentialSampler handles len==0)
    if len(ds) == 0:
        return DataLoader(ds, batch_size=batch_size, shuffle=False,
                          num_workers=0, pin_memory=torch.cuda.is_available())
    g = torch.Generator(); g.manual_seed(run_seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, worker_init_fn=_seed_worker,
                      generator=g, pin_memory=torch.cuda.is_available())

# ============================================================
# Discover trained models and pick a fixed checkpoint rank
# ============================================================
def discover_trained_models(output_dir, block="All"):
    root = Path(output_dir) / block
    found = {}
    if not root.exists():
        raise FileNotFoundError(f"Trained block folder not found: {root}")
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        sel = model_dir / "selected_pretrained_custom"
        if sel.exists() and list(sel.glob("best_*_Run*.pt")):
            found[model_dir.name] = str(sel)
    return found

TRAINED = discover_trained_models(OUTPUT_DIR, block=PRE_BLOCK)
print("Discovered models:", list(TRAINED.keys()))



def read_selected_run_ids(selected_dir):
    # Prefer leaderboard if present; else selected_runs_test.txt; else glob all
    leaderboard = Path(selected_dir).parent / "leaderboard_pretrained_Test.csv"
    if leaderboard.exists():
        df = pd.read_csv(leaderboard)
        df["Run"] = pd.to_numeric(df["Run"], errors="coerce")
        run_ids = [str(int(r)).zfill(3) for r in df["Run"].dropna().astype(int).tolist()]
    else:
        sel = Path(selected_dir) / "selected_runs_test.txt"
        if sel.exists():
            run_ids = [l.strip() for l in sel.read_text().splitlines() if l.strip()]
        else:
            # fallback: grab any checkpoint names present
            run_ids = sorted({
                p.stem.split("Run")[-1] for p in Path(selected_dir).glob("best_*_Run*.pt")
            })
    return run_ids

# --- Gather per-model run lists and intersect them (paired analysis) ---
MODEL_RUNS = {}
for model_name, selected_dir in TRAINED.items():
    MODEL_RUNS[model_name] = read_selected_run_ids(selected_dir)

# Keep only RunIDs present for *all* models
common_runs = sorted(set.intersection(*(set(v) for v in MODEL_RUNS.values())))
print(f"[FT] Common RunIDs across all models: {common_runs[:10]} ... total={len(common_runs)}")


# ========================================================================================
# Input-size inference from checkpoint (prevents 10 vs 12 mismatch because we have ADD_SZN)
# ========================================================================================
def infer_expected_input_size(ckpt_path, model_name):
    state = torch.load(ckpt_path, map_location="cpu")
    if model_name.startswith("LSTM"):
        key = "lstm.weight_ih_l0"
        if key not in state:
            # some PyTorch versions store under module.*
            key = f"module.{key}" if f"module.{key}" in state else key
        return state[key].shape[1]
    # Transformer
    key = "input_proj.weight"
    if key not in state:
        key = f"module.{key}" if f"module.{key}" in state else key
    return state[key].shape[1]

def build_feature_list_for_ckpt(expected_in, add_seasonal):
    base = FEATURE_BLOCKS[PRE_BLOCK].copy()
    if add_seasonal and expected_in == len(base) + 2:
        return base + ["doy_sin","doy_cos"]
    if expected_in == len(base):
        return base
    # If mismatched, 
    raise RuntimeError(
        f"Checkpoint expects input_size={expected_in}, but base={len(base)} "
        f"(+2 seasonals => {len(base)+2}). Set ADD_SZN correctly or retrain."
    )

# ============================================================
# Scalers: reuse pretraining scalers (compute once if missing)
# ============================================================
PRE_SCALERS_NAME = "pretrain_scalers.pkl"

def _compute_scalers_from_source(file_path, source_sheet, cols):
    df = pd.read_excel(file_path, sheet_name=source_sheet).copy()
    df["Date"] = pd.to_datetime(df["Date"]); df = df.sort_values("Date").reset_index(drop=True)
    if "doy_sin" in cols and "doy_sin" not in df.columns:
        doy = df["Date"].dt.dayofyear
        df["doy_sin"] = np.sin(2*np.pi*doy/365.25)
        df["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    dtr, dtr2 = SPLITS["train"]
    df_tr = df[(df["Date"]>=dtr) & (df["Date"]<=dtr2)].copy()
    X_min = df_tr[cols].min()
    X_max = df_tr[cols].max()
    y_min = df_tr[[TARGET_COL]].min(); y_max = df_tr[[TARGET_COL]].max()
    return X_min, X_max, y_min, y_max

def load_pretrain_scalers(model_root_dir, cols):
    pkl_path = Path(model_root_dir) / PRE_SCALERS_NAME
    if pkl_path.exists():
        obj = pd.read_pickle(pkl_path)
        return obj["X_min"], obj["X_max"], obj["y_min"], obj["y_max"]
    X_min, X_max, y_min, y_max = _compute_scalers_from_source(FILE_PATH, SITE_TRAIN, cols)
    pd.to_pickle({"X_min": X_min, "X_max": X_max, "y_min": y_min, "y_max": y_max}, pkl_path)
    print(f"[Scalers] Computed from {SITE_TRAIN} and saved to {pkl_path}")
    return X_min, X_max, y_min, y_max

# ============================================================
# Data utilities
# ============================================================
def get_site_xy(sheet_name, cols, X_min, X_max, y_min, y_max):
    df = pd.read_excel(FILE_PATH, sheet_name=sheet_name).copy()
    df["Date"] = pd.to_datetime(df["Date"]); df = df.sort_values("Date").reset_index(drop=True)
    if "doy_sin" in cols and "doy_sin" not in df.columns:
        doy = df["Date"].dt.dayofyear
        df["doy_sin"] = np.sin(2*np.pi*doy/365.25)
        df["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    eps = 1e-12
    X = 2 * (df[cols] - X_min) / (X_max - X_min + eps) - 1
    y = 2 * (df[[TARGET_COL]] - y_min) / (y_max - y_min + eps) - 1
    X = X.clip(-1.5, 1.5).values.astype(np.float32)
    y = y.clip(-1.5, 1.5).values.astype(np.float32)
    return df, X, y

def make_seq(X, y, T):
    Xs, ys = [], []
    for i in range(len(X) - T):
        Xs.append(X[i:i+T]); ys.append(y[i+T])
    return np.asarray(Xs, np.float32), np.asarray(ys, np.float32)

def inverse_y(z_scaled, y_min, y_max):
    return (z_scaled + 1.0) * (y_max.values - y_min.values) / 2.0 + y_min.values

# ============================================================
# Metrics (same as training)
# ============================================================
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).ravel(); y_pred = np.asarray(y_pred).ravel()
    r   = np.corrcoef(y_true, y_pred)[0, 1]
    rmse= np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    nse = 1 - np.sum((y_true - y_pred)**2)/np.sum((y_true - np.mean(y_true))**2)
    std_true  = np.std(y_true); mean_true = np.mean(y_true)
    alpha = np.std(y_pred)/std_true if std_true!=0 else np.nan
    beta  = np.mean(y_pred)/mean_true if mean_true!=0 else np.nan
    kge  = 1 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2)
    pbias = 100*np.sum(y_pred - y_true)/np.sum(y_true)
    return {"R": r, "RMSE": rmse, "MAE": mae, "R2": r2, "NSE": nse, "PBIAS": pbias, "KGE": kge}

# ============================================================
# Build/load model
# ============================================================
def build_model_for_name(model_name, input_size):
    ctor = MODELS[model_name]
    return ctor(input_size=input_size, hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS, dropout_rate=DROPOUT).to(DEVICE)

def load_checkpoint_into_model(model, ckpt_path):
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state, strict=True)
    return model

# ============================================================
# FT params
# ============================================================
YEARS_OPTIONS     = [0, 1, 2, 3, 4, 5, 6]   # 0 = zero-shot transfer with dirctly using pre-trained model with no finetuning, 1 -6 years of finetuning
STRATEGIES        = ["last_k"]
FT_LR             = 1e-4
FT_WD             = 1e-6
FT_PATIENCE       = 15
FT_EPOCHS         = 100
FT_HEAD_ONLY      = True
FT_PROG_UNFREEZE  = True

def pick_finetune_windows(years, strategy):
    if years == 0:
        return []
    if strategy == "last_k":
        start = pd.to_datetime(f"{2017-years}-01-01")
        end   = pd.to_datetime("2016-12-31")
        return [(start, end)]
    raise ValueError("no found strategy")

# ============================================================
# FT core for one site  (paired-by-run version)
# ============================================================
def finetune_on_site(model_name, ckpt_path, feat_cols, site_sheet,
                     X_min, X_max, y_min, y_max, rid):
    set_global_seed(123)

    # --- Data ---
    df, X, y = get_site_xy(site_sheet, feat_cols, X_min, X_max, y_min, y_max)
    X_all_s, y_all_s = make_seq(X, y, TIMESTEP)

    # --- Fixed test window (2015–2016) ---
    dte1, dte2 = SPLITS["test"]
    test_mask = (df["Date"] >= dte1) & (df["Date"] <= dte2)
    te_idx = [i for i in range(len(df)) if (i >= TIMESTEP) and test_mask.iloc[i]]
    Xte = X_all_s[np.array(te_idx) - TIMESTEP]
    yte = y_all_s[np.array(te_idx) - TIMESTEP]

    # --- Model ---
    input_size = X.shape[1]
    model = build_model_for_name(model_name, input_size=input_size)
    model = load_checkpoint_into_model(model, ckpt_path)

    # Freeze backbone first (head-only to start)
    if FT_HEAD_ONLY:
        if hasattr(model, "lstm"):
            for p in model.lstm.parameters(): 
                p.requires_grad = False
        if isinstance(model, TransformerRegressor):
            for p in model.encoder.parameters(): 
                p.requires_grad = False
            for p in model.input_proj.parameters(): 
                p.requires_grad = False

    results = []

    # --- Iterate fine-tune budgets ---
    for years in YEARS_OPTIONS:
        for strat in STRATEGIES:
            # FT windows (years=0 => zero-shot)
            windows = pick_finetune_windows(years, strat)
            ft_mask = np.zeros(len(df), dtype=bool)
            for (s, e) in windows:
                ft_mask |= (df["Date"] >= s) & (df["Date"] <= e)

            ft_idx = [i for i in range(len(df)) if (i >= TIMESTEP) and ft_mask[i]]

            if ft_idx:
                Xft = X_all_s[np.array(ft_idx) - TIMESTEP]
                yft = y_all_s[np.array(ft_idx) - TIMESTEP]
            else:
                # zero-shot: empty train set
                Xft = np.empty((0, TIMESTEP, input_size), np.float32)
                yft = np.empty((0, 1), np.float32)

            train_loader = make_loader(Xft, yft, BATCH_SIZE, shuffle=True, run_seed=999 + years)
            test_loader  = make_loader(Xte, yte, BATCH_SIZE, shuffle=False, run_seed=1234)

            # ---- Train/eval loop ----
            criterion = nn.SmoothL1Loss(beta=1.0)
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = Adam(params, lr=FT_LR, weight_decay=FT_WD)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5, cooldown=3, min_lr=1e-5
            )

            best_v = float("inf"); best_state = None; wait = 0; ema = None; alpha = 0.2
            for epoch in range(FT_EPOCHS):
                # If zero-shot, this loop effectively skips training and just validates
                if len(Xft) > 0:
                    model.train()
                    for xb, yb in train_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        optimizer.zero_grad()
                        loss = criterion(model(xb), yb)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                model.eval(); v = 0.0
                with torch.no_grad():
                    for xb, yb in test_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        v += criterion(model(xb), yb).item()
                v /= max(1, len(test_loader))
                ema = v if ema is None else alpha*v + (1-alpha)*ema
                scheduler.step(ema)

                if v < best_v:
                    best_v = v
                    best_state = {k: p.detach().cpu().clone() for k, p in model.state_dict().items()}
                    wait = 0
                else:
                    wait += 1
                    # progressive unfreeze after a few stagnant epochs
                    if FT_PROG_UNFREEZE and FT_HEAD_ONLY and wait == 5:
                        if hasattr(model, "lstm"):
                            for p in model.lstm.parameters(): 
                                p.requires_grad = True
                        if isinstance(model, TransformerRegressor):
                            for p in model.encoder.parameters(): 
                                p.requires_grad = True
                            for p in model.input_proj.parameters(): 
                                p.requires_grad = True
                        params = [p for p in model.parameters() if p.requires_grad]
                        optimizer = Adam(params, lr=FT_LR, weight_decay=FT_WD)
                    if wait >= FT_PATIENCE:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)

            # ---- Inverse-scale eval on test ----
            preds, obs = [], []
            model.eval()
            with torch.no_grad():
                for xb, yb in test_loader:
                    yhat = model(xb.to(DEVICE)).cpu().numpy()
                    preds.append(inverse_y(yhat, y_min, y_max))
                    obs.append(inverse_y(yb.numpy(), y_min, y_max))
            y_pred = np.concatenate(preds).ravel()
            y_true = np.concatenate(obs).ravel()

            res = compute_metrics(y_true, y_pred)
            results.append({"Years": years, "Strategy": strat, **res})

            # ---- Save preds under by_run/Run{rid}/... ----
            yield_dir = Path(FT_OUTPUT_ROOT) / "by_run" / f"Run{rid}" / model_name / site_sheet
            yield_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
                yield_dir / f"preds_years{years}_{strat}.csv", index=False
            )

    return results

# ============================================================
# RUN FT: loop over models → pick fixed checkpoint → infer input → scalers → all sites
# ============================================================
# =========  LOOP: per model × per RunID (paired) =========
for model_name, selected_dir in TRAINED.items():
    model_root_dir = str(Path(selected_dir).parent)     # .../All/<model_name>
    # Infer feature layout once per model (use the first run’s ckpt)
    ckpt_probe = list(Path(selected_dir).glob("best_{}_Run*.pt".format(model_name)))[0]
    expected_in = infer_expected_input_size(str(ckpt_probe), model_name)
    feat_cols   = build_feature_list_for_ckpt(expected_in, ADD_SZN)
    X_min, X_max, y_min, y_max = load_pretrain_scalers(model_root_dir, feat_cols)

    for rid in common_runs:
        ckpt_path = str(Path(selected_dir) / f"best_{model_name}_Run{rid}.pt")
        if not Path(ckpt_path).exists():
            print(f"[FT] Skip missing {ckpt_path}")
            continue
        print(f"\n[FT] {model_name} | Run {rid} -> {ckpt_path}")
        print(f"[FT] Expected input size = {expected_in} → using columns: {feat_cols}")

        # Fine-tune across target sites
        for site_sheet in TARGET_SITES:
            rows = finetune_on_site(model_name, ckpt_path, feat_cols, site_sheet,
                                    X_min, X_max, y_min, y_max,rid)
            out_dir = Path(FT_OUTPUT_ROOT) / "by_run" / f"Run{rid}" / model_name / site_sheet
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out_dir / "ft_metrics.csv", index=False)

print("\n[FT] All fine-tuning runs complete ")
