import os
import pandas as pd
import numpy as np

# =========================
# My SETTINGS
# =========================
scenario_name = "output"


# BASE_DIR   = r"/.../22-SWAT_SWC"
# OUTPUT_DIR = os.path.join(BASE_DIR, scenario_name)
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.chdir(BASE_DIR)

# ============================================================
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

sensor_file = os.path.join(BASE_DIR, "GALR_soilmoisture2020_F.csv")
hru_file    = os.path.join(BASE_DIR, f"Model22/{scenario_name}.hru")
swr_file    = os.path.join(BASE_DIR, f"Model22/{scenario_name}.swr")

# (site_label, sensor_id, target_hru)
SITES = [
    ("site22", "GALR0022", 494),
    ("site27", "GALR0027", 276),
    ("site30", "GALR0030", 107),
    ("site31", "GALR0031", 116),
    ("site32", "GALR0032", 158),
    ("site37", "GALR0037", 50),  
    ("site40", "GALR0040", 180),
    ("site43", "GALR0043", 38),
    ("site62", "GALR0062", 338),
    ("site64", "GALR0064", 370),
]

# This is only used to seed a start date. Length comes from HRU rows.
DATE_START = "2000-01-01"
MERGE_HOW  = "left"   # <— keep full SWAT span; sensor is NaN where missing

# =========================  =================== =================== =================== 
# this section is for extracting sensor observations, and SWAT outputs
# =========================  ===================  =================== ===================

def load_and_daily_aggregate_sensor(sensor_csv, sensor_id):
    df = pd.read_csv(sensor_csv)

    # 1) Strip header whitespace 
    df.columns = df.columns.astype(str).str.strip()

    # 2) Handle both "clean" filtered CSV headers and original long headers
    #    Build a rename map only for columns we actually need.
    rename_map = {
        # cleaned names
        "SiteIdentifier": "SiteIdentifier",
        "Date_Time": "Date_Time",
        "soil_mois_2in": "soil_mois_2in",
        "soil_mois_8in": "soil_mois_8in",
        "soil_mois_12in": "soil_mois_12in",
        "soil_tmp_2in": "soil_tmp_2in",
        "soil_tmp_8in": "soil_tmp_8in",
        "soil_tmp_12in": "soil_tmp_12in",
        "rainfall_daily": "rainfall_daily",

        # original long names 
        "Site Identifier": "SiteIdentifier",
        "Date&Time": "Date_Time",
        'Soil moisture, soil, 2" depth, cubic centimeters per cubic centimeter': "soil_mois_2in",
        'Soil moisture, soil, 8" depth, cubic centimeters per cubic centimeter': "soil_mois_8in",
        'Soil moisture, soil, 12" depth, cubic centimeters per cubic centimeter': "soil_mois_12in",
        'Temperature, soil, 2" depth, degrees Celsius': "soil_tmp_2in",
        'Temperature, soil, 8" depth, degrees Celsius': "soil_tmp_8in",
        'Temperature, soil, 12" depth, degrees Celsius': "soil_tmp_12in",
        "Rainfall, no media, daily, millimeters per day": "rainfall_daily",
    }

    # Rename what matches
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    # 3) Verify required columns exist
    required = ["SiteIdentifier", "Date_Time",
                "soil_mois_2in", "soil_mois_8in", "soil_mois_12in",
                "soil_tmp_2in", "soil_tmp_8in", "soil_tmp_12in",
                "rainfall_daily"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required sensor columns: {missing}\n"
                         f"Available columns: {df.columns.tolist()}")

    # 4) Filter site
    df = df[df["SiteIdentifier"] == sensor_id].copy()

    # 5) Parse datetime
    df["Date_Time"] = pd.to_datetime(df["Date_Time"], errors="coerce")
    df = df.dropna(subset=["Date_Time"]).sort_values("Date_Time")

    # 6) Clean numeric + negatives
    for col in ["soil_mois_2in", "soil_mois_8in", "soil_mois_12in",
                "soil_tmp_2in", "soil_tmp_8in", "soil_tmp_12in", "rainfall_daily"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < 0, col] = np.nan

    # 7) Interpolate numeric columns
    df[required[2:]] = df[required[2:]].interpolate(method="nearest", limit_direction="both")

    # 8) Daily aggregation
    df["Date"] = df["Date_Time"].dt.normalize()
    daily = df.groupby("Date", as_index=False).agg({
        "soil_mois_2in": "mean",
        "soil_mois_8in": "mean",
        "soil_mois_12in": "mean",
        "soil_tmp_2in": "mean",
        "soil_tmp_8in": "mean",
        "soil_tmp_12in": "mean",
        "rainfall_daily": "sum",
    })

    daily = daily.rename(columns={"rainfall_daily": "rainfall_mm"})
    daily["mois_avg"] = daily[["soil_mois_2in", "soil_mois_8in", "soil_mois_12in"]].mean(axis=1)
    daily["tmp_avg"]  = daily[["soil_tmp_2in", "soil_tmp_8in", "soil_tmp_12in"]].mean(axis=1)

    return daily[["Date",
                  "soil_mois_2in", "soil_mois_8in", "soil_mois_12in",
                  "soil_tmp_2in", "soil_tmp_8in", "soil_tmp_12in",
                  "rainfall_mm", "tmp_avg", "mois_avg"]]


def read_hru_for_target(hru_path, target_hru_str):
    with open(hru_path, 'r') as f:
        for _ in range(9):
            next(f)
        lines = f.readlines()

    data = [line.strip().split() for line in lines if line.strip()]
    df = pd.DataFrame(data)

    columns = ['LULC', 'HRU', 'GIS', 'SUB', 'MON_MGT',
               'AREAkm2','PRECIPmm', 'PETmm', 'ETmm', 'PERCmm',
               'GW_RCHGmm','REVAPmm', 'SA_STmm','SURQ_GENmm',
               'LATQGENmm', 'WYLDmm', 'DAILYCN', 'TMP_AVdgC', 'TMP_MXdgC', 'TMP_MNdgCS',
               'SOL_TMPdgCS', 'SOLARMJ/m2', 'LAI','VSM1mm/mm','VSM2mm/mm','VSM3mm/mm']
    if df.shape[1] < len(columns):
        raise ValueError(f"output.hru parsed columns ({df.shape[1]}) < expected ({len(columns)}).")
    df = df.iloc[:, :len(columns)]
    df.columns = columns

    df = df[df['HRU'] == str(target_hru_str)].copy()
    numeric_cols = columns[5:]  # from AREAkm2 onward
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df.reset_index(drop=True)

def read_swr_for_target(swr_path, target_hru_int):
    with open(swr_path, 'r') as f:
        for _ in range(3):
            next(f)
        lines = f.readlines()

    data = [line.strip().split() for line in lines if line.strip()]
    df = pd.DataFrame(data)
    df = df.iloc[:, :8]
    df.columns = ['Day', 'HRU', 'GIS', '1', '2', '3', '4', '5']

    numeric_cols = ['Day', 'HRU', '1', '2', '3', '4', '5']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    df = df[df['HRU'] == int(target_hru_int)].copy()

    # Layer mapping (adjust if depths differ)
    df['SW_10_mm'] = df['1']
    df['SW_1_mm']  = df['2']
    df['SW_2_mm']  = df['3']

    df['SW_10_v'] = df['1'] / 10.0   # 10 mm
    df['SW_1_v']  = df['2'] / 240.0  # 240 mm
    df['SW_2_v']  = df['3'] / 130.0  # 130 mm

    return df[['Day', 'HRU', 'SW_10_mm', 'SW_1_mm', 'SW_2_mm', 'SW_10_v', 'SW_1_v', 'SW_2_v']].reset_index(drop=True)

def run_one_site(site_label, sensor_id, target_hru, scenario):
    target_hru_str = str(target_hru)

    # === Output file paths ===
    output_excel    = os.path.join(OUTPUT_DIR, f"SWAT_FINAL_{sensor_id}_{target_hru_str}_{scenario}.xlsx")
    output_combined = os.path.join(OUTPUT_DIR, f"DL_In_{site_label}_Hru_{target_hru_str}_{scenario}.xlsx")

    output_sheet_sensor = f"Sensor_{site_label}"
    sheet_name_hru = site_label
    sheet_name_swr = f"SWR_{site_label}"
    final_sheet_name = f"In_{site_label}_Hru_{target_hru_str}"

    print(f"\n=== Processing {site_label} | sensor={sensor_id} | HRU={target_hru} ===")

    # 1) Sensor → daily
    sensor_daily = load_and_daily_aggregate_sensor(sensor_file, sensor_id)
    with pd.ExcelWriter(output_excel, engine='openpyxl', mode='w') as w:
        sensor_daily.to_excel(w, sheet_name=output_sheet_sensor, index=False)

    # 2) HRU slice
    hru_df = read_hru_for_target(hru_file, target_hru_str)
    with pd.ExcelWriter(output_excel, engine='openpyxl', mode='a', if_sheet_exists='replace') as w:
        hru_df.to_excel(w, sheet_name=sheet_name_hru, index=False)

    # 3) SWR slice
    swr_df = read_swr_for_target(swr_file, target_hru)
    with pd.ExcelWriter(output_excel, engine='openpyxl', mode='a', if_sheet_exists='replace') as w:
        swr_df.to_excel(w, sheet_name=sheet_name_swr, index=False)

    # 4) Build combined DL input (keep full SWAT span; sensor is NaN outside coverage)
    hru_cols = ['LULC', 'HRU', 'GIS', 'SUB','PRECIPmm', 'PETmm', 'ETmm', 'PERCmm',
                'GW_RCHGmm','REVAPmm', 'SA_STmm','SURQ_GENmm',
                'LATQGENmm', 'WYLDmm', 'DAILYCN', 'TMP_AVdgC', 'TMP_MXdgC', 'TMP_MNdgCS',
                'SOL_TMPdgCS', 'SOLARMJ/m2', 'LAI','VSM1mm/mm','VSM2mm/mm','VSM3mm/mm']
    sensor_cols = ['mois_avg']  # 

    hru_saved    = pd.read_excel(output_excel, sheet_name=sheet_name_hru)
    sensor_saved = pd.read_excel(output_excel, sheet_name=output_sheet_sensor)

    # Build HRU Date using its true length (keeps full SWAT window)
    hru_len = len(hru_saved)
    date_for_hru = pd.date_range(start=DATE_START, periods=hru_len, freq='D')

    hru_trimmed = hru_saved[hru_cols].copy()
    hru_trimmed.insert(0, 'Date', date_for_hru)

    # Ensure sensor Date is datetime and keep chosen columns
    if 'Date' in sensor_saved.columns:
        sensor_saved['Date'] = pd.to_datetime(sensor_saved['Date'])
    sensor_keep = sensor_saved[['Date'] + sensor_cols].copy()

    # Left-merge: keep all HRU (SWAT) dates; sensor is NaN where missing
    combined_df = pd.merge(hru_trimmed, sensor_keep, on='Date', how=MERGE_HOW)

    # --------  include SWR and align it over HRU dates ----------
    # swr_saved = pd.read_excel(output_excel, sheet_name=sheet_name_swr)
    # swr_len = len(swr_saved)
    # swr_dates = date_for_hru[:swr_len]  # assumes same ordering as HRU daily rows
    # swr_saved = swr_saved.copy()
    # swr_saved['Date'] = swr_dates
    # combined_df = pd.merge(
    #     combined_df,
    #     swr_saved[['Date', 'SW_10_v', 'SW_1_v', 'SW_2_v']],
    #     on='Date', how=MERGE_HOW
    # )
    # --------------------------------------------------------------------

    combined_df.to_excel(output_combined, sheet_name=final_sheet_name, index=False)
    print(f"✓ {site_label}: combined rows = {len(combined_df)} (merge='{MERGE_HOW}') → {output_combined}")

# =========================
#  LOOP
# =========================
failures = []
for site_label, sensor_id, target_hru in SITES:
    try:
        run_one_site(site_label, sensor_id, target_hru, scenario_name)
    except Exception as e:
        failures.append((site_label, str(e)))
        print(f"✗ Failed: {site_label} → {e}")

if failures:
    print("\nSites with errors:")
    for s, err in failures:
        print(f" - {s}: {err}")
else:
    print(" completed successfully.")



##############################################################################################################
##############################################################################################################

##COMBINING THE SHEETS IN ONE SINGLE EXCEL FILES READY TO BE USED FOR DL MODELS

##############################################################################################################
##############################################################################################################
##############################################################################################################



import os
import pandas as pd

# ============ My SETTINGS ============


#BASE_DIR = r"/.../22-SWAT_SWC"

import os
from pathlib import Path
import yaml

# ============================================================
# Load path configuration
# ============================================================

##This script combines the time series data from each site in one single excle file
scenario_name="output"
model = "model22"

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

BASE_DIR = Path(config["base_dir"])

os.chdir(BASE_DIR)
print("New Directory:", os.getcwd())
input_folder = os.path.join(BASE_DIR, scenario_name)
output_file = os.path.join(input_folder, f"1-Inputs_{model}.xlsx")


# === Files to combine ===
selected_files = [
    f"DL_In_site22_Hru_494_{scenario_name}.xlsx",
    f"DL_In_site27_Hru_276_{scenario_name}.xlsx",
    f"DL_In_site30_Hru_107_{scenario_name}.xlsx",
    f"DL_In_site31_Hru_116_{scenario_name}.xlsx",
    f"DL_In_site32_Hru_158_{scenario_name}.xlsx",
    f"DL_In_site37_Hru_50_{scenario_name}.xlsx",
    f"DL_In_site40_Hru_180_{scenario_name}.xlsx",
    f"DL_In_site43_Hru_38_{scenario_name}.xlsx",
    f"DL_In_site62_Hru_338_{scenario_name}.xlsx",
    f"DL_In_site64_Hru_370_{scenario_name}.xlsx",
]

# ============ COMBINE FILES ============
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    for file_name in selected_files:
        file_path = os.path.join(input_folder, file_name)
        if not os.path.exists(file_path):
            print(f"{file_name}")
            continue

        try:
            # Load the Excel file (first sheet only)
            xls = pd.ExcelFile(file_path, engine="openpyxl")
            
            # Read the first (or only) sheet in the file
            first_sheet = xls.sheet_names[0]
            df = xls.parse(first_sheet)
            
            # Use only the file name (without .xlsx) for the sheet name
            sheet_name = os.path.splitext(file_name)[0]
            sheet_name = sheet_name[:31]  # Excel limit for sheet names
            
            # Write to the combined file
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"Added: {sheet_name}")

        except Exception as e:
            print(f"Failed {file_name}: {e}")

print(f"\n Combined file saved as: {output_file}")


