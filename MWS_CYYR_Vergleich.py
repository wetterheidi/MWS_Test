import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import re
import sys
import os

# --- 1. KONFIGURATION ---

# A) ZENTRALER PFAD
ARBEITSVERZEICHNIS = '/Users/wetterheidi/Documents/Dienst/GooseBay/MWS'

# B) DATEINAMEN
DATEI_NAME_A = '260223_MWS.csv'
DATEI_NAME_B = '260223_CYYR.csv'

# C) METADATEN
CURRENT_YEAR = 2026
CURRENT_MONTH = 2
PLOT_TYPES = ['BR', 'FG', 'RA', 'SN', 'SH', 'TS']

# --- PFADE ZUSAMMENBAUEN ---
FILE_A = os.path.join(ARBEITSVERZEICHNIS, DATEI_NAME_A)
FILE_B = os.path.join(ARBEITSVERZEICHNIS, DATEI_NAME_B)

# --- HELFER: DATEI LADEN ---
def load_raw_lines(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"FEHLER beim Lesen von {filepath}: {e}")
        return []

# --- 2. PARSING LOGIK ---
def parse_metar_full(raw_line):
    if not isinstance(raw_line, str) or not raw_line.strip(): return None
    parts = raw_line.strip().split()
    
    # Zeitstempel
    time_token = next((p for p in parts if p.endswith('Z') and len(p) >= 7 and p[:6].isdigit()), None)
    if not time_token: return None 
    
    day, hour, minute = int(time_token[:2]), int(time_token[2:4]), int(time_token[4:6])
    try: ts = datetime(CURRENT_YEAR, CURRENT_MONTH, day, hour, minute)
    except: return None

    # Init
    wind_str, vis, temp, dew, qnh = "N/A", "N/A", "N/A", "N/A", "N/A"
    wdir, wspd, wgust = None, None, None
    wx_list = []
    skip_next, stop_wx = False, False

    for i, p in enumerate(parts):
        if skip_next: skip_next = False; continue
        
        # --- WIND --- (Nur den ersten Treffer verarbeiten!)
        if p.endswith('KT') and wind_str == "N/A":
            wind_str = p; val = p.replace('KT', '')
            if 'G' in val:
                try: val, g = val.split('G'); wgust = int(g)
                except: pass
            if val.startswith('VRB'): wdir = None; wspd = int(val[3:]) if val[3:].isdigit() else 0
            elif len(val) >= 5:
                try: wdir, wspd = int(val[:3]), int(val[3:])
                except: pass
            continue
            
        # --- SICHT --- (Nur den ersten Treffer verarbeiten!)
        if 'SM' in p and vis == "N/A":
            vis = f"{parts[i-1]} {p}" if i > 0 and parts[i-1].isdigit() else p
            continue
            
        # --- TEMPERATUR --- (Nur den ersten Treffer verarbeiten!)
        if '/' in p and not p.startswith('/') and not p.endswith('/') and 'SM' not in p and temp == "N/A":
            clean = p.replace('M','').replace('/','').replace('-','')
            if clean.isdigit():
                t_parts = p.split('/'); 
                if len(t_parts) == 2: temp, dew = t_parts[0], t_parts[1]
            continue
            
        # --- QNH --- (Nur den ersten Treffer verarbeiten!)
        if p.startswith('A') and len(p) == 5 and p[1:].isdigit() and qnh == "N/A": 
            qnh = p; continue
            
        if not stop_wx:
            if p.startswith('SF') or (p.startswith('SC') and not p.startswith('SCT')) or (p.startswith('SN') and len(p)>2 and p[2].isdigit()): stop_wx = True
        if stop_wx: continue
        
        ignore = {'METAR', 'SPECI', 'AUTO', 'RMK', 'COR', 'AO2', 'SLPNO', time_token, wind_str}
        if p.endswith('°') or p in ignore or p.isdigit(): continue
        wx_list.append(p)

    return {"Timestamp": ts, "Time_Str": f"{day:02d}. {hour:02d}:{minute:02d}Z",
            "Wind_Dir": wdir, "Wind_Spd": wspd, "Wind_Gust": wgust, "Vis": vis,
            "Temp": temp, "Dew": dew, "QNH": qnh, "Wx": " ".join(wx_list)}

def get_all_clouds(w):
    """Sucht nach ALLEN Wolkenschichten in einem METAR."""
    if pd.isna(w): return []
    w_str = str(w)
    layers = []
    matches = re.finditer(r'(FEW|SCT|BKN|OVC|VV|///)(\d{3})', w_str)
    for m in matches:
        layers.append(int(m.group(2)) * 100)
    if not layers and any(x in w_str for x in ['CLR','FEW','SCT','SKC']):
        layers.append(6000)
    return layers

# --- 3. HAUPTPROGRAMM ---
print("Starte Verarbeitung...")
lines_a = load_raw_lines(FILE_A)
lines_b = load_raw_lines(FILE_B)

data_a = [x for x in [parse_metar_full(r) for r in lines_a] if x]
data_b = [x for x in [parse_metar_full(r) for r in lines_b] if x]

if not data_a or not data_b:
    print("Keine gültigen METARs gefunden."); sys.exit()

print(f"- Station A: {len(data_a)} Datensätze | Station B: {len(data_b)} Datensätze")

df_a = pd.DataFrame(data_a).sort_values('Timestamp')
df_b = pd.DataFrame(data_b).sort_values('Timestamp')

results = []
for idx, row_a in df_a.iterrows():
    diffs = (df_b['Timestamp'] - row_a['Timestamp']).abs()
    if diffs.empty: continue
    min_diff = diffs.min()
    if min_diff <= timedelta(minutes=15):
        row_b = df_b.loc[diffs.idxmin()]
        res = {"Timestamp": row_a['Timestamp'], "Time (Z)": row_a['Time_Str'], "Diff_Min": int(min_diff.total_seconds()//60)}
        for k in ['Wind_Dir', 'Wind_Spd', 'Wind_Gust', 'Vis', 'Wx', 'Temp', 'Dew', 'QNH']:
            res[f"MWS_{k}"] = row_a[k]
            res[f"CYYR_{k}"] = row_b[k]
        results.append(res)

df = pd.DataFrame(results)
if df.empty: print("Keine Matches gefunden."); sys.exit()

# --- NAMENSGEBUNG ---
first_ts = df['Timestamp'].iloc[0]
date_prefix = first_ts.strftime('%y%m%d')

csv_filename = f'{date_prefix}_MWS_CYYR_Vergleich.csv'
plot1_filename = f'{date_prefix}_Vergleich_Parameter.png'
plot2_filename = f'{date_prefix}_Vergleich_Wetter.png'

out_csv = os.path.join(ARBEITSVERZEICHNIS, csv_filename)
out_plot1 = os.path.join(ARBEITSVERZEICHNIS, plot1_filename)
out_plot2 = os.path.join(ARBEITSVERZEICHNIS, plot2_filename)

df.drop(columns=['Timestamp']).to_csv(out_csv, index=False)
print(f"OK: Tabelle gespeichert: {csv_filename}")

# --- GRAFIKEN ---
def cvt_t(v):
    if pd.isna(v) or v == 'N/A': return None
    s = str(v).replace('M', '-')
    try: return float(s)
    except ValueError: return None
def cvt_q(v):
    if pd.isna(v) or v == 'N/A': return None
    s = str(v).replace('A', '')
    try: return float(s)/100.0
    except ValueError: return None
def cvt_v(v):
    if pd.isna(v) or v == 'N/A': return None
    s = str(v).replace('SM', '').strip(); is_l = s.startswith('M'); s = s[1:] if is_l else s
    try: val = float(eval(s.replace(' ', '+'))); return max(0.1, val-0.1) if is_l else val
    except: return None
def get_ceil(w):
    if pd.isna(w): return np.nan
    w_str = str(w)
    m1 = re.search(r'(VV|OVC|BKN)(\d{3})', w_str)
    if m1: return int(m1.group(2)) * 100
    m2 = re.search(r'///(\d{3})', w_str)
    if m2: return int(m2.group(1)) * 100
    return np.nan
def check_wx(w, c): return c in str(w)

for col in ['Temp', 'Dew']:
    df[f'MWS_{col}_Val'] = df[f'MWS_{col}'].apply(cvt_t)
    df[f'CYYR_{col}_Val'] = df[f'CYYR_{col}'].apply(cvt_t)
df['MWS_QNH_Val'] = df['MWS_QNH'].apply(cvt_q); df['CYYR_QNH_Val'] = df['CYYR_QNH'].apply(cvt_q)
df['MWS_Vis_Val'] = df['MWS_Vis'].apply(cvt_v); df['CYYR_Vis_Val'] = df['CYYR_Vis'].apply(cvt_v)
df['MWS_Ceil'] = df['MWS_Wx'].apply(get_ceil); df['CYYR_Ceil'] = df['CYYR_Wx'].apply(get_ceil)

print("- Erstelle Grafik 1 (Parameter & Wind)...")
fig1, ax = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
ax[0].plot(df['Timestamp'], df['MWS_Temp_Val'], 'g-o', markersize=4, label='MWS Temp')
ax[0].plot(df['Timestamp'], df['MWS_Dew_Val'], 'g--', alpha=0.5, label='MWS Td')
ax[0].plot(df['Timestamp'], df['CYYR_Temp_Val'], 'r-x', markersize=4, label='CYYR Temp')
ax[0].plot(df['Timestamp'], df['CYYR_Dew_Val'], 'r--', alpha=0.5, label='CYYR Td')
ax[0].legend(ncol=2, fontsize='small'); ax[0].set_ylabel('°C'); ax[0].grid(True, alpha=0.3); ax[0].set_title('Temperatur & Taupunkt')

ax[1].plot(df['Timestamp'], df['MWS_Vis_Val'], 'g-o', markersize=4, label='MWS')
ax[1].plot(df['Timestamp'], df['CYYR_Vis_Val'], 'r-x', markersize=4, label='CYYR')
ax[1].set_ylabel('SM'); ax[1].grid(True, alpha=0.3); ax[1].set_title('Sichtweite')

ax[2].plot(df['Timestamp'], df['MWS_QNH_Val'], 'g-o', markersize=4, label='MWS')
ax[2].plot(df['Timestamp'], df['CYYR_QNH_Val'], 'r-x', markersize=4, label='CYYR')
ax[2].set_ylabel('inHg'); ax[2].grid(True, alpha=0.3); ax[2].set_title('Luftdruck (QNH)')

# --- WIND MIT BÖEN ---
ax[3].plot(df['Timestamp'], df['MWS_Wind_Spd'], 'g-', label='MWS Wind')
ax[3].plot(df['Timestamp'], df['CYYR_Wind_Spd'], 'r-', label='CYYR Wind')

mask_mws_g = df['MWS_Wind_Gust'].notna()
mask_cyyr_g = df['CYYR_Wind_Gust'].notna()
if mask_mws_g.any():
    ax[3].scatter(df.loc[mask_mws_g, 'Timestamp'], df.loc[mask_mws_g, 'MWS_Wind_Gust'], color='green', marker='^', s=40, label='MWS Böen')
if mask_cyyr_g.any():
    ax[3].scatter(df.loc[mask_cyyr_g, 'Timestamp'], df.loc[mask_cyyr_g, 'CYYR_Wind_Gust'], color='red', marker='^', s=40, label='CYYR Böen')
ax[3].set_ylabel('kt'); ax[3].legend(loc='upper right', fontsize='small', ncol=2)
ax[3].grid(True, alpha=0.3); ax[3].set_title('Windgeschwindigkeit & Böen')

ax[4].scatter(df['Timestamp'], df['MWS_Wind_Dir'], c='g', s=15, label='MWS', alpha=0.7)
ax[4].scatter(df['Timestamp'], df['CYYR_Wind_Dir'], c='r', marker='x', s=15, label='CYYR', alpha=0.7)
ax[4].set_yticks([0,90,180,270,360]); ax[4].set_yticklabels(['N','E','S','W','N']); ax[4].grid(True, alpha=0.3)
ax[4].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); ax[4].set_title('Windrichtung')
plt.tight_layout(); plt.savefig(out_plot1)

print("- Erstelle Grafik 2 (Wetter & Wolken)...")
fig2, ax2 = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={'height_ratios':[1,1]})

ax2[0].fill_between(df['Timestamp'], df['MWS_Ceil'], 7000, color='green', alpha=0.1)
ax2[0].fill_between(df['Timestamp'], df['CYYR_Ceil'], 7000, color='red', alpha=0.1)
ax2[0].plot(df['Timestamp'], df['MWS_Ceil'], 'g-', linewidth=1.5, alpha=0.8, label='MWS Ceiling')
ax2[0].plot(df['Timestamp'], df['CYYR_Ceil'], 'r-', linewidth=1.5, alpha=0.8, label='CYYR Ceiling')

mws_cloud_x, mws_cloud_y = [], []
cyyr_cloud_x, cyyr_cloud_y = [], []
for idx, row in df.iterrows():
    ts = row['Timestamp']
    for hgt in get_all_clouds(row['MWS_Wx']):
        mws_cloud_x.append(ts); mws_cloud_y.append(hgt)
    for hgt in get_all_clouds(row['CYYR_Wx']):
        cyyr_cloud_x.append(ts); cyyr_cloud_y.append(hgt)

ax2[0].scatter(mws_cloud_x, mws_cloud_y, color='darkgreen', marker='_', s=150, lw=2, label='MWS Schichten')
ax2[0].scatter(cyyr_cloud_x, cyyr_cloud_y, color='darkred', marker='_', s=150, lw=2, label='CYYR Schichten')
ax2[0].set_ylabel('ft'); ax2[0].set_ylim(0, 7000); ax2[0].grid(True, alpha=0.3)
ax2[0].legend(loc='lower right', fontsize='small', ncol=2)
ax2[0].set_title('Wolkenschichten (Fläche = BKN/OVC, Striche = Alle gemeldeten Schichten)')

for i, code in enumerate(PLOT_TYPES):
    ax2[1].axhline(i, c='lightgray', alpha=0.3)
    t_m = df[df['MWS_Wx'].apply(lambda w: check_wx(w, code))]['Timestamp']
    t_c = df[df['CYYR_Wx'].apply(lambda w: check_wx(w, code))]['Timestamp']
    ax2[1].scatter(t_m, [i+0.15]*len(t_m), c='g', s=60, label='MWS' if i==0 else "")
    ax2[1].scatter(t_c, [i-0.15]*len(t_c), c='r', marker='x', s=60, label='CYYR' if i==0 else "")
ax2[1].set_yticks(range(len(PLOT_TYPES))); ax2[1].set_yticklabels(PLOT_TYPES)
ax2[1].grid(True, axis='x', alpha=0.3); ax2[1].legend(loc='upper right', ncol=2)
ax2[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); ax2[1].set_title('Signifikante Wettererscheinungen')
plt.tight_layout(); plt.savefig(out_plot2)

print("FERTIG! Alle Dateien im Arbeitsverzeichnis gespeichert.")