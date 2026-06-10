import sqlite3
import io
import os
import math
import datetime as dt
import zipfile
import textwrap
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image as PILImage
from scipy.signal import find_peaks

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.utils import ImageReader

# ---------------------------------------------------------------------
# CONFIGURAÇÃO DE IMAGENS E MAPEAMENTO DE FUROS
# ---------------------------------------------------------------------
IMG_DIR = "imagens"

FILES_CONFIG = {
    "LOGO": "logo",
    "COVER": "capa",
    "METOD_ROTOR": "metodologia_rotor",
    "METOD_MAPA": "metodologia_2",
    "METOD_BASE": "metodologia_3"
}

MAPA_FUROS = {
    "PS-TE": 4, "PS-CTE": 13, "PS-C": 21, "PS-CLE": 29, "PS-LE": 38,
    "SS-LE": 46, "SS-CLE": 55, "SS-C": 63, "SS-CTE": 71, "SS-TE": 80
}

# ---------------------------------------------------------------------
# PATCH 1 — Mapeamento fixo de relógios → região
# Série LT677x (campanhas antigas) e LT823x (campanhas novas).
# Relógios fora deste mapa recebem selectbox manual no sidebar.
# ---------------------------------------------------------------------
DIAL_REGION_MAP = {
    "BlueDialLT6777": "LE",
    "BlueDialLT6778": "CLE",
    "BlueDialLT6779": "C",
    "BlueDialLT6780": "CTE",
    "BlueDialLT6783": "TE",
    "BlueDialLT8230": "LE",
    "BlueDialLT8231": "CLE",
    "BlueDialLT8232": "C",
    "BlueDialLT8233": "CTE",
    "BlueDialLT8234": "TE",
}

def calculate_angle(furo):
    return (180.0 - (furo / 84.0) * 360.0) % 360.0

def get_stud_zone(stud_id: int) -> str:
    if 11 <= stud_id <= 31: return "PS"
    elif 32 <= stud_id <= 52: return "BA"
    elif 53 <= stud_id <= 73: return "SS"
    else: return "BF"

# ---------------------------------------------------------------------
# PARÂMETROS DA CAPA DO PDF
# ---------------------------------------------------------------------
COVER_LEFT_STRIP_W_CM = 4.8
COVER_IMG_H_RATIO = 0.7
COVER_IMG_TOP_PAD_CM = 0.0
COVER_TITLE_BAR_H_CM = 2.0
COVER_TITLE_BAR_COLOR = "#1F4E79"
COVER_BELOW_IMAGE_BG_COLOR = "#E0EFF1"
COVER_IMAGE_MODE = "cover"
COVER_IMAGE_CROP_ANCHOR = "top"
COVER_TITLE_BAR_Y_FROM_IMG_BOTTOM_CM = 1.7
COVER_LOGO_X_CM = 0.9
COVER_LOGO_Y_FROM_TOP_CM = 2.3
COVER_LOGO_W_CM = 3.5
COVER_LOGO_H_CM = 2

COVER_META_LABEL_X_CM = 1.6
COVER_META_VALUE_X_CM = 7.2
COVER_META_START_Y_FROM_BOTTOM_CM = 6.3
COVER_META_LINE_H_CM = 0.95
COVER_META_LABEL_SIZE = 9
COVER_META_VALUE_SIZE = 9

def find_image_path(base_name: str) -> Optional[str]:
    if not os.path.exists(IMG_DIR): return None
    try:
        for f in os.listdir(IMG_DIR):
            fl = f.lower()
            if fl.startswith(base_name.lower()) and fl.endswith(('.png', '.jpg', '.jpeg')):
                return os.path.join(IMG_DIR, f)
    except Exception: pass
    return None

def load_image_for_pdf(key_name: str):
    path = find_image_path(FILES_CONFIG.get(key_name, ""))
    if path:
        try:
            img = PILImage.open(path)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format)
            img_byte_arr.seek(0)
            return img_byte_arr
        except Exception: return None
    return None

# ---------------------------------------------------------------------
# Helpers de Cálculo de Calibre (Folga)
# ---------------------------------------------------------------------
def normalize_reference(ref: pd.Series) -> pd.Series:
    return ref.astype("string").str.upper().str.strip().replace({"B.F.": "BF", "B.A.": "BA"})

def dist_to_bf_perimeter(dist_mm: pd.Series, ref: pd.Series, perim_mm: float) -> pd.Series:
    half = perim_mm / 2.0
    d = pd.to_numeric(dist_mm, errors="coerce")
    r = normalize_reference(ref)
    out = np.where(r.eq("BA"), half + d, d)
    return pd.Series(np.clip(out, 0.0, perim_mm), index=dist_mm.index, dtype=float)

def theta_deg_from_perimeter(dist_bf_mm: pd.Series, perim_mm: float) -> pd.Series:
    d = pd.to_numeric(dist_bf_mm, errors="coerce")
    theta = 180.0 - (d / perim_mm) * 360.0
    return pd.Series(np.mod(theta, 360.0), index=dist_bf_mm.index, dtype=float)

def process_calibre_data(df_raw_cal: pd.DataFrame, perim_mm_val: float) -> pd.DataFrame:
    df = pd.DataFrame()
    cols = list(df_raw_cal.columns)
    df["Turbina"] = df_raw_cal[cols[0]].astype("string").str.strip()
    df["Blade"] = df_raw_cal[cols[1]].astype("string").str.strip()
    df["Campaign"] = df_raw_cal[cols[2]].astype("string").str.strip()
    df["Reference"] = df_raw_cal[cols[3]].astype("string").str.upper().str.strip()
    df["Distance"] = pd.to_numeric(df_raw_cal[cols[4]], errors="coerce")
    df["Shell"] = df_raw_cal[cols[5]].astype("string").str.upper().str.strip()
    df["M3H"] = pd.to_numeric(df_raw_cal[cols[6]], errors="coerce")
    df["M9H"] = pd.to_numeric(df_raw_cal[cols[7]], errors="coerce")
    df["GAP"] = pd.to_numeric(df_raw_cal[cols[8]], errors="coerce")
    perim = float(perim_mm_val)
    meio = perim / 2.0
    def calcular_distancia_absoluta(row):
        shell = row["Shell"]; ref = row["Reference"]; dist = row["Distance"]
        if pd.isna(dist): return 0.0
        if shell == "PS":
            if "BF" in ref: return dist
            if "BA" in ref: return meio - dist
        elif shell == "SS":
            if "BA" in ref: return meio + dist
            if "BF" in ref: return perim - dist
        return dist
    df["dist_bf_mm"] = df.apply(calcular_distancia_absoluta, axis=1)
    df["theta_deg"] = theta_deg_from_perimeter(df["dist_bf_mm"], perim)
    return df.rename(columns={"Reference": "Bordo Ref.", "Distance": "Distância (mm)", "GAP": "Gap (mm)", "Shell": "Casca"})

# =====================================================================
# CLASSIFICAÇÃO DE SEVERIDADE
# =====================================================================

def classify_severity_arthwind(delta_mm: float) -> str:
    """
    Classificação por sensor individual.
    Para classificação por pá no padrão Arthwind, use classify_blade_arthwind().
    """
    if delta_mm is None or (isinstance(delta_mm, float) and np.isnan(delta_mm)) or delta_mm <= 0:
        return "SEV0"
    d = float(abs(delta_mm))
    if d < 1.0:  return "SEV0"
    if d < 1.5:  return "SEV1"
    if d < 3.0:  return "SEV2"
    return "SEV5"

def classify_blade_arthwind(sensor_gaps: pd.Series) -> str:
    """
    Critério Arthwind — Magnitude + Extensão (por pá).

    Extensão          | Mag < 1,5mm | Mag ≥ 1,5mm < 3,0mm | Mag ≥ 3,0mm
    ------------------|-------------|----------------------|------------
    0 sensores  (0%)  | SEV0        | —                    | —
    1 sensor   (10%)  | SEV1        | SEV2                 | SEV5
    2 sensores (20%)  | SEV2        | SEV3                 | SEV5
    3–4 sensores      | SEV4        | SEV5                 | SEV5
    ≥5 sensores (50%) | SEV5        | SEV5                 | SEV5
    """
    gaps = pd.to_numeric(sensor_gaps, errors="coerce").dropna()
    if gaps.empty: return "SEV0"
    max_gap    = gaps.max()
    n_afetados = int((gaps >= 1.0).sum())
    if max_gap >= 3.0 or n_afetados >= 5: return "SEV5"
    if n_afetados >= 3 and max_gap >= 1.5: return "SEV5"
    if n_afetados >= 3: return "SEV4"
    if n_afetados == 2 and max_gap >= 1.5: return "SEV3"
    if n_afetados == 2 or max_gap >= 1.5: return "SEV2"
    if n_afetados == 1: return "SEV1"
    return "SEV0"

def classify_severity_enel(delta_mm: float) -> str:
    """Critérios ENEL por sensor: 0 / 0.5 / 1.0 / 2.0 / 2.5 / >2.5"""
    if delta_mm is None or (isinstance(delta_mm, float) and np.isnan(delta_mm)) or delta_mm <= 0:
        return "SEV0"
    d = float(abs(delta_mm))
    if d <= 0.5:  return "SEV1"
    if d <= 1.0:  return "SEV2"
    if d <= 2.0:  return "SEV3"
    if d <= 2.5:  return "SEV4"
    return "SEV5"

def get_classify_fn(modelo: str):
    return classify_severity_enel if modelo == "ENEL" else classify_severity_arthwind

def classify_severity(delta_mm: float) -> str:
    return classify_severity_arthwind(delta_mm)

def severity_color(sev: str) -> str:
    palette = {
        "SEV0": "#c6efce", "SEV1": "#a9d18e", "SEV2": "#ffd966",
        "SEV3": "#f4b183", "SEV4": "#ff8c00", "SEV5": "#ff0000"
    }
    return palette.get(sev, "#ffffff")

# ---------------------------------------------------------------------
# Configurações Streamlit e Barra Lateral
# ---------------------------------------------------------------------
st.set_page_config(page_title="Relatório GAP - Arthwind", layout="wide")
st.title("Dashboard_Gap de Insertos")

if not os.path.exists(IMG_DIR): st.sidebar.error(f"❌ Pasta '{IMG_DIR}' não encontrada!")
else: st.sidebar.success(f"📂 Pasta '{IMG_DIR}' encontrada.")

st.sidebar.header("Entrada de Dados")

@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None: return pd.DataFrame()
    try:
        fname = uploaded_file.name.lower()
        if fname.endswith(".csv"): return pd.read_csv(uploaded_file)
        if fname.endswith((".xlsx", ".xls")): return pd.read_excel(uploaded_file)
        if fname.endswith(".db"):
            with open("temp_db.db", "wb") as f: f.write(uploaded_file.getbuffer())
            conn = sqlite3.connect("temp_db.db")
            query_table = "SELECT name FROM sqlite_master WHERE type='table';"
            tables = pd.read_sql_query(query_table, conn)
            if tables.empty:
                st.error("O arquivo .db não contém tabelas.")
                return pd.DataFrame()
            table_name = tables.iloc[0]['name']
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            conn.close()
            return df
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {e}")
        return pd.DataFrame()
    return pd.DataFrame()

uploaded = st.sidebar.file_uploader("Upload Base BlueDial (Excel/CSV/DB)", type=["csv", "xlsx", "xls", "db"])

if uploaded is not None:
    if "last_file" not in st.session_state or st.session_state["last_file"] != uploaded.name:
        st.session_state["last_file"] = uploaded.name
        for key in ["results", "excel_bytes", "pdf_ind_bytes", "turb_sel"]:
            if key in st.session_state: del st.session_state[key]

df_raw = load_data(uploaded)

uploaded_calibre = st.sidebar.file_uploader("Upload Base Calibre (Excel/CSV) - Opcional", type=["xlsx", "xls", "csv"])

if uploaded_calibre is not None:
    if "last_calibre" not in st.session_state or st.session_state["last_calibre"] != uploaded_calibre.name:
        st.session_state["last_calibre"] = uploaded_calibre.name
        if "df_calibre_proc" in st.session_state: del st.session_state["df_calibre_proc"]

df_calibre = load_data(uploaded_calibre)

if df_raw.empty:
    st.info("Aguardando arquivo BlueDial...")
    st.stop()

required_cols = ["Turbina", "SN_da_Pa", "Casca", "Inspecao"]
missing = [c for c in required_cols if c not in df_raw.columns]
if missing:
    st.error(f"Colunas obrigatórias ausentes na base: {missing}")
    st.stop()

reading_cols = sorted([c for c in df_raw.columns if "Reading" in str(c)])
if not reading_cols:
    st.error("Não encontrei colunas de leitura (ex.: '... Reading (mm)').")
    st.stop()

dial_names = [(str(c).split(" Reading")[0].strip() if " Reading" in str(c) else str(c)) for c in reading_cols]

# ---------------------------------------------------------------------
# PATCH 2 — Mapeamento automático para relógios conhecidos
# ---------------------------------------------------------------------
st.sidebar.subheader("Mapeamento Relógio")
region_options = ["LE", "CLE", "C", "CTE", "TE"]
region_map: Dict[str, str] = {}
for dial in dial_names:
    if dial in DIAL_REGION_MAP:
        region_map[dial] = DIAL_REGION_MAP[dial]
    else:
        region_map[dial] = st.sidebar.selectbox(
            f"⚠️ {dial} (desconhecido):",
            region_options,
            key=f"rmap_{dial}"
        )

with st.sidebar.expander("🔍 Ver mapeamento ativo", expanded=False):
    for dial in dial_names:
        origem = "fixo" if dial in DIAL_REGION_MAP else "manual"
        st.caption(f"{'🔒' if origem == 'fixo' else '✏️'} `{dial}` → **{region_map[dial]}** ({origem})")

st.sidebar.markdown("---")
st.sidebar.subheader("Parâmetros Calibre de Folga")
perim_mm = st.sidebar.number_input("Perímetro de referência (mm)", min_value=1000.0, max_value=20000.0, value=5900.0, step=10.0, key="in_perim")
bin_mm = st.sidebar.number_input("BIN (mm) p/ consolidado", min_value=10.0, max_value=1000.0, value=200.0, step=10.0, key="in_bin")

st.sidebar.markdown("---")
st.sidebar.subheader("Performance (Dash)")
max_plots_view = st.sidebar.slider("Limite de Gráficos Simultâneos", 1, 50, 10, 1)

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros Analíticos")

# PATCH 3 — Slider de corte inicial de transiente
trim_inicial = st.sidebar.slider(
    "Corte inicial (pontos)",
    min_value=0, max_value=2000, value=0, step=10,
    help="Remove N pontos do início de cada sinal para eliminar o transiente de partida.",
    key="trim_inicial"
)

enable_hampel = st.sidebar.checkbox("Filtro Hampel", value=False)
hampel_window = st.sidebar.slider("Janela Hampel", 5, 101, 21, 2)
hampel_n_sigma = st.sidebar.slider("Limite Hampel (sigma)", 1.0, 6.0, 3.0, 0.5)
enable_deriv = st.sidebar.checkbox("Filtro Derivada", value=False)
deriv_threshold = st.sidebar.slider("Limite Derivada", 0.01, 5.0, 2.0, 0.1)
enable_peak_filter = st.sidebar.checkbox("Filtro Peak", value=True)
peak_threshold_mm = st.sidebar.slider("Limite Peak (mm)", 0.1, 10.0, 2.0, 0.1)
usar_align = st.sidebar.checkbox("Centralizar (Média Zero)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Seleção Base")
turbinas = sorted(df_raw["Turbina"].dropna().unique().tolist())
if "turb_sel" not in st.session_state: st.session_state["turb_sel"] = [turbinas[0]] if turbinas else []
c_sel, c_clr, c_inv = st.sidebar.columns(3)
with c_sel:
    if st.button("✅ Todas", key="b_all", use_container_width=True): st.session_state["turb_sel"] = turbinas; st.rerun()
with c_clr:
    if st.button("🧹 Nenhuma", key="b_non", use_container_width=True): st.session_state["turb_sel"] = []; st.rerun()
with c_inv:
    if st.button("🔁 Inverter", key="b_inv", use_container_width=True):
        cur = set(st.session_state["turb_sel"])
        st.session_state["turb_sel"] = [t for t in turbinas if t not in cur]; st.rerun()

turb_sel = st.sidebar.multiselect("Turbinas", turbinas, key="turb_sel")
df_turb = df_raw[df_raw["Turbina"].isin(turb_sel)].copy() if turb_sel else df_raw.copy()

blades = sorted(df_turb["SN_da_Pa"].dropna().astype(str).unique().tolist())
blades_sel = st.sidebar.multiselect("Pás", blades, default=blades)

insps = sorted(df_turb["Inspecao"].dropna().unique().tolist())
insps_sel = st.sidebar.multiselect("Campanhas", insps, default=insps)

st.sidebar.markdown("---")
st.sidebar.subheader("🔩 Hardware: Studs Ausentes")
studs_ausentes_dict = {}
if blades_sel:
    for b in blades_sel:
        studs_ausentes_dict[b] = st.sidebar.multiselect(f"Studs Pá {b} (0-83):", options=list(range(84)), default=[], key=f"studs_{b}")
else:
    st.sidebar.info("Selecione uma pá acima para registrar studs ausentes.")

# =====================================================================
# PARTE 2: MOTOR MATEMÁTICO
# =====================================================================

def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty: return None
    cols = [str(c) for c in df.columns]
    lower_map = {c: c.lower().strip() for c in cols}
    exact_priority = ["data", "date", "data_coleta", "inspection_date", "data_inspecao"]
    for c in cols:
        if lower_map[c] in exact_priority: return c
    for c in cols:
        lc = lower_map[c]
        if ("data" in lc or "date" in lc) and all(bad not in lc for bad in ["update", "atual", "criado"]): return c
    return None

def severity_recommendation(sev: str, modelo: str = "Arthwind") -> Tuple[str, dt.timedelta]:
    if modelo == "ENEL":
        recs = {
            "SEV0": ("6 Months",           dt.timedelta(days=182)),
            "SEV1": ("6 Months",           dt.timedelta(days=182)),
            "SEV2": ("3 Months",           dt.timedelta(days=91)),
            "SEV3": ("1 Month",            dt.timedelta(days=30)),
            "SEV4": ("15 Days",            dt.timedelta(days=15)),
            "SEV5": ("STOP WTG",           dt.timedelta(days=0)),
        }
    else:
        recs = {
            "SEV0": ("4 Months",           dt.timedelta(days=120)),
            "SEV1": ("2 Months",           dt.timedelta(days=60)),
            "SEV2": ("1 Month",            dt.timedelta(days=30)),
            "SEV3": ("15 Days",            dt.timedelta(days=15)),
            "SEV4": ("Gap Gauge or Weekly", dt.timedelta(days=7)),
            "SEV5": ("Stop Turbine",       dt.timedelta(days=0)),
        }
    return recs.get(sev, ("Review", dt.timedelta(days=90)))

def _insp_num(x) -> float:
    s = str(x)
    m = pd.Series([s]).str.extract(r'(\d+)')[0].iloc[0]
    try: return float(m)
    except Exception: return np.nan

def pick_latest_rows(df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    if df is None or df.empty: return df
    tmp = df.copy()
    tmp["_Data"] = pd.to_datetime(tmp.get("Data"), errors="coerce", dayfirst=True)
    tmp["_HasDate"] = tmp["_Data"].notna().astype(int)
    tmp["_InspNum"] = tmp["Inspecao"].map(_insp_num)
    tmp["_InspecaoStr"] = tmp["Inspecao"].astype(str)
    tmp = tmp.sort_values(group_keys + ["_HasDate", "_Data", "_InspNum", "_InspecaoStr"])
    latest = tmp.groupby(group_keys, as_index=False).tail(1)
    return latest.drop(columns=["_Data", "_HasDate", "_InspNum", "_InspecaoStr"], errors="ignore")

def trim_and_rebase(g: pd.DataFrame, xcol="Ponto", ycol="Valor_mm", eps=1e-12) -> Optional[pd.DataFrame]:
    if g is None or g.empty: return None
    g = g.sort_values(xcol).copy()
    y = pd.to_numeric(g[ycol], errors="coerce").interpolate(limit_area="inside")
    valid = y.notna() & (y.abs() > eps)
    if not valid.any(): return None
    start_pos = int(np.argmax(valid.values))
    # PATCH 4 — Corte de transiente inicial
    start_pos += trim_inicial
    end_pos = int(len(valid) - 1 - np.argmax(valid.values[::-1]))
    if start_pos >= end_pos: return None
    g = g.iloc[start_pos:end_pos + 1].copy()
    g[ycol] = y.iloc[start_pos:end_pos + 1].values
    g[xcol] = pd.to_numeric(g[xcol], errors="coerce") - float(pd.to_numeric(g[xcol], errors="coerce").iloc[0])
    return g

def process_data_core(df_target: pd.DataFrame) -> pd.DataFrame:
    frames = []
    date_col = detect_date_column(df_target)

    for col, dial in zip(reading_cols, dial_names):
        # PATCH 5 — Ignora colunas sem dados neste subset
        if df_target[col].dropna().empty:
            continue

        base_cols = ["Turbina", "SN_da_Pa", "Casca", "Inspecao"]
        if date_col is not None and date_col in df_target.columns: base_cols.append(date_col)
        base = df_target[base_cols].copy()
        if date_col is not None and date_col in base.columns:
            base["Data"] = pd.to_datetime(base[date_col], errors="coerce", dayfirst=True)
            base.drop(columns=[date_col], inplace=True, errors="ignore")
        elif "Data" in df_target.columns:
            base["Data"] = pd.to_datetime(df_target["Data"], errors="coerce")
        else:
            base["Data"] = pd.NaT

        base["Relogio"] = dial
        base["Regiao"] = region_map.get(dial, "C")
        base["Valor_mm"] = pd.to_numeric(df_target[col], errors="coerce")
        base["Ponto"] = base.groupby(["Turbina", "SN_da_Pa", "Casca", "Inspecao"]).cumcount()
        frames.append(base)

    # PATCH 6 — Guard frames vazios
    if not frames: return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    long_df.rename(columns={"SN_da_Pa": "Blade"}, inplace=True)

    if enable_hampel:
        def _hampel(g):
            s = g["Valor_mm"]
            if s.notna().sum() < 3: return g
            med = s.rolling(window=hampel_window, center=True, min_periods=1).median()
            mad = (s - med).abs().rolling(window=hampel_window, center=True, min_periods=1).median() * 1.4826
            g.loc[(s - med).abs() > (hampel_n_sigma * mad), "Valor_mm"] = np.nan
            return g
        long_df = long_df.groupby(["Turbina", "Blade", "Casca", "Regiao", "Inspecao"], group_keys=False).apply(_hampel)

    if enable_deriv:
        def _deriv(g):
            g = g.sort_values("Ponto")
            g.loc[g["Valor_mm"].diff().abs() > deriv_threshold, "Valor_mm"] = np.nan
            return g
        long_df = long_df.groupby(["Turbina", "Blade", "Casca", "Regiao", "Inspecao"], group_keys=False).apply(_deriv)

    if enable_peak_filter:
        def _peak(g):
            med = g["Valor_mm"].rolling(window=5, center=True, min_periods=1).median()
            g.loc[(g["Valor_mm"] - med).abs() > peak_threshold_mm, "Valor_mm"] = np.nan
            return g
        long_df = long_df.groupby(["Turbina", "Blade", "Casca", "Regiao", "Inspecao"], group_keys=False).apply(_peak)

    if usar_align:
        long_df["Valor_mm"] = long_df.groupby(["Turbina", "Blade", "Casca", "Regiao", "Inspecao"])["Valor_mm"].transform(lambda x: x - x.mean())

    return long_df

def compute_cycle_delta(g: pd.DataFrame) -> Tuple[float, int]:
    g = g.sort_values("Ponto").copy()
    s = pd.to_numeric(g["Valor_mm"], errors="coerce").interpolate(limit_area="inside")
    if s.notna().sum() < 10: return float("nan"), 0
    y = s.values.astype(float)
    global_amp = np.nanmax(y) - np.nanmin(y)
    if global_amp < 0.05: return 0.0, 0
    win = max(5, int(len(y) * 0.02)) | 1
    smooth = pd.Series(y).rolling(window=win, center=True, min_periods=1).mean().values
    peaks, _ = find_peaks(smooth, prominence=global_amp * 0.1)
    valleys, _ = find_peaks(-smooth, prominence=global_amp * 0.1)
    if len(peaks) == 0 or len(valleys) == 0: return global_amp, 1
    amplitudes = []
    turning_points = np.sort(np.concatenate([peaks, valleys]))
    for i in range(len(turning_points) - 1):
        amp = abs(smooth[turning_points[i]] - smooth[turning_points[i + 1]])
        if amp > 0.25 * global_amp: amplitudes.append(amp)
    if not amplitudes: return global_amp, 1
    return float(np.mean(amplitudes)), max(1, int(len(amplitudes) // 2))

def run_analysis(df_in: pd.DataFrame, full_process=False, t_sel=None, b_sel=None, i_sel=None, modelo="Arthwind"):
    _classify = get_classify_fn(modelo)
    _turb_sel   = t_sel if t_sel is not None else turb_sel
    _blades_sel = b_sel if b_sel is not None else blades_sel
    _insps_sel  = i_sel if i_sel is not None else insps_sel

    df_subset = df_in.copy() if full_process else df_in[
        (df_in["Turbina"].astype(str).isin([str(t) for t in _turb_sel])) &
        (df_in["SN_da_Pa"].astype(str).isin([str(b) for b in _blades_sel])) &
        (df_in["Inspecao"].astype(str).isin([str(i) for i in _insps_sel]))
    ].copy()

    if df_subset.empty: return None

    long_df = process_data_core(df_subset)
    # PATCH 7 — Guard long_df vazio
    if long_df.empty: return None

    delta_rows, verify_items = [], []
    for keys, g in long_df.groupby(["Turbina", "Blade", "Casca", "Regiao", "Relogio", "Inspecao"]):
        turb, blade, casca, reg, relogio, insp = keys
        d, n = compute_cycle_delta(g)
        dmax = pd.to_datetime(g["Data"], errors="coerce", dayfirst=True).max() if "Data" in g.columns else pd.NaT
        delta_rows.append({
            "Turbina": turb, "Blade": blade, "Casca": casca, "Regiao": reg,
            "Relogio": relogio, "Inspecao": insp,
            "Delta_medio_ciclo_mm": d, "N_ciclos": n, "Data": dmax
        })
        if not full_process:
            verify_items.append({
                "turbina": turb, "blade": blade, "casca": casca, "regiao": reg,
                "relogio": relogio, "inspecao": insp,
                "data": g.sort_values("Ponto")[["Ponto", "Valor_mm"]], "delta": d
            })

    delta_summary = pd.DataFrame(delta_rows)
    if full_process: return {"delta_summary": delta_summary}

    camp_dates = delta_summary.groupby("Inspecao", as_index=False)["Data"].max().dropna(subset=["Data"])
    camp_dates = camp_dates.sort_values("Inspecao", key=lambda s: s.map(_insp_num).fillna(1e9))
    campaign_dates_str = [f"{row['Inspecao']}: {pd.to_datetime(row['Data']).strftime('%d-%m-%Y')}" for _, row in camp_dates.iterrows()]
    last_overall = camp_dates["Data"].max() if not camp_dates.empty else pd.NaT
    cover_date = pd.to_datetime(last_overall).strftime("%d-%m-%Y") if pd.notna(last_overall) else dt.datetime.now().strftime("%d-%m-%Y")

    pdf_detailed_data = []
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for blade in _blades_sel:
        df_b = long_df[long_df["Blade"].astype(str) == str(blade)]
        if df_b.empty: continue
        sensors_data_list = []
        for (casca, reg, relogio), g_sensor in df_b.groupby(["Casca", "Regiao", "Relogio"]):
            stats_rows = []
            fig, ax = plt.subplots(figsize=(6, 3.2))
            has_data, idx_color = False, 0
            for (tb, isp), g_trace in g_sensor.groupby(["Turbina", "Inspecao"]):
                g_trim = trim_and_rebase(g_trace, xcol="Ponto", ycol="Valor_mm")
                if g_trim is None or g_trim.empty: continue
                lbl = f"{isp}" if len(_turb_sel) == 1 else f"{tb}-{isp}"
                c = colors_list[idx_color % len(colors_list)]
                ax.plot(g_trim["Ponto"], g_trim["Valor_mm"], label=lbl, linewidth=1.5, color=c)
                val_d = delta_summary[
                    (delta_summary["Turbina"].astype(str) == str(tb)) &
                    (delta_summary["Blade"].astype(str) == str(blade)) &
                    (delta_summary["Casca"].astype(str) == str(casca)) &
                    (delta_summary["Regiao"].astype(str) == str(reg)) &
                    (delta_summary["Relogio"].astype(str) == str(relogio)) &
                    (delta_summary["Inspecao"].astype(str) == str(isp))
                ]["Delta_medio_ciclo_mm"].max()
                stats_rows.append({"Campanha": lbl, "Gap (mm)": val_d})
                idx_color += 1; has_data = True
            if has_data:
                sensor_name = f"{casca}-{reg} ({relogio})"
                ax.set_title(sensor_name, fontsize=10, fontweight='bold', pad=8)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(fontsize=7, loc='best')
                plt.tight_layout()
                sensors_data_list.append({"sensor": sensor_name, "fig": fig, "stats": stats_rows})
            else: plt.close(fig)
        pdf_detailed_data.append((blade, sensors_data_list))

    latest_sensors = pick_latest_rows(delta_summary, ["Turbina", "Blade", "Casca", "Regiao", "Relogio"])

    def _classify_blade(group):
        if modelo == "ENEL": return _classify(group["Delta_medio_ciclo_mm"].max())
        else: return classify_blade_arthwind(group["Delta_medio_ciclo_mm"])

    blade_latest = latest_sensors.groupby(["Turbina", "Blade"], as_index=False).agg(
        Delta_latest_max_mm=("Delta_medio_ciclo_mm", "max"), Last_Date=("Data", "max"))
    sev_por_pa = (latest_sensors.groupby(["Turbina", "Blade"]).apply(_classify_blade)
        .reset_index().rename(columns={0: "Severity"}))
    blade_latest = blade_latest.merge(sev_por_pa, on=["Turbina", "Blade"], how="left")

    grouped_all = blade_latest[["Turbina", "Blade", "Delta_latest_max_mm", "Severity"]].copy()
    grouped_all.rename(columns={"Delta_latest_max_mm": "Delta_max_mm"}, inplace=True)

    rec_txt, next_dates = [], []
    for sev, d_last in zip(blade_latest["Severity"], blade_latest["Last_Date"]):
        r_txt, delta_t = severity_recommendation(sev, modelo)
        rec_txt.append(r_txt)
        next_dates.append(pd.to_datetime(d_last) + delta_t if pd.notna(d_last) else pd.NaT)
    blade_latest["Recommendation"], blade_latest["Next_Inspection"] = rec_txt, next_dates

    turbine_latest = blade_latest.groupby(["Turbina"], as_index=False).agg(
        Delta_latest_max_mm=("Delta_latest_max_mm", "max"), Last_Date=("Last_Date", "max"))
    _sev_order = {"SEV0": 0, "SEV1": 1, "SEV2": 2, "SEV3": 3, "SEV4": 4, "SEV5": 5}
    worst_sev = (blade_latest.groupby("Turbina")["Severity"]
        .apply(lambda s: max(s, key=lambda x: _sev_order.get(x, 0)))
        .reset_index().rename(columns={"Severity": "Severity"}))
    turbine_latest = turbine_latest.merge(worst_sev, on="Turbina", how="left")
    turbine_latest["Recommendation"] = turbine_latest["Severity"].apply(lambda s: severity_recommendation(s, modelo)[0])

    reinspection = delta_summary.groupby(["Turbina", "Blade"], as_index=False).agg(
        Reinspections=("Inspecao", "nunique"), First_Date=("Data", "min"), Last_Date=("Data", "max"))
    reinspection = reinspection.merge(
        blade_latest[["Turbina", "Blade", "Delta_latest_max_mm", "Severity", "Recommendation", "Next_Inspection"]],
        on=["Turbina", "Blade"], how="left")

    critical_blades   = blade_latest.sort_values("Delta_latest_max_mm", ascending=False).head(15).copy()
    critical_turbines = turbine_latest.sort_values("Delta_latest_max_mm", ascending=False).head(15).copy()

    return {
        "meta": {
            "turb": ", ".join(map(str, _turb_sel)) if _turb_sel else "All",
            "blades": _blades_sel, "insps": _insps_sel,
            "date": cover_date, "campaign_dates": campaign_dates_str, "modelo": modelo
        },
        "delta_summary": delta_summary,
        "severity_by_blade_latest": blade_latest,
        "severity_by_turbine_latest": turbine_latest,
        "reinspection_table": reinspection,
        "critical_blades": critical_blades,
        "critical_turbines": critical_turbines,
        "pdf_detailed_data": pdf_detailed_data,
        "verify_data": verify_items,
        "latest_sensors": latest_sensors,
        "classify_fn": _classify
    }

# ---------------------------------------------------------------------
# Helpers de Gráficos e PDFs
# ---------------------------------------------------------------------
def create_radar_chart(blade, latest_sensors, studs_ausentes_dict, engine='plotly', classify_fn=None):
    if classify_fn is None: classify_fn = classify_severity_arthwind
    df_latest_b = latest_sensors[latest_sensors["Blade"].astype(str) == str(blade)].copy()
    studs_ausentes = studs_ausentes_dict.get(str(blade), []) if isinstance(studs_ausentes_dict, dict) else []

    if engine == 'plotly':
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[1.1, 0, 1.1], theta=[0, 0, 180], mode='lines',
            line=dict(color='black', width=1.5, dash='dash'),
            hoverinfo='none', showlegend=False
        ))
        for f in range(84):
            ang = calculate_angle(f)
            fig.add_trace(go.Scatterpolar(
                r=[0.9, 1.1], theta=[ang, ang], mode='lines',
                line=dict(color='grey', width=0.5), showlegend=False, hoverinfo='none'
            ))
        for sens_key in MAPA_FUROS.keys():
            row = df_latest_b[(df_latest_b["Casca"] + "-" + df_latest_b["Regiao"]) == sens_key]
            gap = row["Delta_medio_ciclo_mm"].max() if not row.empty else np.nan
            if pd.isna(gap): continue
            sev = classify_fn(gap)
            cor = severity_color(sev)
            ang = calculate_angle(MAPA_FUROS[sens_key])
            fig.add_trace(go.Scatterpolar(
                r=[1.1], theta=[ang], mode='markers+text',
                marker=dict(size=18, color=cor, line=dict(color='black', width=1)),
                text=[f"<b>{gap:.1f}</b>"], textposition="top center",
                hoverinfo='text',
                hovertext=f"<b>{sens_key}</b><br>Gap: {gap:.1f}mm<br>Sev: {sev}",
                name=sens_key
            ))
        for stud in studs_ausentes:
            ang = calculate_angle(stud)
            fig.add_trace(go.Scatterpolar(
                r=[1.1], theta=[ang], mode='markers',
                marker=dict(size=12, symbol='x', color='black', line=dict(width=2)),
                hoverinfo='text',
                hovertext=f"Stud {stud} Ausente (Zona {get_stud_zone(stud)})",
                name='Stud Ausente', showlegend=False
            ))
        fig.update_layout(
            polar=dict(
                angularaxis=dict(direction="counterclockwise", rotation=0, showticklabels=False),
                radialaxis=dict(showticklabels=False, range=[0, 1.3])
            ),
            showlegend=False, height=450,
            margin=dict(l=30, r=30, t=30, b=30),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)'
        )
        fig.add_annotation(x=0.5,  y=1.05,  text="<b>PS</b>", showarrow=False, xref="paper", yref="paper", font=dict(size=14))
        fig.add_annotation(x=0.5,  y=-0.05, text="<b>SS</b>", showarrow=False, xref="paper", yref="paper", font=dict(size=14))
        fig.add_annotation(x=-0.05, y=0.5,  text="<b>TE</b>", showarrow=False, xref="paper", yref="paper", font=dict(size=14))
        fig.add_annotation(x=1.05,  y=0.5,  text="<b>LE</b>", showarrow=False, xref="paper", yref="paper", font=dict(size=14))
        return fig

def create_dual_line_chart(blade, df_viz_cli, colors_list, data_str):
    df_b = df_viz_cli[df_viz_cli["Blade"].astype(str) == str(blade)]
    fig, (ax_ps, ax_ss) = plt.subplots(1, 2, figsize=(10, 3.5))
    insp_max = df_b["Inspecao"].max() if not df_b.empty else "Data Indisponível"
    ps_idx, ss_idx = 0, 0
    for sens in sorted(df_b["Sensor"].unique()):
        g_leg = df_b[(df_b["Sensor"] == sens) & (df_b["Inspecao"] == insp_max)].copy()
        g_leg["Valor_mm"] = pd.to_numeric(g_leg["Valor_mm"], errors="coerce")
        g_trim = trim_and_rebase(g_leg, xcol="Ponto", ycol="Valor_mm")
        if g_trim is not None and not g_trim.empty:
            if sens.startswith("PS"):
                c = colors_list[ps_idx % len(colors_list)]
                ax_ps.plot(g_trim["Ponto"], g_trim["Valor_mm"], label=sens, linewidth=1.5, color=c)
                ps_idx += 1
            elif sens.startswith("SS"):
                c = colors_list[ss_idx % len(colors_list)]
                ax_ss.plot(g_trim["Ponto"], g_trim["Valor_mm"], label=sens, linewidth=1.5, color=c)
                ss_idx += 1
    ax_ps.set_title(f"PS Curves - {data_str}", fontsize=9, fontweight='bold', pad=8)
    ax_ps.grid(True, linestyle='--', alpha=0.5); ax_ps.legend(fontsize=7, loc='upper right')
    ax_ss.set_title(f"SS Curves - {data_str}", fontsize=9, fontweight='bold', pad=8)
    ax_ss.grid(True, linestyle='--', alpha=0.5); ax_ss.legend(fontsize=7, loc='upper right')
    plt.tight_layout()
    img_io = io.BytesIO()
    fig.savefig(img_io, format='png', dpi=150, bbox_inches='tight')
    img_io.seek(0); plt.close(fig)
    return img_io

def create_radar_chart_and_table(blade, latest_sensors, studs_ausentes_dict, classify_fn=None):
    if classify_fn is None: classify_fn = classify_severity_arthwind
    fig_polar, ax_polar = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    theta_circle = np.linspace(0, 2 * math.pi, 200)
    ax_polar.plot(theta_circle, [1.1] * 200, color='black', linewidth=1)
    ax_polar.plot(theta_circle, [0.9] * 200, color='black', linewidth=1)
    ax_polar.plot([0, 0], [0, 1.1], color='black', linewidth=1.5, linestyle='--')
    ax_polar.plot([math.pi, math.pi], [0, 1.1], color='black', linewidth=1.5, linestyle='--')
    for f in range(84):
        ang_rad = calculate_angle(f) * math.pi / 180.0
        ax_polar.plot([ang_rad, ang_rad], [0.9, 1.1], color='grey', linewidth=0.5)
    df_latest_b = latest_sensors[latest_sensors["Blade"].astype(str) == str(blade)].copy()
    table_data = [["Sensor", "Gap (mm)", "Severity"]]
    sorted_sensors = df_latest_b.copy()
    sorted_sensors['Furo'] = sorted_sensors.apply(lambda row: MAPA_FUROS.get(f"{row['Casca']}-{row['Regiao']}"), axis=1)
    sorted_sensors = sorted_sensors.dropna(subset=['Furo']).sort_values('Furo')
    for _, row in sorted_sensors.iterrows():
        sens_key = f"{row['Casca']}-{row['Regiao']}"
        furo = MAPA_FUROS.get(sens_key)
        if furo is None: continue
        angulo_rad = calculate_angle(furo) * math.pi / 180.0
        val_gap = row["Delta_medio_ciclo_mm"]
        sev = classify_fn(val_gap)
        ax_polar.scatter(angulo_rad, 1, color=severity_color(sev), s=150, edgecolors='black', zorder=10)
        ax_polar.text(angulo_rad, 1.35, f"{val_gap:.1f}mm\n{sev}\n{sens_key}", ha='center', va='center', fontsize=9.5, fontweight='bold', zorder=11)
        table_data.append([sens_key, f"{val_gap:.1f}".replace(".", ",") if pd.notna(val_gap) else "-", sev])
    studs_ausentes = studs_ausentes_dict.get(str(blade), []) if isinstance(studs_ausentes_dict, dict) else []
    for stud in studs_ausentes:
        angulo_rad = calculate_angle(stud) * math.pi / 180.0
        ax_polar.scatter(angulo_rad, 1.1, color='black', marker='x', s=100, linewidths=2, zorder=12)
    ax_polar.text(90  * math.pi / 180.0, 1.65, "PS", ha='center', va='center', fontsize=12, fontweight='bold')
    ax_polar.text(270 * math.pi / 180.0, 1.65, "SS", ha='center', va='center', fontsize=12, fontweight='bold')
    ax_polar.text(180 * math.pi / 180.0, 1.65, "TE", ha='right',  va='center', fontsize=12, fontweight='bold')
    ax_polar.text(0   * math.pi / 180.0, 1.65, "LE", ha='left',   va='center', fontsize=12, fontweight='bold')
    ax_polar.set_ylim(0, 1.8); ax_polar.axis('off'); plt.tight_layout()
    img_io_polar = io.BytesIO()
    fig_polar.savefig(img_io_polar, format='png', dpi=150, bbox_inches='tight', transparent=True)
    img_io_polar.seek(0); plt.close(fig_polar)
    return img_io_polar, table_data

def get_proportional_image(img_bytes, max_w, max_h):
    try:
        img_bytes.seek(0); img_reader = ImageReader(img_bytes)
        iw, ih = img_reader.getSize(); aspect = ih / float(iw)
        w, h = max_w, max_w * aspect
        if h > max_h: h, w = max_h, max_h / aspect
        img_bytes.seek(0)
        return Image(img_bytes, width=w, height=h)
    except Exception:
        img_bytes.seek(0)
        return Image(img_bytes, width=max_w, height=max_h)

def _draw_wrapped(canvas, text: str, x: float, y: float, max_chars: int, line_h_pt: float):
    if not text: return
    for i, line in enumerate(textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)):
        canvas.drawString(x, y - i * line_h_pt, line)

def draw_image_cover(canvas, img_reader, x: float, y: float, w: float, h: float):
    try:
        iw, ih = img_reader.getSize()
        if iw <= 0 or ih <= 0: return
        scale = max(w / float(iw), h / float(ih))
        sw, sh = float(iw) * scale, float(ih) * scale
        dx, dy = x - (sw - w) / 2.0, y - (sh - h) / 2.0
        p = canvas.beginPath(); p.rect(x, y, w, h)
        canvas.saveState(); canvas.clipPath(p, stroke=0, fill=0)
        canvas.drawImage(img_reader, dx, dy, width=sw, height=sh, mask='auto')
        canvas.restoreState()
    except Exception: pass

def _create_cover_and_intro(doc, results, h1, normal, modelo="Arthwind", windfarm=None, customer=None):
    meta = results["meta"]
    sev_df = results.get("severity_by_blade_latest", results.get("severity_by_blade"))
    _classify = results.get("classify_fn", get_classify_fn(modelo))
    turbina_txt = meta.get("turb", "-")
    blades_list_txt = ", ".join(map(str, meta.get("blades", [])))
    camp_dates_txt = " | ".join(meta.get("campaign_dates", [])) or "-"
    windfarm_txt = windfarm if windfarm else ("COMPLEXO EÓLICO SERRA AZUL" if modelo == "ENEL" else "COMPLEXO EÓLICO ASSURUÁ")
    customer_txt = customer if customer else ("ENEL" if modelo == "ENEL" else "SERENA")

    def draw_cover_full(canvas, _doc):
        canvas.saveState()
        page_w, page_h = A4
        canvas.setFillColor(colors.white); canvas.rect(0, 0, page_w, page_h, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor(COVER_BELOW_IMAGE_BG_COLOR)); canvas.rect(0, 0, page_w, page_h, stroke=0, fill=1)
        cover_bytes = load_image_for_pdf("COVER")
        img_h = page_h * COVER_IMG_H_RATIO
        y = page_h - img_h - (COVER_IMG_TOP_PAD_CM * cm)
        if cover_bytes:
            try: draw_image_cover(canvas, ImageReader(cover_bytes), 0, y, page_w, img_h)
            except Exception: pass
        else:
            canvas.setFillColor(colors.HexColor("#dbeef3")); canvas.rect(0, y, page_w, img_h, stroke=0, fill=1)
        logo_bytes = load_image_for_pdf("LOGO")
        if logo_bytes:
            try: canvas.drawImage(ImageReader(logo_bytes), COVER_LOGO_X_CM*cm, page_h-(COVER_LOGO_Y_FROM_TOP_CM*cm), width=COVER_LOGO_W_CM*cm, height=COVER_LOGO_H_CM*cm, mask='auto', preserveAspectRatio=True)
            except Exception: pass
        bar_h = COVER_TITLE_BAR_H_CM * cm
        bar_y = (page_h-(page_h*COVER_IMG_H_RATIO)-(COVER_IMG_TOP_PAD_CM*cm))+(COVER_TITLE_BAR_Y_FROM_IMG_BOTTOM_CM*cm)
        bar_x = _doc.leftMargin; bar_w = page_w - _doc.leftMargin - _doc.rightMargin
        canvas.setFillColor(colors.HexColor(COVER_TITLE_BAR_COLOR)); canvas.rect(bar_x, bar_y, bar_w, bar_h, stroke=0, fill=1)
        canvas.setFillColor(colors.white); canvas.setFont("Helvetica-Bold", 18)
        canvas.drawCentredString(page_w/2, bar_y+bar_h*0.62, "ROOT GAP MEASUREMENT INSPECTION")
        canvas.drawCentredString(page_w/2, bar_y+bar_h*0.22, "REPORT")
        y0 = COVER_META_START_Y_FROM_BOTTOM_CM * cm; line_h = COVER_META_LINE_H_CM * cm
        labels = ["Wtg Serial Number:", "Windfarm:", "Campaign Dates:", "Date (dd-mm-yyyy):", "Blade Model:", "Customer:"]
        values = [turbina_txt, windfarm_txt, camp_dates_txt, meta.get("date", "-"), "LM47.6", customer_txt]
        canvas.setFont("Helvetica-Bold", COVER_META_LABEL_SIZE); canvas.setFillColor(colors.HexColor("#1F4E79"))
        for i, lab in enumerate(labels): canvas.drawString(COVER_META_LABEL_X_CM*cm, y0-i*line_h, lab)
        canvas.setFont("Helvetica", COVER_META_VALUE_SIZE); canvas.setFillColor(colors.black)
        for i, val in enumerate(values):
            yy = y0 - i * line_h
            if i == 2: _draw_wrapped(canvas, str(val), COVER_META_VALUE_X_CM*cm, yy, max_chars=65, line_h_pt=10)
            else: canvas.drawString(COVER_META_VALUE_X_CM*cm, yy, str(val))
        canvas.restoreState()

    story = [PageBreak(), Paragraph("1. Summary", h1)]
    t_toc = Table([["Section","Page"],["2. Introduction","3"],["3. Conclusion","3"],["4. Methodology","4"],["5. Scope","5"],["6. Damages Categorization","5"],["7. Inspection Evidence","6+"]], colWidths=[14*cm, 2*cm])
    t_toc.setStyle(TableStyle([('LINEBELOW',(0,0),(-1,-1),0.5,colors.lightgrey)]))
    story.append(t_toc); story.append(PageBreak())

    story.append(Paragraph("2. Introduction", h1))
    story.append(Paragraph(f"On {meta.get('date','-')}, a gap measurement inspection was performed on LM47.6 model blades, serial numbers {blades_list_txt}, installed on the {turbina_txt} wind turbine located at the {windfarm_txt}.", normal))
    story.append(Spacer(1, 1*cm))

    story.append(Paragraph("3. Conclusion", h1))
    if sev_df is not None and not sev_df.empty:
        conc_data = [["Trb-Blade", "Max Gap (mm)", "Severity"]]
        worst_sev_idx = 0; s_map = {"SEV0":0,"SEV1":1,"SEV2":2,"SEV3":3,"SEV4":4,"SEV5":5}
        for _, row in sev_df.iterrows():
            delta_val = row.get("Delta_latest_max_mm", row.get("Delta_max_mm", np.nan))
            sev_correta = _classify(float(delta_val)) if pd.notna(delta_val) else "SEV0"
            gap_fmt = f"{float(delta_val):.1f}".replace(".",",") if pd.notna(delta_val) else "-"
            lbl = (f"{row.get('Turbina','')}-{row.get('Blade','')}" if (',' in turbina_txt or 'Selected' in turbina_txt) else str(row.get("Blade","")))
            conc_data.append([lbl, gap_fmt, sev_correta])
            worst_sev_idx = max(worst_sev_idx, s_map.get(sev_correta, 0))
        t_conc = Table(conc_data, colWidths=[6*cm, 3*cm, 4*cm])
        t_conc.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1F4E79")),('TEXTCOLOR',(0,0),(-1,0),colors.white),('GRID',(0,0),(-1,-1),0.5,colors.black)]))
        story.append(t_conc)
        if modelo == "ENEL": recs_list = ["6 Months","6 Months","3 Months","1 Month","15 Days","STOP WTG"]
        else: recs_list = ["12 Months","6 Months","3 Months","1 Month","15 Days","Stop Turbine"]
        rec_final = recs_list[worst_sev_idx]
        story.append(Spacer(1, 0.5*cm))
        if worst_sev_idx == 5:
            rec_text = "We recommend an immediate turbine shutdown (STOP WTG)." if modelo == "ENEL" else "We recommend an immediate turbine shutdown."
        else:
            rec_text = f"We recommend a new inspection within {rec_final}."
        story.append(Paragraph(rec_text, normal))
    story.append(PageBreak())

    story.append(Paragraph("4. Methodology", h1))
    story.append(Paragraph("The operation developed by Arthwind consists on inspecting the root gap of wind turbine blades using dial indicators, with the equipment installed externally around the blade root.", normal))
    if load_image_for_pdf("METOD_ROTOR"):
        story.append(Table([[get_proportional_image(load_image_for_pdf("METOD_ROTOR"),max_w=12*cm,max_h=5.5*cm)]], colWidths=[A4[0]-doc.leftMargin-doc.rightMargin], style=[('ALIGN',(0,0),(-1,-1),'CENTER')]))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Dial indicators are mounted at specific points around the blade root using suction bases and rods. The blade is rotated 360°while the dial indicators remain in position.", normal))
    if load_image_for_pdf("METOD_MAPA"):
        story.append(Table([[get_proportional_image(load_image_for_pdf("METOD_MAPA"),max_w=12*cm,max_h=5.5*cm)]], colWidths=[A4[0]-doc.leftMargin-doc.rightMargin], style=[('ALIGN',(0,0),(-1,-1),'CENTER')]))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("After the rotation, the maximum and minimum displacement values are analyzed to check the total variation at each point. This procedure is repeated for all blades on the turbine.", normal))
    if load_image_for_pdf("METOD_BASE"):
        story.append(Table([[get_proportional_image(load_image_for_pdf("METOD_BASE"),max_w=12*cm,max_h=5.5*cm)]], colWidths=[A4[0]-doc.leftMargin-doc.rightMargin], style=[('ALIGN',(0,0),(-1,-1),'CENTER')]))
    story.append(PageBreak())

    story.append(Paragraph("5. Scope", h1))
    story.append(Paragraph("This report presents the findings of the root gap inspection performed on the wind turbine blades. The scope encompasses the analysis of displacement data collected via dial indicators during a full rotor rotation. The primary objective is to evaluate the gap variation at multiple specific points around the circumference (PS, SS, LE, TE), classify the severity of any deviations according to the client standards, and provide actionable maintenance recommendations to ensure the structural integrity and safe operation of the equipment.", normal))
    story.append(Spacer(1, 0.5*cm))

    if modelo == "ENEL":
        story.append(Paragraph("6. Damages Categorization", h1))
        story.append(Paragraph("Note: Categorization and recommendations follow ENEL specific standards.", normal))
        cat_data = [["Severity","Description","Recommendation"],["0","No gaps detected","Inspect every 6 months"],["1","Gap \u2264 0,5mm","Inspect every 6 months"],["2","0,5mm < Gap \u2264 1mm","Inspect every 3 months"],["3","1mm < Gap \u2264 2mm","Inspect every 1 month"],["4","2mm < Gap \u2264 2,5mm","Inspect every 15 days"],["5","Gap > 2,5 mm","STOP WTG"]]
        col_w = [2.5*cm, 8.5*cm, 4.5*cm]
    else:
        story.append(Paragraph("6. Damages Categorization", h1))
        cat_data = [["Severity","Description","Recommendation"],["SEV 0","Gap < 1.0mm or no affected area","4 Months"],["SEV 1","One gap > 1.0mm","2 Months"],["SEV 2","20% of area OR Gap \u2265 1.5mm","1 Month"],["SEV 3","20% of area AND Gap \u2265 1.5mm","15 Days"],["SEV 4","30% - 40% of area","Gauge Measurement or Weekly"],["SEV 5","50% of area OR Gap > 3.0mm","Stop Turbine"]]
        col_w = [2.5*cm, 8.5*cm, 4.5*cm]

    c0,c1,c2,c3,c4,c5 = "#c6efce","#a9d18e","#ffd966","#f4b183","#ff8c00","#ff0000"
    sev_style  = ParagraphStyle("CatSev",  fontName="Helvetica-Bold", fontSize=9,   alignment=1, leading=11)
    desc_style = ParagraphStyle("CatDesc", fontName="Helvetica",      fontSize=9,   alignment=1, leading=11)
    rec_style  = ParagraphStyle("CatRec",  fontName="Helvetica",      fontSize=8.5, alignment=1, leading=11)
    head_style = ParagraphStyle("CatHead", fontName="Helvetica-Bold", fontSize=9,   textColor=colors.white, alignment=1, leading=11)
    def make_row(row, i):
        if i == 0: return [Paragraph(cell, head_style) for cell in row]
        return [Paragraph(row[0], sev_style), Paragraph(row[1], desc_style), Paragraph(row[2], rec_style)]
    cat_paragraphs = [make_row(row, i) for i, row in enumerate(cat_data)]
    t_cat = Table(cat_paragraphs, colWidths=col_w, repeatRows=1)
    t_cat.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1F4E79")),
        ('ALIGN',(0,0),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('GRID',(0,0),(-1,-1),0.5,colors.black),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
        ('BACKGROUND',(0,1),(-1,1),colors.HexColor(c0)),('BACKGROUND',(0,2),(-1,2),colors.HexColor(c1)),
        ('BACKGROUND',(0,3),(-1,3),colors.HexColor(c2)),('BACKGROUND',(0,4),(-1,4),colors.HexColor(c3)),
        ('BACKGROUND',(0,5),(-1,5),colors.HexColor(c4)),('BACKGROUND',(0,6),(-1,6),colors.HexColor(c5)),
    ]))
    story.append(t_cat); story.append(PageBreak())
    return story, draw_cover_full

def draw_header_footer(canvas, _doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10); canvas.setFillColor(colors.black)
    canvas.drawString(2*cm, A4[1]-1.5*cm, "Wind Blade Inspection Report - Preventive Maintenance")
    canvas.setFont("Helvetica", 9); canvas.setFillColor(colors.HexColor("#1F4E79"))
    canvas.drawRightString(A4[0]-2*cm, A4[1]-1.5*cm, "Arthwind Visibility and Prediction")
    canvas.setFont("Helvetica-Oblique", 8); canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[0]-2*cm, A4[1]-1.9*cm, "Manufacturing - Construction - Warrant - [Preventive]")
    canvas.setFont("Helvetica", 9); canvas.setFillColor(colors.black)
    canvas.drawCentredString(A4[0]/2, 1*cm, f"{canvas.getPageNumber()}")
    if load_image_for_pdf("LOGO"):
        try: canvas.drawImage(ImageReader(load_image_for_pdf("LOGO")), A4[0]-5*cm, 0.8*cm, width=3*cm, height=1.2*cm, mask='auto', preserveAspectRatio=True)
        except Exception: pass
    canvas.restoreState()

def generate_pdf(results, studs_ausentes_dict, progress_callback=None, modelo="Arthwind", windfarm=None, customer=None):
    _classify = results.get("classify_fn", get_classify_fn(modelo))
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2.5*cm, bottomMargin=2*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, textColor=colors.HexColor("#1F4E79"), spaceAfter=10)
    normal = ParagraphStyle("Norm", parent=styles["Normal"], fontSize=10, leading=12, alignment=TA_JUSTIFY)
    story, draw_cover_full = _create_cover_and_intro(doc, results, h1, normal, modelo=modelo, windfarm=windfarm, customer=customer)
    story.append(Paragraph("7. Inspection Evidence", h1))
    usable_w = A4[0] - doc.leftMargin - doc.rightMargin
    pdf_data = results["pdf_detailed_data"]; latest_sensors = results["latest_sensors"]
    total_blades = len(pdf_data)
    for idx, (blade, sensors_data) in enumerate(pdf_data):
        if progress_callback: progress_callback(idx+1, total_blades, f"Processando Pá {blade}...")
        story.append(Paragraph(f"BLADE {blade}", h1))
        img_io_polar, table_data_radar = create_radar_chart_and_table(blade, latest_sensors, studs_ausentes_dict, classify_fn=_classify)
        rl_polar = Image(img_io_polar, width=9*cm, height=9*cm)
        t_sev = Table(table_data_radar, colWidths=[3.5*cm, 2*cm, 2*cm])
        ts = [('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1F4E79")),('TEXTCOLOR',(0,0),(-1,0),colors.white),
              ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('ALIGN',(0,0),(-1,-1),'CENTER'),
              ('GRID',(0,0),(-1,-1),0.5,colors.grey),('VALIGN',(0,0),(-1,-1),'MIDDLE')]
        for i in range(1,len(table_data_radar)): ts.append(('BACKGROUND',(2,i),(2,i),colors.HexColor(severity_color(table_data_radar[i][2]))))
        t_sev.setStyle(TableStyle(ts))
        t_bottom = Table([[rl_polar, Spacer(1,1), t_sev]], colWidths=[9*cm, usable_w*0.05, usable_w*0.45])
        t_bottom.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'CENTER'),('ALIGN',(0,0),(-1,-1),'CENTER')]))
        story.append(t_bottom); story.append(Spacer(1, 0.5*cm))
        story.append(Paragraph("Registro de Studs Ausentes/Quebrados", h1))
        stud_data = [["ID Stud","Ângulo (°)","Zona Engenharia"]]
        studs_ausentes = studs_ausentes_dict.get(str(blade),[]) if isinstance(studs_ausentes_dict, dict) else []
        for stud in studs_ausentes: stud_data.append([str(stud), f"{calculate_angle(stud):.1f}°", get_stud_zone(stud)])
        if len(stud_data) > 1:
            t_studs = Table(stud_data, colWidths=[4*cm,4*cm,4*cm])
            t_studs.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1F4E79")),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ALIGN',(0,0),(-1,-1),'CENTER'),('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
            story.append(t_studs)
        else: story.append(Paragraph("Nenhum stud estrutural ausente reportado para esta pá.", normal))
        story.append(Spacer(1, 0.5*cm)); story.append(Paragraph("Detailed Sensor Curves:", normal)); story.append(Spacer(1, 0.3*cm))
        for item in sensors_data:
            img_io = io.BytesIO(); item["fig"].savefig(img_io, format='png', dpi=100, bbox_inches='tight'); img_io.seek(0)
            img_w = usable_w*0.70; tbl_w = usable_w-img_w
            rl_img = Image(img_io, width=img_w, height=6.5*cm)
            tdata = [["Campanha","Gap(mm)"]]
            for stat in item["stats"]: tdata.append([stat["Campanha"], f"{stat['Gap (mm)']:.1f}".replace(".",",") if pd.notna(stat['Gap (mm)']) else "-"])
            t_stats = Table(tdata, colWidths=[tbl_w*0.65, tbl_w*0.35])
            t_stats.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#e0e0e0")),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('ALIGN',(0,0),(-1,-1),'CENTER'),('GRID',(0,0),(-1,-1),0.5,colors.grey),('VALIGN',(0,0),(-1,-1),'MIDDLE')]))
            t_master = Table([[rl_img, t_stats]], colWidths=[img_w, tbl_w])
            t_master.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'CENTER')]))
            story.append(KeepTogether(t_master)); story.append(Spacer(1, 0.5*cm))
        story.append(PageBreak())
    doc.build(story, onFirstPage=draw_cover_full, onLaterPages=draw_header_footer)
    buffer.seek(0); return buffer.getvalue()

def generate_client_pdf(results, studs_ausentes_dict, progress_callback=None, modelo="Arthwind", windfarm=None, customer=None):
    _classify = results.get("classify_fn", get_classify_fn(modelo))
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2.5*cm, bottomMargin=2*cm, leftMargin=1.5*cm, rightMargin=1.5*cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, textColor=colors.HexColor("#1F4E79"), spaceAfter=10)
    normal = ParagraphStyle("Norm", parent=styles["Normal"], fontSize=10, leading=12, alignment=TA_JUSTIFY)
    story, draw_cover_full = _create_cover_and_intro(doc, results, h1, normal, modelo=modelo, windfarm=windfarm, customer=customer)
    story.append(Paragraph("7. Inspection Evidence", h1))
    usable_w = A4[0] - doc.leftMargin - doc.rightMargin
    verify_data, latest_sensors = results["verify_data"], results["latest_sensors"]
    colors_list = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
    flat_data = []
    for item in verify_data:
        sub = item["data"].copy(); sub["Sensor"] = f"{item['casca']}-{item['regiao']}"; sub["Blade"] = item["blade"]; sub["Inspecao"] = item["inspecao"]; flat_data.append(sub)
    df_viz_cli = pd.concat(flat_data, ignore_index=True) if flat_data else pd.DataFrame()
    total_blades = len(results["meta"]["blades"])
    for idx, blade in enumerate(results["meta"]["blades"]):
        if progress_callback: progress_callback(idx+1, total_blades, f"Montando vista da Pá {blade}...")
        story.append(Paragraph(f"BLADE {blade}", h1))
        if not df_viz_cli.empty:
            df_b_cli = df_viz_cli[df_viz_cli["Blade"].astype(str)==str(blade)]
            insp_max = df_b_cli["Inspecao"].max() if not df_b_cli.empty else "-"
            dt_insp = results["delta_summary"][(results["delta_summary"]["Blade"].astype(str)==str(blade))&(results["delta_summary"]["Inspecao"]==insp_max)]["Data"].max()
            data_str = pd.to_datetime(dt_insp).strftime('%d/%m/%Y') if pd.notna(dt_insp) else str(insp_max)
            img_io_line = create_dual_line_chart(blade, df_viz_cli, colors_list, data_str)
            story.append(Image(img_io_line, width=usable_w, height=5*cm)); story.append(Spacer(1, 0.5*cm))
        img_io_polar, table_data = create_radar_chart_and_table(blade, latest_sensors, studs_ausentes_dict, classify_fn=_classify)
        rl_polar = Image(img_io_polar, width=9*cm, height=9*cm)
        t_sev = Table(table_data, colWidths=[3.5*cm, 2*cm, 2*cm])
        ts = [('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1F4E79")),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('ALIGN',(0,0),(-1,-1),'CENTER'),('GRID',(0,0),(-1,-1),0.5,colors.grey),('VALIGN',(0,0),(-1,-1),'MIDDLE')]
        for i in range(1,len(table_data)): ts.append(('BACKGROUND',(2,i),(2,i),colors.HexColor(severity_color(table_data[i][2]))))
        t_sev.setStyle(TableStyle(ts))
        t_bottom = Table([[rl_polar,Spacer(1,1),t_sev]], colWidths=[9*cm,usable_w*0.05,usable_w*0.45])
        t_bottom.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'CENTER'),('ALIGN',(0,0),(-1,-1),'CENTER')]))
        story.append(t_bottom); story.append(Spacer(1,0.5*cm))
        story.append(Paragraph("Missing/Broken Studs Record", h1))
        stud_data = [["ID Stud","Ângulo (°)","Zona Engenharia"]]
        studs_ausentes = studs_ausentes_dict.get(str(blade),[]) if isinstance(studs_ausentes_dict, dict) else []
        for stud in studs_ausentes: stud_data.append([str(stud), f"{calculate_angle(stud):.1f}°", get_stud_zone(stud)])
        if len(stud_data) > 1:
            t_studs = Table(stud_data, colWidths=[4*cm,4*cm,4*cm])
            t_studs.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#1F4E79")),('TEXTCOLOR',(0,0),(-1,0),colors.white),('ALIGN',(0,0),(-1,-1),'CENTER'),('GRID',(0,0),(-1,-1),0.5,colors.grey)]))
            story.append(t_studs)
        else: story.append(Paragraph("No missing studs reported for this blade.", normal))
        story.append(PageBreak())
    doc.build(story, onFirstPage=draw_cover_full, onLaterPages=draw_header_footer)
    buffer.seek(0); return buffer.getvalue()

def generate_excel_report(delta_summary: pd.DataFrame):
    if delta_summary is None or delta_summary.empty: return None
    df_pivot = delta_summary.copy()
    df_pivot['Data'] = pd.to_datetime(df_pivot['Data'], errors='coerce')
    df_pivot['Data_Inspeção'] = df_pivot.groupby(['Turbina','Blade','Inspecao'])['Data'].transform('min').dt.strftime('%d/%m/%Y').fillna("-")
    df_pivot['Sensor_Key'] = df_pivot['Casca'].astype(str) + '-' + df_pivot['Regiao'].astype(str)
    pivot = df_pivot.pivot_table(index=['Turbina','Blade','Inspecao','Data_Inspeção'], columns='Sensor_Key', values='Delta_medio_ciclo_mm', aggfunc='first')
    pivot['Media'] = pivot.mean(axis=1); pivot['Maximo'] = pivot.max(axis=1)
    pivot['Severidade'] = pivot['Maximo'].apply(classify_severity_arthwind)
    pivot = pivot.reset_index()
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pivot.to_excel(writer, sheet_name='Base_Consolidada', index=False)
        wb, ws = writer.book, writer.sheets['Base_Consolidada']
        fmt_border = wb.add_format({'border':1,'align':'center','valign':'vcenter'})
        fmt_head = wb.add_format({'bold':True,'align':'center','border':1,'bg_color':'#D3D3D3'})
        hex_colors = {"SEV0":"#c6efce","SEV1":"#a9d18e","SEV2":"#ffd966","SEV3":"#f4b183","SEV4":"#ff8c00","SEV5":"#ff0000"}
        sev_fmts = {k: wb.add_format({'bg_color':v,'border':1,'align':'center'}) for k,v in hex_colors.items()}
        max_r, max_c = pivot.shape
        for c_idx, val in enumerate(pivot.columns.values): ws.write(0, c_idx, val, fmt_head)
        col_sev = pivot.columns.get_loc('Severidade')
        for r in range(max_r):
            sev_val = pivot.iloc[r, col_sev]
            for c_idx in range(max_c):
                val = pivot.iloc[r, c_idx]
                ws.write(r+1, c_idx, val if c_idx==col_sev else ("" if pd.isna(val) else val), sev_fmts.get(sev_val,fmt_border) if c_idx==col_sev else fmt_border)
    output.seek(0); return output.getvalue()

# =====================================================================
# PARTE 3: INTERFACE DE USUÁRIO (STREAMLIT) E DASHBOARDS
# =====================================================================

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Relatório")

modelo_dash = st.sidebar.selectbox(
    "Padrão de Severidade:",
    ["Arthwind", "ENEL"],
    help="Define os limiares usados em TODA a pipeline: tabelas, radares, cores e PDFs.",
    key="modelo_dash"
)

PARQUES_OPCOES = [
    "Acaraú","Acauã","Achiras II","Agreste Potiguar","Albatroz","Alegria","Alena","Alto Sertão",
    "Alto Sertão 3","Alto Sertão II","Amontada Windfarm","Anemus","Angicos","Anticline Wind Farm",
    "Aquiraz","Aracati","Areia Branca","Arizona","Arizona & Honorato","Aroeira","Asa Branca",
    "Assuruá","Assuruá 4","Assuruá 5","Atacama","Atlântica","Aventura","Aventura I",
    "Aventura Windfarm","Babilonia","Babilônia - Vestas","Babilônia Centro","Baixa do Feijão",
    "Barra dos Coqueiros","Bela Vista (Salinas)","Bom Jardim da Serra","Bons Ventos",
    "Bons Ventos da Serra I","Brisa Potiguar","Brotas de Macaubas","Bureau Veritas Wind Farm",
    "CGN","Cabeço Preto","Cabeção Preto III","Cabeção Preto V","Cabeção Preto VI","Cacimbas",
    "Caetité","Caetité 1","Caetité 2","Caetité 3","Caetité Norte","Caetés","Cajuina I e II",
    "Calango 1","Calango 2","Calango 3","Calango 4","Calango 5","Calangos 6",
    "Caldeirão Grande Windfarm","Campo Largo","Campo Largo I","Campo dos Ventos",
    "Campo dos Ventos Bloco Norte Bloco Sul","Canoa Quebrada","Canoas","Canudos","Casa Nova",
    "Casqueira","Cassino","Catanduba","Cerro Chato","Chafariz","Chapada","Chubut","Chui GE",
    "Chui Gamesa","Cidreira","Copel Windfarm","Coqueiros","Corredor do Senandes","Corti",
    "Coxilha Negra","Cristal","Cumaru","Curva dos Ventos","De La Bahia I","De Praia Formosa",
    "Delfina - Enel","Delta 1","Delta 2","Delta 3","Delta 5","Delta 6","Delta 7","Delta 8",
    "Delta Wind Farm","Demonstration 2","El Liano","El Mataco","El Mezquite",
    "Elera - Renascença Vestas","Embuaca (Mandacaru)","Enacel","Entorno 2","Eurus I","Eurus II",
    "Eurus III","Faisa","Feijão","Fenicias","Flat Ridge II","Folha Larga Norte","Folha Larga Sul",
    "Fonte dos Ventos I","Fonte dos Ventos II","Fortim","Foz do Rio Choró Windfarm","Gameleira",
    "Garayde","Gargaú","Genoveva I","Genoveva II","Geribatu","Goodnight I Wind Farm",
    "Grand Ridge I","Gravatá","Gravier","Hermenegildo","Honda","Ibirapuitã I",
    "Icarai (Mandacaru)","Icarai Windfarm","Icaraizinho","Itarema","Jandaíra","Jaú","Jerusalém",
    "Kaheawa I","Kairós","Kossuth Wind Farm","La Banderita","La Castellana","Lagoa 1","Lagoa 2",
    "Lagoa Nova","Lagoa do Barro do Piaui (Acciona)","Lagoa do Barro do Piaui (Goldwind)",
    "Lagoa do Mato Windfarm","Lagoa dos Ventos","MX4","Macacos","Macambira","Macambira I",
    "Macambira II","Malleco","Mangueira Mirim","Mar e Terra (Salinas)","Maral","Mel 2",
    "Mesa La Paz","Miassaba","Milenium","Modelo","Monte Verde","Morgado","Morrinhos",
    "Morro do Chapeu I","Morro do Chapeu II","Morro do Cruzeiro","Morro do Vento",
    "Morro dos Ventos I","Morro dos Ventos II","Morro dos Ventos III","Morro dos Ventos IV",
    "Morro dos Ventos IX","Morro dos Ventos VI","Mundo Novo","Necochea","Negrete",
    "Novo Horizonte","Oitis - GE","Oitis - LM","Ouro Branco","P13/ITU",
    "Panther Creek Wind Farm","Papagaios","Paracuru","Parajuru","Pedra Pintada","Pedra Rajada",
    "Pedra do Reino","Pedra do Reino III","Pegasus","Penenomé","Pepe","Pitombeira","Pontal",
    "Porto de Suape","Puelche Sur Wind Farm","Puerto Madryn","Quatro Ventos","Quixaba",
    "Rei dos Ventos","Rei dos Ventos / Miassaba","Renascença","Reynosa","Riachão Windfarm",
    "Rio do Vento I","Rio do Vento II","Rosa dos Ventos Windfarm","Salitrillos","San Gabriel",
    "San Jorge","San Pedro","San Pedro de Dalcahue 1","Santa Rosa Mundo Novo","Santana 1",
    "Santana 2","Santana do Livramento","Santo Agostinho","Santo Antonio de Padua (Mandacaru)",
    "Santo Inécio","Sempra I","Sempra II","Senandes","Sento Se","Seridó","Serra Azul",
    "Serra da Babilonia","Serra das Almas","Serra das Vacas","Serra de Assuruá","Serra de Santana",
    "Serra de Seridó","Serra do Mato","Serra do Mel I","Serra do Mel II","Serrote Windfarm",
    "Simões","Sinoma Blades - Auditoria","São Cristovao (Mandacaru)","São Fernando",
    "São Jorge (Mandacaru)","São Miguel do Gostoso","Taiba Windfarm","Talas I","Tanque Novo",
    "Terra Santa","Tolpan Sur","Trairi","Trairi - GE","Trairi - Gamesa","Tres Mesas 1 e 2",
    "Tres Mesas 3","Tres Mesas 4","Tubarão","Tucano","Umari","Umburanas","VDB3-SPDA",
    "Vale dos Ventos","Valle de Los Vientos","Ventika I","Ventika II","Vento de Serra do Mel I",
    "Vento de Serra do Mel II","Vento de Serra do Mel III","Ventos da Bahia","Ventos da Bahia 1",
    "Ventos da Bahia 2","Ventos da Bahia 3","Ventos de Santa Brígida","Ventos de Santa Eugenia",
    "Ventos de São Clemente","Ventos de São Vitor","Ventos de Tiangua","Ventos do Araripe",
    "Ventos do Araripe 3","Ventos do Piauí I","Ventos do Piauí II e III","Ventos dos Índios II",
    "Ventus (El Salvador)","Vicente Guerrero","Vila Acre","Vila Amazonas","Vila Mato Grosso",
    "Vila Pará","Vila Piaui","Villalonga","Vineyard","Vitória","Volta do Rio",
    "Xangrila Windfarm","Água Doce","Outro (digitar)",
]

MAPA_CLIENTE_PARQUES = {
    "ADS": ["Corredor do Senandes"],"AES": ["Penenomé"],
    "Acciona - Chile": ["San Gabriel","Tolpan Sur"],
    "Aliança Energia": ["Acauã","Gravier","Santo Inécio"],
    "Alupar": ["Agreste Potiguar","Pitombeira"],"Atlantic": ["Morrinhos"],
    "Auren": ["Alto Sertão II","Bela Vista (Salinas)","Caetés","Cajuina I e II","Cassino","Embuaca (Mandacaru)","Icarai (Mandacaru)","Mar e Terra (Salinas)","Miassaba","Rei dos Ventos","Rei dos Ventos / Miassaba","Santo Antonio de Padua (Mandacaru)","Sinoma Blades - Auditoria","São Cristovao (Mandacaru)","São Jorge (Mandacaru)","Tucano","Ventos do Araripe","Ventos do Araripe 3","Ventos do Piauí I","Ventos do Piauí II e III"],
    "BVS2": ["Cacimbas"],"Bureau Veritas": ["Bureau Veritas Wind Farm"],
    "CGN Brasil Energia": ["Lagoa do Barro do Piaui (Goldwind)","Morrinhos"],
    "COPEL": ["Brisa Potiguar","Vento de Serra do Mel II"],
    "CPFL": ["Albatroz","Aracati","Atlântica","Bons Ventos","Canoa Quebrada","De Praia Formosa","Enacel","Eurus I","Eurus III","Foz do Rio Choró Windfarm","Gameleira","Icaraizinho","Lagoa do Mato Windfarm","Macacos","Morro do Vento","Morro dos Ventos I","Morro dos Ventos II","Morro dos Ventos III","Morro dos Ventos IV","Morro dos Ventos IX","Morro dos Ventos VI","Paracuru","Rosa dos Ventos Windfarm"],
    "Casa dos Ventos": ["Maral","Terra Santa"],"Cemig": ["Parajuru","Volta do Rio"],
    "Cubico": ["Simões","Trairi","Ventos de Santa Brígida"],"DNV": ["Tucano"],
    "EDF": ["Folha Larga Norte","Ventos da Bahia"],
    "EDP Renováveis Brasil": ["Aventura I","Baixa do Feijão","Catanduba","Cidreira","Jaú","Monte Verde"],
    "Echoenergia": ["Lagoa Nova","Pedra do Reino III","Serra do Mel I","Serra do Mel II","Ventos de São Clemente","Ventos de Tiangua"],
    "Elawan Wind": ["Cabeção Preto III","Cabeção Preto V","Cabeção Preto VI","Macambira I","Macambira II"],
    "Elera": ["Alto Sertão","Faisa","Pontal","Renascença"],
    "Eletrosul": ["Cerro Chato","Coxilha Negra","Entorno 2"],
    "Enel": ["Aroeira","Cristal","Curva dos Ventos","Delfina - Enel","Fonte dos Ventos I","Modelo","Morro do Chapeu I","Pedra Pintada","Serra Azul"],
    "Energimp": ["Bom Jardim da Serra","Coqueiros","Morgado","Papagaios","Quixaba","Água Doce"],
    "Engie": ["Campo Largo","Umburanas"],
    "Essentia Energia": ["Asa Branca","Chapada","Ventos de São Vitor"],
    "European": ["Ouro Branco","Quatro Ventos"],"Eólica Ibirapuitã": ["Ibirapuitã I"],
    "Eólicas Babilônia": ["Babilonia"],"FCG": ["Aquiraz"],"GE México": ["El Mezquite"],
    "GE Renewable Energy": ["Assuruá 5","Campo Largo","Delta 3","Fonte dos Ventos II","Hermenegildo","Oitis - GE","Serra da Babilonia","Serra de Seridó","Trairi - GE","Umburanas","Ventos da Bahia 3","Ventos de Tiangua"],
    "Goldwind": ["Acaraú","Aquiraz","Bom Jardim da Serra","Casa Nova","Lagoa do Barro do Piaui (Goldwind)","Tanque Novo","Vitória"],
    "Ibitu": ["Amontada Windfarm","Caldeirão Grande Windfarm","Icarai Windfarm","Riachão Windfarm","Taiba Windfarm"],
    "Invenergy": ["Demonstration 2","Grand Ridge I"],
    "LM Wind Power": ["Oitis - LM","Santo Agostinho","Tucano","Ventos de São Vitor"],
    "NAWP": ["Areia Branca","Atlântica","Cajuina I e II","Casqueira","Feijão","Fortim","Itarema","Jandaíra","Lagoa do Barro do Piaui (Acciona)","Lagoa dos Ventos","Mangueira Mirim","Morro do Cruzeiro","São Fernando","São Miguel do Gostoso","Ventos da Bahia","Ventos de Santa Eugenia","Vila Amazonas","Vila Mato Grosso","Vila Pará","Vila Piaui"],
    "NAWP - Chile": ["Alena","Atacama","Puelche Sur Wind Farm"],
    "NC Energias Renováveis": ["Senandes"],
    "Neoenergia": ["Arizona","Caetité 1","Caetité 2","Caetité 3","Calango 1","Calango 2","Calango 3","Calango 4","Calango 5","Calangos 6","Canoas","Lagoa 1","Lagoa 2","Mel 2","Santana 1","Santana 2"],
    "New Energy": ["Alegria"],"Nextera": ["Pegasus"],
    "Pontal Energy": ["Caetité","Caetité Norte","Itarema"],"Renova": ["Alto Sertão 3","P13/ITU"],
    "Rio Energy": ["Caetité","Porto de Suape","Serra da Babilonia"],
    "SGRE": ["Arizona & Honorato","Campo dos Ventos","Campo dos Ventos Bloco Norte Bloco Sul","Canudos","Cassino","Chafariz","Chui Gamesa","Delta 1","Gameleira","Geribatu","Maral","San Pedro de Dalcahue 1","Santana do Livramento","Santo Agostinho","Serra de Santana","Talas I","Terra Santa","Trairi - Gamesa","Tucano","Vento de Serra do Mel I","Vento de Serra do Mel II","Vento de Serra do Mel III","Ventos de São Vitor","Ventos do Piauí I"],
    "SPIC": ["Milenium","Vale dos Ventos"],"Sempra Infraestructura": ["Ventika I","Ventika II"],
    "Serena": ["Assuruá","Assuruá 4","Assuruá 5","Chui GE","Chui Gamesa","Delta 1","Delta 2","Delta 3","Delta 5","Delta 6","Delta 7","Delta 8","Gargaú","Geribatu","Goodnight I Wind Farm","Ventos da Bahia 1","Ventos da Bahia 2","Ventos da Bahia 3"],
    "Serra das Vacas": ["Serra das Vacas"],
    "Skyspecs": ["Anticline Wind Farm","Delta Wind Farm","Flat Ridge II","Kaheawa I","Kossuth Wind Farm","Panther Creek Wind Farm","Vineyard"],
    "Statkraft Energias Renováveis": ["Barra dos Coqueiros","Brotas de Macaubas","São Fernando","Ventos de Santa Eugenia"],
    "Storz": ["VDB3-SPDA"],"TPI": ["MX4"],
    "Vestas Argentina": ["Achiras II","Chubut","Corti","De La Bahia I","El Liano","El Mataco","Garayde","Genoveva I","Genoveva II","La Banderita","La Castellana","Necochea","Pepe","Puerto Madryn","San Jorge","Villalonga"],
    "Vestas Brasil": ["Amontada Windfarm","Angicos","Aroeira","Assuruá 4","Aventura","Aventura Windfarm","Babilônia - Vestas","Babilônia Centro","CGN","Cabeço Preto","Caldeirão Grande Windfarm","Campo Largo I","Catanduba","Copel Windfarm","Cumaru","Elera - Renascença Vestas","Eurus II","Folha Larga Norte","Folha Larga Sul","Gravatá","Honda","Icarai Windfarm","Jerusalém","Kairós","Macambira","Monte Verde","Morro do Chapeu II","Mundo Novo","Novo Horizonte","Ouro Branco","Pedra Pintada","Pedra Rajada","Pedra do Reino","Quatro Ventos","Riachão Windfarm","Rio do Vento I","Rio do Vento II","Santa Rosa Mundo Novo","Sento Se","Seridó","Serra das Almas","Serra das Vacas","Serra de Assuruá","Serra do Mato","Serra do Mel I","Serra do Mel II","Serrote Windfarm","Taiba Windfarm","Umari","Ventos do Piauí II e III","Xangrila Windfarm"],
    "Vestas Chile": ["Malleco","Negrete","Valle de Los Vientos"],
    "Vestas El Salvador": ["Ventus (El Salvador)"],
    "Vestas México": ["Fenicias","Mesa La Paz","Reynosa","Salitrillos","San Pedro","Sempra I","Sempra II","Tres Mesas 1 e 2","Tres Mesas 3","Tres Mesas 4","Vicente Guerrero"],
    "Voltalia Energia Renovável": ["Areia Branca","Vento de Serra do Mel I","Vento de Serra do Mel III","Vila Acre","Vila Amazonas","Vila Pará"],
    "WEG": ["Acauã","Agreste Potiguar","Anemus","Bons Ventos da Serra I","Gravier","Ibirapuitã I","Santo Inécio","Tubarão"],
    "Windcraft": ["Cumaru","Gravier","Morro do Chapeu II"],
    "Wobben Wind Power": ["Ventos dos Índios II"],
}

cliente_sel = st.sidebar.selectbox("Cliente / Customer:", list(MAPA_CLIENTE_PARQUES.keys()) + ["Outro (digitar)"], key="cliente_sel")
if cliente_sel == "Outro (digitar)":
    cliente_final = st.sidebar.text_input("Nome do cliente:", value="", key="cliente_custom").strip() or "CLIENTE"
    parques_disponiveis = PARQUES_OPCOES
else:
    cliente_final = cliente_sel
    parques_disponiveis = MAPA_CLIENTE_PARQUES.get(cliente_sel, []) + ["Outro (digitar)"]

parque_sel = st.sidebar.selectbox("Parque / Windfarm:", parques_disponiveis, key="parque_sel")
if parque_sel == "Outro (digitar)":
    parque_final = st.sidebar.text_input("Nome do parque:", value="", key="parque_custom").strip() or "COMPLEXO EOLICO"
else:
    parque_final = parque_sel

if st.button("Calcular Análise"):
    with st.spinner("Calculando Visualização..."):
        results = run_analysis(df_raw, full_process=False, modelo=modelo_dash)
        st.session_state["results"] = results
        st.session_state["modelo_usado"] = modelo_dash
        st.session_state["parque_usado"] = parque_final
        st.session_state["cliente_usado"] = cliente_final
        if 'df_calibre' in globals() and df_calibre is not None and not df_calibre.empty:
            st.session_state["df_calibre_proc"] = process_calibre_data(df_calibre, float(perim_mm))

if "results" in st.session_state and st.session_state["results"] is not None:
    results        = st.session_state["results"]
    modelo_atual   = st.session_state.get("modelo_usado",  "Arthwind")
    parque_atual   = st.session_state.get("parque_usado",  "COMPLEXO EÓLICO SERRA AZUL")
    cliente_atual  = st.session_state.get("cliente_usado", "ENEL")
    _classify_dash = results.get("classify_fn", get_classify_fn(modelo_atual))

    st.markdown("<br>", unsafe_allow_html=True)
    aba_selecionada = st.radio(
        "Navegação de Abas",
        ["📊 Resumo Executivo", "⚙️ Visão Engenharia", "🎯 Visão Cliente", "📥 Downloads"],
        horizontal=True, label_visibility="collapsed"
    )

    verify_data    = results.get("verify_data", [])
    latest_sensors = results["latest_sensors"]

    # =========================================================================
    # ABA 1: RESUMO EXECUTIVO
    # =========================================================================
    if aba_selecionada == "📊 Resumo Executivo":
        st.subheader(f"📊 Resumo Global — Padrão: {modelo_atual}")
        qtd_turbs  = len(turb_sel)  if turb_sel  else df_raw['Turbina'].nunique()
        qtd_blades = len(blades_sel) if blades_sel else df_raw['SN_da_Pa'].nunique()
        qtd_insps  = len(insps_sel)  if insps_sel  else df_raw['Inspecao'].nunique()
        c1, c2, c3 = st.columns(3)
        c1.metric("Turbinas Inspecionadas", qtd_turbs); c2.metric("Pás Inspecionadas", qtd_blades); c3.metric("Inspeções Realizadas", qtd_insps)
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pás mais críticas (Top 15)**")
            cb = results.get("critical_blades", pd.DataFrame())
            if not cb.empty:
                cb_show = cb.copy(); cb_show["Delta_latest_max_mm"] = cb_show["Delta_latest_max_mm"].round(1)
                if "Last_Date" in cb_show.columns: cb_show["Last_Date"] = pd.to_datetime(cb_show["Last_Date"],errors="coerce").dt.strftime("%d-%m-%Y")
                if "Next_Inspection" in cb_show.columns: cb_show["Next_Inspection"] = pd.to_datetime(cb_show["Next_Inspection"],errors="coerce").dt.strftime("%d-%m-%Y")
                st.dataframe(cb_show, use_container_width=True)
        with c2:
            st.markdown("**Turbinas mais críticas (Top 15)**")
            ct = results.get("critical_turbines", pd.DataFrame())
            if not ct.empty:
                ct_show = ct.copy(); ct_show["Delta_latest_max_mm"] = ct_show["Delta_latest_max_mm"].round(1)
                if "Last_Date" in ct_show.columns: ct_show["Last_Date"] = pd.to_datetime(ct_show["Last_Date"],errors="coerce").dt.strftime("%d-%m-%Y")
                st.dataframe(ct_show, use_container_width=True)
        st.markdown("---")
        st.markdown("### 🚨 Gestão de Próximas Inspeções")
        st.info("Painel de priorização: Acompanhe as turbinas e pás com inspeções vencidas ou próximas do vencimento.")
        df_gestao = results.get("severity_by_blade_latest", pd.DataFrame()).copy()
        if not df_gestao.empty and "Next_Inspection" in df_gestao.columns:
            hoje = pd.Timestamp(dt.datetime.now().date())
            df_gestao["Next_Inspection_dt"] = pd.to_datetime(df_gestao["Next_Inspection"], errors="coerce")
            def calc_status(row):
                if pd.isna(row["Next_Inspection_dt"]): return "N/A"
                dias = (row["Next_Inspection_dt"] - hoje).days
                if dias < 0: return "🚨 Vencida"
                if dias <= 30: return "⚠️ Próxima (30d)"
                return "✅ No Prazo"
            def calc_dias(row):
                if pd.isna(row["Next_Inspection_dt"]): return None
                return (row["Next_Inspection_dt"] - hoje).days
            df_gestao["Status"] = df_gestao.apply(calc_status, axis=1)
            df_gestao["Dias Restantes"] = df_gestao.apply(calc_dias, axis=1)
            df_gestao["Última Inspeção"] = pd.to_datetime(df_gestao["Last_Date"],errors="coerce").dt.strftime("%d/%m/%Y")
            df_gestao["Próxima Inspeção"] = df_gestao["Next_Inspection_dt"].dt.strftime("%d/%m/%Y")
            cols_show = ["Turbina","Blade","Severity","Última Inspeção","Recommendation","Próxima Inspeção","Dias Restantes","Status"]
            df_gestao_show = df_gestao[cols_show].sort_values(by="Dias Restantes", ascending=True, na_position="last")
            def color_status(val):
                if val == "🚨 Vencida": return 'background-color: #ff4c4c; color: white; font-weight: bold;'
                if val == "⚠️ Próxima (30d)": return 'background-color: #ffc107; color: black; font-weight: bold;'
                if val == "✅ No Prazo": return 'background-color: #28a745; color: white;'
                return ''
            st.dataframe(df_gestao_show.style.map(color_status, subset=["Status"]), use_container_width=True, hide_index=True)
        else:
            st.warning("Dados insuficientes para gerar a gestão de próximas inspeções.")
        st.markdown("---")
        st.markdown("### 📈 Distribuição por Severidade")
        sev_order = ["SEV0","SEV1","SEV2","SEV3","SEV4","SEV5"]
        g1, g2 = st.columns(2)
        with g1:
            bl = results.get("severity_by_blade_latest", pd.DataFrame())
            if not bl.empty and "Severity" in bl.columns:
                counts = bl["Severity"].value_counts().reindex(sev_order,fill_value=0).reset_index(); counts.columns=["Severity","Count"]
                st.plotly_chart(px.bar(counts,x="Severity",y="Count",title="Pás por Severidade"), use_container_width=True)
        with g2:
            tl = results.get("severity_by_turbine_latest", pd.DataFrame())
            if not tl.empty and "Severity" in tl.columns:
                counts = tl["Severity"].value_counts().reindex(sev_order,fill_value=0).reset_index(); counts.columns=["Severity","Count"]
                st.plotly_chart(px.bar(counts,x="Severity",y="Count",title="Turbinas por Severidade"), use_container_width=True)
        st.markdown("---")
        st.markdown("### 📦 Correlação: Gap Máximo vs. Área Afetada")
        limiar_area = 1.0 if modelo_atual == "Arthwind" else 0.5
        st.info(f"Distribuição do maior GAP registrado em cada pá vs. a porcentagem da raiz que ultrapassou o limite de normalidade (>{limiar_area}mm — padrão {modelo_atual}).")
        df_box = results.get("latest_sensors", pd.DataFrame()).copy()
        if not df_box.empty:
            def calc_blade_stats(g): return pd.Series({"Max_Gap": g["Delta_medio_ciclo_mm"].max(), "Afetados": (g["Delta_medio_ciclo_mm"] > limiar_area).sum()})
            blade_stats = df_box.groupby(["Turbina","Blade"]).apply(calc_blade_stats).reset_index()
            def categorize_area(a):
                p = a * 10
                if p == 0: return "0%"
                elif p == 10: return "10%"
                elif p == 20: return "20%"
                elif p == 30: return "30%"
                elif p == 40: return "40%"
                else: return "≥50%"
            blade_stats["Area_Afetada"] = blade_stats["Afetados"].apply(categorize_area)
            cat_order = ["0%","10%","20%","30%","40%","≥50%"]
            contagem = blade_stats["Area_Afetada"].value_counts().to_dict()
            blade_stats["Categoria_Eixo"] = blade_stats["Area_Afetada"].apply(lambda x: f"{x} (n={contagem.get(x,0)})")
            cat_order_eixo = [f"{c} (n={contagem.get(c,0)})" for c in cat_order if c in contagem]
            fig_box = px.box(blade_stats,x="Categoria_Eixo",y="Max_Gap",points="all",hover_data=["Turbina","Blade"],
                category_orders={"Categoria_Eixo":cat_order_eixo},labels={"Categoria_Eixo":"Área Afetada","Max_Gap":"Gap Máximo (mm)"},color="Area_Afetada")
            fig_box.add_hline(y=limiar_area,line_dash="dash",line_color="#ff4c4c",annotation_text=f"{limiar_area} mm (gatilho)")
            fig_box.update_layout(showlegend=False,height=450,margin=dict(t=30,b=40,l=40,r=40))
            st.plotly_chart(fig_box, use_container_width=True)
        else: st.warning("Dados insuficientes para gerar o Boxplot de Frota.")

    # =========================================================================
    # ABA 2: VISÃO ENGENHARIA
    # =========================================================================
    elif aba_selecionada == "⚙️ Visão Engenharia":
        st.subheader(f"Auditoria Técnica e Processamento de Sinal — Padrão: {modelo_atual}")
        has_calibre = "df_calibre_proc" in st.session_state and not st.session_state["df_calibre_proc"].empty
        if has_calibre:
            sub_tabs = st.tabs(["📈 Auditoria BlueDial","🧭 Dashboard Calibre","🎯 Cruzamento Relógio vs Calibre"])
        else:
            sub_tabs = st.tabs(["📈 Auditoria BlueDial"])
            st.info("💡 Faça o upload de um arquivo de Calibre na barra lateral para habilitar as abas espaciais e cruzadas.")

        with sub_tabs[0]:
            st.markdown("### 📈 Progressão de Deslocamento Temporal (Tendência de Gap)")
            df_eng = results.get("delta_summary", pd.DataFrame()).copy()
            if not df_eng.empty:
                df_eng["Sensor"] = df_eng["Casca"].astype(str) + "-" + df_eng["Regiao"].astype(str)
                df_eng = df_eng.sort_values(by=["Data","Inspecao"])
                c_eng1, c_eng2 = st.columns(2)
                with c_eng1: eng_turb  = st.selectbox("Turbina:", sorted(df_eng["Turbina"].unique()), key="eng_t")
                with c_eng2: eng_blade = st.selectbox("Pá:", sorted(df_eng[df_eng["Turbina"]==eng_turb]["Blade"].unique()), key="eng_b")
                df_eng_plot = df_eng[(df_eng["Turbina"]==eng_turb) & (df_eng["Blade"]==eng_blade)]
                df_eng_ps = df_eng_plot[df_eng_plot["Casca"]=="PS"]
                df_eng_ss = df_eng_plot[df_eng_plot["Casca"]=="SS"]
                limite_sev5 = 3.0 if modelo_atual=="Arthwind" else 2.5
                g1, g2 = st.columns(2)
                with g1:
                    if not df_eng_ps.empty:
                        fig_ps = px.line(df_eng_ps,x="Data",y="Delta_medio_ciclo_mm",color="Sensor",hover_data=["Inspecao"],markers=True,title=f"Evolução de Gap - PS (Pá {eng_blade})")
                        fig_ps.add_hline(y=limite_sev5,line_dash="dash",line_color="red",annotation_text=f"Limite crítico ({limite_sev5}mm)")
                        fig_ps.update_xaxes(tickformat="%d/%m/%Y"); st.plotly_chart(fig_ps, use_container_width=True)
                with g2:
                    if not df_eng_ss.empty:
                        fig_ss = px.line(df_eng_ss,x="Data",y="Delta_medio_ciclo_mm",color="Sensor",hover_data=["Inspecao"],markers=True,title=f"Evolução de Gap - SS (Pá {eng_blade})")
                        fig_ss.add_hline(y=limite_sev5,line_dash="dash",line_color="red",annotation_text=f"Limite crítico ({limite_sev5}mm)")
                        fig_ss.update_xaxes(tickformat="%d/%m/%Y"); st.plotly_chart(fig_ss, use_container_width=True)
            st.divider()
            st.markdown("### 🧭 Radar de Deslocamento Angular (BlueDial)")
            limiar_radar = 1.0 if modelo_atual=="Arthwind" else 0.5
            st.info(f"Distribuição do GAP dinâmico máximo por sensor. Limite de tolerância em {limiar_radar}mm (linha verde tracejada).")
            if not df_eng_plot.empty:
                fig_bd_radar = go.Figure()
                theta_circle = np.linspace(0,360,100)
                fig_bd_radar.add_trace(go.Scatterpolar(r=[limiar_radar]*100,theta=theta_circle,mode='lines',line=dict(color='green',width=2,dash='dash'),name=f'Normalidade ({limiar_radar}mm)',hoverinfo='none'))
                for camp in sorted(df_eng_plot["Inspecao"].unique()):
                    d_c = df_eng_plot[df_eng_plot["Inspecao"]==camp]; pts,afetados = [],0
                    for sens_key,furo in MAPA_FUROS.items():
                        casca,reg = sens_key.split("-")
                        row = d_c[(d_c["Casca"]==casca)&(d_c["Regiao"]==reg)]
                        gap = row["Delta_medio_ciclo_mm"].max() if not row.empty else np.nan
                        if not pd.isna(gap):
                            pts.append({"ang":calculate_angle(furo),"gap":gap,"sensor":sens_key})
                            if gap>limiar_radar: afetados+=1
                    if pts:
                        df_pts = pd.DataFrame(pts).sort_values("ang")
                        df_pts = pd.concat([df_pts,df_pts.iloc[[0]]],ignore_index=True)
                        fig_bd_radar.add_trace(go.Scatterpolar(r=df_pts["gap"],theta=df_pts["ang"],mode='lines+markers',name=f"{camp} (Área: {afetados*10}%)"))
                fig_bd_radar.update_layout(polar=dict(angularaxis=dict(direction="counterclockwise",rotation=0,tickmode='array',tickvals=[0,90,180,270],ticktext=["LE","PS","TE","SS"]),radialaxis=dict(range=[0,5.0])),height=550)
                st.plotly_chart(fig_bd_radar, use_container_width=True)
            st.divider()
            st.markdown("### 🔍 Análise de Sinal por Relógio")
            plots_sinal = []
            for item in verify_data:
                if str(item["turbina"])==str(eng_turb) and str(item["blade"])==str(eng_blade):
                    sub = item["data"].copy(); sub["Sensor"]=f"{item['casca']}-{item['regiao']} ({item['relogio']})"; sub["Campanha"]=item["inspecao"]; plots_sinal.append(sub)
            if plots_sinal:
                df_sinal = pd.concat(plots_sinal,ignore_index=True); sensores_disponiveis = sorted(df_sinal["Sensor"].unique())
                sens_sel = st.selectbox("Selecione o Relógio:",["Visualizar Todos (Empilhado)"]+sensores_disponiveis)
                if sens_sel=="Visualizar Todos (Empilhado)":
                    for s in sensores_disponiveis:
                        df_p=df_sinal[df_sinal["Sensor"]==s]; fig=px.line(df_p,x="Ponto",y="Valor_mm",color="Campanha",title=f"Sinal: {s}",height=300); fig.update_layout(margin=dict(t=40,b=20)); st.plotly_chart(fig,use_container_width=True)
                else:
                    df_p=df_sinal[df_sinal["Sensor"]==sens_sel]; fig=px.line(df_p,x="Ponto",y="Valor_mm",color="Campanha",title=f"Sinal em Destaque: {sens_sel}",height=500); st.plotly_chart(fig,use_container_width=True)
            else: st.warning("Certifique-se de que há campanhas selecionadas na barra lateral.")

        if has_calibre:
            with sub_tabs[1]:
                st.markdown("### 🧭 Dashboard Completo - Calibre de Folga")
                df_cal_raw = st.session_state["df_calibre_proc"]
                st.markdown("#### 🔎 Filtragem de Calibre")
                c_f1,c_f2,c_f3 = st.columns(3)
                cal_turbs = sorted(df_cal_raw["Turbina"].unique())
                f_turb  = c_f1.multiselect("Turbina:",cal_turbs,default=cal_turbs,key="f_cal_t")
                cal_blades = sorted(df_cal_raw[df_cal_raw["Turbina"].isin(f_turb)]["Blade"].unique())
                f_blade = c_f2.multiselect("Pá:",cal_blades,default=cal_blades,key="f_cal_b")
                cal_camps = sorted(df_cal_raw[df_cal_raw["Turbina"].isin(f_turb)]["Campaign"].unique())
                f_camp  = c_f3.multiselect("Campanha:",cal_camps,default=cal_camps,key="f_cal_c")
                df_cal = df_cal_raw[df_cal_raw["Turbina"].isin(f_turb)&df_cal_raw["Blade"].isin(f_blade)&df_cal_raw["Campaign"].isin(f_camp)]
                if df_cal.empty: st.warning("A filtragem não retornou dados para o Calibre.")
                else:
                    def build_ticks_perimeter(pm,step_mm=500.0):
                        marks=np.arange(0,pm,step_mm).astype(float); theta=np.mod(180.0-(marks/pm)*360.0,360.0); return theta.tolist(),[f"{int(m)}" for m in marks]
                    def _clip(x): return np.clip(np.abs(np.asarray(x,dtype=float)),0.0,5.0)
                    def _has_nz(y_raw,abs_mode):
                        y=np.asarray(y_raw,dtype=float); f=np.isfinite(y)
                        if f.sum()<2: return False
                        return np.nanmax(np.abs(y[f]))>0 if abs_mode else np.nanmax(y[f])>0
                    def build_radar_mm(d_bl,pm,y_col,camps,title,overlay_red=False):
                        fig=go.Figure(); tv,tt=build_ticks_perimeter(pm); abs_mode=y_col in("M3H","M9H"); added=False
                        for camp in camps:
                            dc=d_bl[d_bl["Campaign"]==camp].dropna(subset=["theta_deg","dist_bf_mm"]).sort_values("dist_bf_mm")
                            if dc.empty: continue
                            theta=dc["theta_deg"].values.astype(float); y_raw=pd.to_numeric(dc[y_col],errors="coerce").values.astype(float)
                            if not _has_nz(y_raw,abs_mode): continue
                            y=_clip(y_raw); fig.add_trace(go.Scatterpolar(r=y,theta=theta,mode="lines",line=dict(width=2),name=camp)); added=True
                            if overlay_red and y_col=="Gap (mm)":
                                g_raw=pd.to_numeric(dc["Gap (mm)"],errors="coerce").values.astype(float)
                                if _has_nz(g_raw,False):
                                    g=_clip(g_raw); g_red=np.where(g>0,g,np.nan)
                                    if np.isfinite(g_red).sum()>=2: fig.add_trace(go.Scatterpolar(r=g_red,theta=theta,mode="lines",line=dict(width=6,color="red"),showlegend=False,name="Affected (GAP>0)"))
                        fig.update_layout(title=title,margin=dict(l=40,r=40,t=80,b=30),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0),
                            polar=dict(angularaxis=dict(tickmode="array",tickvals=tv,ticktext=tt,rotation=0,direction="counterclockwise"),radialaxis=dict(range=[0,5.0],ticks="outside",ticklen=3,showline=True)))
                        fig.add_annotation(text="PS",x=0.5,y=0.92,xref="paper",yref="paper",showarrow=False,font=dict(size=16,color="rgba(31,78,121,0.65)"))
                        fig.add_annotation(text="SS",x=0.5,y=0.10,xref="paper",yref="paper",showarrow=False,font=dict(size=16,color="rgba(31,78,121,0.65)"))
                        if not added: fig.update_layout(showlegend=False)
                        return fig
                    def build_freq_radar(df_sel,pm,bm):
                        d=df_sel.copy(); d["bin_mm"]=(np.round(pd.to_numeric(d["dist_bf_mm"],errors="coerce")/bm)*bm).clip(0.0,pm)
                        pbp=d.groupby(["Turbina","Blade","bin_mm"],as_index=False).agg(Max_GAP=("Gap (mm)","max"))
                        pbp["Affected"]=(pd.to_numeric(pbp["Max_GAP"],errors="coerce").fillna(0)>0).astype(int)
                        freq=pbp.groupby(["bin_mm"],as_index=False)["Affected"].sum().rename(columns={"Affected":"Blades_Affected"}).sort_values("bin_mm")
                        tv,tt=build_ticks_perimeter(pm); theta=np.mod(180.0-(freq["bin_mm"].values.astype(float)/pm)*360.0,360.0)
                        r=pd.to_numeric(freq["Blades_Affected"],errors="coerce").fillna(0).values.astype(float)
                        fig=go.Figure(); fig.add_trace(go.Barpolar(r=r,theta=theta,width=360.0*(bm/pm),marker_color=np.where(r>0,"red","rgba(200,200,200,0.35)"),marker_line_color="rgba(255,255,255,0.2)",marker_line_width=0.5))
                        fig.update_layout(title="Radar Consolidado — Frequência de Ponto Afetado",margin=dict(l=40,r=40,t=80,b=30),showlegend=False,
                            polar=dict(angularaxis=dict(tickmode="array",tickvals=tv,ticktext=tt,rotation=0,direction="counterclockwise"),radialaxis=dict(title="Qtd. de pás",ticks="outside",ticklen=3,showline=True)))
                        return fig, freq
                    st.markdown("---"); fig_cons,_=build_freq_radar(df_cal,float(perim_mm),float(bin_mm)); st.plotly_chart(fig_cons,use_container_width=True)
                    st.markdown("---"); st.markdown("#### 📌 Pontos mais críticos (Top 25) — 1 ponto por pá")
                    def pick_worst(df_in,score):
                        tmp=df_in.copy(); tmp["_s"]=pd.to_numeric(score,errors="coerce"); tmp=tmp.dropna(subset=["_s"])
                        if tmp.empty: return tmp.drop(columns=["_s"],errors="ignore")
                        idx=tmp.groupby(["Turbina","Blade"])["_s"].idxmax(); return tmp.loc[idx].drop(columns=["_s"],errors="ignore")
                    w3h =pick_worst(df_cal,df_cal["M3H"].abs()).sort_values("M3H",key=lambda s:s.abs(),ascending=False).head(25)
                    w9h =pick_worst(df_cal,df_cal["M9H"].abs()).sort_values("M9H",key=lambda s:s.abs(),ascending=False).head(25)
                    wgap=pick_worst(df_cal,df_cal["Gap (mm)"]).sort_values("Gap (mm)",ascending=False).head(25)
                    cc1,cc2,cc3=st.columns(3)
                    with cc1: st.markdown("**Top 3h**");  st.dataframe(w3h[["Turbina","Blade","Campaign","M3H","M9H","Gap (mm)"]].style.format(precision=2),use_container_width=True)
                    with cc2: st.markdown("**Top 9h**");  st.dataframe(w9h[["Turbina","Blade","Campaign","M3H","M9H","Gap (mm)"]].style.format(precision=2),use_container_width=True)
                    with cc3: st.markdown("**Top GAP**"); st.dataframe(wgap[["Turbina","Blade","Campaign","M3H","M9H","Gap (mm)"]].style.format(precision=2),use_container_width=True)
                    st.markdown("---")
                    for tb,bl in df_cal[["Turbina","Blade"]].drop_duplicates().values.tolist()[:20]:
                        st.markdown(f"**Turbina `{tb}` — Pá `{bl}`**")
                        d_bl=df_cal[(df_cal["Turbina"]==tb)&(df_cal["Blade"]==bl)].copy(); camps=sorted(d_bl["Campaign"].unique().tolist())
                        cA,cB,cC=st.columns(3)
                        with cA: st.plotly_chart(build_radar_mm(d_bl,float(perim_mm),"M3H",camps,"3h (mm)"),use_container_width=True,key=f"cal_3h_{tb}_{bl}")
                        with cB: st.plotly_chart(build_radar_mm(d_bl,float(perim_mm),"M9H",camps,"9h (mm)"),use_container_width=True,key=f"cal_9h_{tb}_{bl}")
                        with cC: st.plotly_chart(build_radar_mm(d_bl,float(perim_mm),"Gap (mm)",camps,"GAP (mm) — Afetados em vermelho",True),use_container_width=True,key=f"cal_gap_{tb}_{bl}")
                        st.divider()

            with sub_tabs[2]:
                st.markdown("### 🎯 Comparativo de Perfil Angular (Relógio vs Calibre)")
                df_cal_raw2 = st.session_state.get("df_calibre_proc", pd.DataFrame())
                if not df_cal_raw2.empty:
                    bd_max = results["severity_by_blade_latest"].copy().rename(columns={"Delta_latest_max_mm":"Max_Gap_BlueDial"})
                    cal_max = df_cal_raw2.groupby(["Turbina","Blade"],as_index=False).agg(Max_Gap_Calibre=("Gap (mm)","max"))
                    for df_m in [bd_max,cal_max]: df_m["Turbina"]=df_m["Turbina"].astype(str).str.strip(); df_m["Blade"]=df_m["Blade"].astype(str).str.strip()
                    cross = pd.merge(bd_max[["Turbina","Blade","Max_Gap_BlueDial"]],cal_max,on=["Turbina","Blade"],how="outer").fillna(0)
                    cross["Diferença (mm)"] = (cross["Max_Gap_BlueDial"]-cross["Max_Gap_Calibre"]).abs()
                    with st.expander("Tabela de Diferencial (Expandir)"):
                        st.dataframe(cross.style.format({c:"{:.2f}" for c in ["Max_Gap_BlueDial","Max_Gap_Calibre","Diferença (mm)"]}).background_gradient(subset=["Diferença (mm)"],cmap="Reds"),use_container_width=True)
                    st.divider()
                    cx1,cx2=st.columns(2)
                    with cx1: cross_turb = st.selectbox("Turbina:",sorted(cross["Turbina"].unique()),key="cross_t")
                    with cx2:
                        cross_blades = sorted(cross[cross["Turbina"]==cross_turb]["Blade"].unique()); cross_blade = st.selectbox("Pá:",cross_blades,key="cross_b")
                    df_bd_blade  = results["latest_sensors"][(results["latest_sensors"]["Turbina"].astype(str)==cross_turb)&(results["latest_sensors"]["Blade"].astype(str)==cross_blade)].copy()
                    df_cal_blade = df_cal_raw2[(df_cal_raw2["Turbina"].astype(str)==cross_turb)&(df_cal_raw2["Blade"].astype(str)==cross_blade)].copy()
                    if not df_bd_blade.empty or not df_cal_blade.empty:
                        fig_cross=go.Figure(); max_bd=0; max_cal=0
                        if not df_bd_blade.empty:
                            bd_pts=[]
                            for sk in MAPA_FUROS.keys():
                                row=df_bd_blade[(df_bd_blade["Casca"]+"-"+df_bd_blade["Regiao"])==sk]; gap=row["Delta_medio_ciclo_mm"].max() if not row.empty else np.nan
                                if not pd.isna(gap): bd_pts.append({"angulo":calculate_angle(MAPA_FUROS[sk]),"gap":gap,"sensor":sk})
                            if bd_pts:
                                df_bd_pts=pd.DataFrame(bd_pts).sort_values("angulo"); max_bd=df_bd_pts["gap"].max()
                                df_bd_pts=pd.concat([df_bd_pts,df_bd_pts.iloc[[0]]],ignore_index=True)
                                fig_cross.add_trace(go.Scatterpolar(r=df_bd_pts["gap"],theta=df_bd_pts["angulo"],mode='lines+markers+text',text=[f"{g:.1f}" for g in df_bd_pts["gap"]],textposition="top center",marker=dict(size=10,color='blue',symbol='square'),line=dict(color='blue',width=2),name='Relógio (Dinâmico)'))
                        if not df_cal_blade.empty:
                            df_cal_blade=df_cal_blade.sort_values("theta_deg"); max_cal=df_cal_blade["Gap (mm)"].max()
                            df_cyc=pd.concat([df_cal_blade,df_cal_blade.iloc[[0]]],ignore_index=True)
                            fig_cross.add_trace(go.Scatterpolar(r=df_cyc["Gap (mm)"],theta=df_cyc["theta_deg"],mode='lines+markers',marker=dict(size=6,color='red'),line=dict(color='red',width=2,dash='dash'),name='Calibre (Estático)'))
                        max_r=max(max_bd,max_cal)*1.2
                        if pd.isna(max_r) or max_r<1: max_r=5.0
                        studs_aus=studs_ausentes_dict.get(cross_blade,[]) if isinstance(studs_ausentes_dict,dict) else []
                        for stud in studs_aus:
                            ang=calculate_angle(stud); fig_cross.add_trace(go.Scatterpolar(r=[max_r*0.98],theta=[ang],mode='markers',marker=dict(size=14,symbol='x',color='black',line=dict(width=2)),hovertext=f"Stud {stud} Ausente ({get_stud_zone(stud)})",name='Stud Ausente',showlegend=False))
                        fig_cross.update_layout(polar=dict(angularaxis=dict(direction="counterclockwise",rotation=0,tickmode='array',tickvals=[0,90,180,270],ticktext=["LE","PS","TE","SS"]),radialaxis=dict(showticklabels=True,range=[0,max_r])),title=f"Sobreposição Dimensional: Pá {cross_blade}",height=550,margin=dict(l=40,r=40,t=60,b=40),legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
                        st.plotly_chart(fig_cross, use_container_width=True)
                        st.divider()
                        col_raw_bd, col_raw_cal = st.columns(2)
                        with col_raw_bd:
                            st.markdown("**Relógios (Dinâmico)**")
                            if not df_bd_blade.empty:
                                df_bd_show=df_bd_blade.copy(); df_bd_show["Sensor"]=df_bd_show["Casca"]+"-"+df_bd_show["Regiao"]
                                df_bd_show=df_bd_show.groupby("Sensor").agg(Gap_mm=("Delta_medio_ciclo_mm","max")).reset_index()
                                df_bd_show["Severity"]=df_bd_show["Gap_mm"].apply(_classify_dash)
                                st.dataframe(df_bd_show.style.format({"Gap_mm":"{:.2f}"}),use_container_width=True,hide_index=True)
                            else: st.warning("Sem dados de relógio para esta pá.")
                        with col_raw_cal:
                            st.markdown("**Calibre (Estático)**")
                            if not df_cal_blade.empty:
                                df_cs=df_cal_blade.copy(); df_cs["Casca"]=df_cs["theta_deg"].apply(lambda a:"PS" if 0<=a<=180 else "SS")
                                df_cs=df_cs[["Casca","Bordo Ref.","Distância (mm)","Gap (mm)"]]
                                st.dataframe(df_cs.sort_values(["Casca","Distância (mm)"]).style.format({"Gap (mm)":"{:.2f}","Distância (mm)":"{:.0f}"}),use_container_width=True,hide_index=True)
                            else: st.warning("Sem dados de calibre para esta pá.")

    # =========================================================================
    # ABA 3: VISÃO CLIENTE
    # =========================================================================
    elif aba_selecionada == "🎯 Visão Cliente":
        st.subheader(f"🎯 Dashboard Executivo Simplificado — Padrão: {modelo_atual}")
        flat_data_cli = []
        for item in verify_data:
            sub = item["data"].copy(); sub["Sensor"]=f"{item['casca']}-{item['regiao']}"; sub["Blade"]=item["blade"]; sub["Inspecao"]=item["inspecao"]; flat_data_cli.append(sub)
        df_viz_cli = pd.concat(flat_data_cli,ignore_index=True) if flat_data_cli else pd.DataFrame()
        for blade in blades_sel[:max_plots_view]:
            st.markdown(f"### ➡️ Análise da Pá SN: {blade}")
            if not df_viz_cli.empty:
                df_b_cli=df_viz_cli[df_viz_cli["Blade"].astype(str)==str(blade)]
                insp_max=df_b_cli["Inspecao"].max() if not df_b_cli.empty else "-"
                dt_insp=results["delta_summary"][(results["delta_summary"]["Blade"].astype(str)==str(blade))&(results["delta_summary"]["Inspecao"]==insp_max)]["Data"].max()
                data_str=pd.to_datetime(dt_insp).strftime('%d/%m/%Y') if pd.notna(dt_insp) else str(insp_max)
                col_ps,col_ss=st.columns(2); plots_ps,plots_ss=[],[]
                for sens in df_b_cli["Sensor"].unique():
                    g_leg=df_b_cli[(df_b_cli["Sensor"]==sens)&(df_b_cli["Inspecao"]==insp_max)].copy()
                    g_leg["Valor_mm"]=pd.to_numeric(g_leg["Valor_mm"],errors="coerce")
                    g_trim=trim_and_rebase(g_leg,xcol="Ponto",ycol="Valor_mm")
                    if g_trim is not None and not g_trim.empty:
                        g_trim["Sensor"]=sens
                        if sens.startswith("PS"): plots_ps.append(g_trim)
                        else: plots_ss.append(g_trim)
                with col_ps:
                    if plots_ps:
                        fig_ps=px.line(pd.concat(plots_ps,ignore_index=True),x="Ponto",y="Valor_mm",color="Sensor",height=350,title=f"PS Curves - {data_str}"); st.plotly_chart(fig_ps,use_container_width=True,key=f"cli_line_ps_{blade}")
                with col_ss:
                    if plots_ss:
                        fig_ss=px.line(pd.concat(plots_ss,ignore_index=True),x="Ponto",y="Valor_mm",color="Sensor",height=350,title=f"SS Curves - {data_str}"); st.plotly_chart(fig_ss,use_container_width=True,key=f"cli_line_ss_{blade}")
            col_radar,col_table=st.columns([3,1])
            df_latest_b=latest_sensors[latest_sensors["Blade"].astype(str)==str(blade)].copy()
            radar_data=[]
            for sk in MAPA_FUROS.keys():
                row=df_latest_b[(df_latest_b["Casca"]+"-"+df_latest_b["Regiao"])==sk]; gap=row["Delta_medio_ciclo_mm"].max() if not row.empty else np.nan
                if not pd.isna(gap): radar_data.append({"Sensor":sk,"Gap_mm":gap,"Severidade":_classify_dash(gap)})
            with col_radar:
                fig_radar=create_radar_chart(blade,latest_sensors,studs_ausentes_dict,engine='plotly',classify_fn=_classify_dash); st.plotly_chart(fig_radar,use_container_width=True,key=f"cli_polar_{blade}")
            with col_table:
                df_radar=pd.DataFrame(radar_data)
                if not df_radar.empty:
                    st.markdown("<br><br><br>",unsafe_allow_html=True); st.dataframe(df_radar.style.format({"Gap_mm":"{:.1f}"}),hide_index=True,use_container_width=True)
            st.divider()

    # =========================================================================
    # ABA 4: DOWNLOADS
    # =========================================================================
    elif aba_selecionada == "📥 Downloads":
        st.subheader("📥 Central de Relatórios e Exportações")
        modelo_ativo = st.session_state.get("modelo_usado","Arthwind")
        st.info(f"Os relatórios abaixo usarão o padrão de severidade **{modelo_ativo}**, conforme selecionado na barra lateral antes do cálculo.")
        st.markdown("---")
        c_d1,c_d2=st.columns(2)
        with c_d1:
            st.markdown("### 1️⃣ Base Consolidada Geral (Excel)")
            if "excel_bytes" not in st.session_state: st.session_state["excel_bytes"]=None
            if st.button("🚀 Processar Base (Excel)"):
                with st.spinner("Processando dados..."):
                    global_res=run_analysis(df_raw,full_process=True,modelo=modelo_ativo); st.session_state["excel_bytes"]=generate_excel_report(global_res["delta_summary"])
            if st.session_state["excel_bytes"] is not None:
                st.download_button("📥 Baixar Excel Consolidado",data=st.session_state["excel_bytes"],file_name=f"Base_Medicao_Gap_{modelo_ativo}.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        with c_d2:
            st.markdown("### 2️⃣ PDF Geral (Engenharia)")
            st.info("Gera um arquivo ZIP com os PDFs técnicos de TODAS as turbinas.")
            if st.button("🚀 Processar ZIP (Engenharia)"):
                turbs_to_run=sorted(df_raw["Turbina"].dropna().unique().tolist())
                if turbs_to_run:
                    zip_buffer,errors=io.BytesIO(),[]
                    total=len(turbs_to_run); my_bar=st.progress(0,text="Preparando lote...")
                    with zipfile.ZipFile(zip_buffer,mode="w",compression=zipfile.ZIP_DEFLATED) as zf:
                        for i,tb in enumerate(turbs_to_run,start=1):
                            my_bar.progress(i/total,text=f"Gerando: {tb} ({int(i/total*100)}%)")
                            df_tb=df_raw[df_raw["Turbina"]==tb].copy()
                            b_sel=sorted(df_tb["SN_da_Pa"].dropna().unique().tolist()); i_sel_z=sorted(df_tb["Inspecao"].dropna().unique().tolist())
                            if b_sel and i_sel_z:
                                try:
                                    res_tb=run_analysis(df_raw,full_process=False,t_sel=[tb],b_sel=b_sel,i_sel=i_sel_z,modelo=modelo_ativo)
                                    pdf_tb=generate_pdf(res_tb,studs_ausentes_dict,modelo=modelo_ativo,windfarm=parque_atual,customer=cliente_atual)
                                    safe=str(tb).replace("/","-").replace("\\","-").strip(); zf.writestr(f"ATW-{safe}-GAP-ENG-{modelo_ativo}.pdf",pdf_tb)
                                except Exception as e: errors.append((tb,str(e)))
                    my_bar.progress(1.0,text=f"✅ ZIP ({modelo_ativo}) concluído!"); zip_buffer.seek(0)
                    st.download_button(f"📥 Baixar ZIP ({modelo_ativo})",data=zip_buffer.getvalue(),file_name=f"Relatorios_Engenharia_{modelo_ativo}.zip",mime="application/zip")
                    if errors: st.warning("Falhas:"); st.dataframe(pd.DataFrame(errors,columns=["Turbina","Erro"]))
        st.markdown("---")
        st.markdown("### 3️⃣ PDF Individual da Turbina (Visão Cliente)")
        st.info(f"Relatório executivo seguindo o padrão **{modelo_ativo}**.")
        col_down_t,col_down_i,col_down_btn=st.columns([2,2,1])
        with col_down_t: down_turb=st.selectbox("Selecione a Turbina:",sorted(df_raw["Turbina"].dropna().unique()),key="down_t")
        with col_down_i: down_insp=st.selectbox("Selecione a Campanha:",sorted(df_raw[df_raw["Turbina"]==down_turb]["Inspecao"].dropna().unique()),key="down_i")
        with col_down_btn:
            st.markdown("<br>",unsafe_allow_html=True)
            if st.button("📄 Processar PDF"):
                with st.spinner(f"Gerando Relatório {modelo_ativo}..."):
                    b_sel_cli=sorted(df_raw[(df_raw["Turbina"]==down_turb)&(df_raw["Inspecao"]==down_insp)]["SN_da_Pa"].dropna().unique().tolist())
                    if b_sel_cli:
                        try:
                            res_cli=run_analysis(df_raw,full_process=False,t_sel=[down_turb],b_sel=b_sel_cli,i_sel=[down_insp],modelo=modelo_ativo)
                            pdf_cli=generate_client_pdf(res_cli,studs_ausentes_dict,modelo=modelo_ativo,windfarm=parque_atual,customer=cliente_atual)
                            safe_tb=str(down_turb).replace("/","-").replace("\\","-").strip(); safe_isp=str(down_insp).replace("/","-").replace("\\","-").strip()
                            nome_pdf=f"ATW-{safe_tb}-{safe_isp}-{modelo_ativo}.pdf"
                            st.session_state["pdf_ind_bytes"]=pdf_cli; st.session_state["pdf_ind_name"]=nome_pdf
                            st.success(f"✅ Relatório da Campanha {down_insp} pronto!")
                        except Exception as e: st.error(f"Erro ao gerar: {e}")
                    else: st.warning("Não há pás disponíveis.")
        if "pdf_ind_bytes" in st.session_state and st.session_state["pdf_ind_bytes"] is not None:
            st.markdown("---")
            st.download_button(label=f"📥 Baixar Agora: {st.session_state['pdf_ind_name']}",data=st.session_state["pdf_ind_bytes"],file_name=st.session_state["pdf_ind_name"],mime="application/pdf",use_container_width=True)
