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

# ---------------------------------------------------------------------
# Imports ReportLab (PDF)
# ---------------------------------------------------------------------
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

# Mapa de furos da circunferência (Baseado em 84 furos)
MAPA_FUROS = {
    "PS-TE": 4, "PS-CTE": 13, "PS-C": 21, "PS-CLE": 29, "PS-LE": 38,
    "SS-LE": 46, "SS-CLE": 55, "SS-C": 63, "SS-CTE": 71, "SS-TE": 80
}

def calculate_angle(furo):
    """
    Converte o furo para ângulo garantindo a posição física:
    - LE (BA)   = Furo 42 -> 0°   (Direita)
    - PS-C      = Furo 21 -> 90°  (Cima)
    - TE (BF)   = Furo 0  -> 180° (Esquerda)
    - SS-C      = Furo 63 -> 270° (Baixo)
    """
    return (180.0 - (furo / 84.0) * 360.0) % 360.0

# ---------------------------------------------------------------------
# PARÂMETROS DA CAPA
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
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Relatório GAP - Arthwind", layout="wide")
st.title("Medição de Gap - Relógio Comparador")

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

uploaded = st.sidebar.file_uploader("Upload Base (Excel/CSV/DB)", type=["csv", "xlsx", "xls", "db"])
df_raw = load_data(uploaded)

if df_raw.empty:
    st.info("Aguardando arquivo...")
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

st.sidebar.subheader("Mapeamento Relógio")
region_options = ["LE", "CLE", "C", "CTE", "TE"]
region_map: Dict[str, str] = {}
for i, dial in enumerate(dial_names):
    def_val = region_options[i] if i < len(region_options) else region_options[-1]
    region_map[dial] = st.sidebar.selectbox(f"{dial}", region_options, index=region_options.index(def_val))

st.sidebar.markdown("---")
st.sidebar.subheader("Filtros (Aplicados na Exportação)")
enable_hampel = st.sidebar.checkbox("Filtro Hampel", value=False)
hampel_window = st.sidebar.slider("Janela Hampel", 5, 101, 21, 2)
hampel_n_sigma = st.sidebar.slider("Limite Hampel (sigma)", 1.0, 6.0, 3.0, 0.5)
enable_deriv = st.sidebar.checkbox("Filtro Derivada", value=False)
deriv_threshold = st.sidebar.slider("Limite Derivada", 0.01, 5.0, 2.0, 0.1)
enable_peak_filter = st.sidebar.checkbox("Filtro Peak", value=True)
peak_threshold_mm = st.sidebar.slider("Limite Peak (mm)", 0.1, 10.0, 2.0, 0.1)
usar_align = st.sidebar.checkbox("Centralizar (Média Zero)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Seleção (Visualização)")
turbinas = sorted(df_raw["Turbina"].dropna().unique().tolist())

if "turb_sel" not in st.session_state: st.session_state["turb_sel"] = [turbinas[0]] if turbinas else []

c_sel, c_clr, c_inv = st.sidebar.columns(3)
with c_sel:
    if st.button("✅ Todas", key="btn_all_turbs", use_container_width=True):
        st.session_state["turb_sel"] = turbinas
        st.rerun()
with c_clr:
    if st.button("🧹 Nenhuma", key="btn_none_turbs", use_container_width=True):
        st.session_state["turb_sel"] = []
        st.rerun()
with c_inv:
    if st.button("🔁 Inverter", key="btn_inv_turbs", use_container_width=True):
        current = set(st.session_state["turb_sel"])
        st.session_state["turb_sel"] = [t for t in turbinas if t not in current]
        st.rerun()

turb_sel = st.sidebar.multiselect("Turbinas", turbinas, key="turb_sel")
df_turb = df_raw[df_raw["Turbina"].isin(turb_sel)].copy() if turb_sel else df_raw.copy()

blades = sorted(df_turb["SN_da_Pa"].dropna().astype(str).unique().tolist())
blades_sel = st.sidebar.multiselect("Pás", blades, default=blades)

insps = sorted(df_turb["Inspecao"].dropna().unique().tolist())
insps_sel = st.sidebar.multiselect("Campanhas", insps, default=insps)

# ---------------------------------------------------------------------
# Lógica Principal
# ---------------------------------------------------------------------
def classify_severity(delta_mm: float) -> str:
    if delta_mm is None or np.isnan(delta_mm) or delta_mm <= 0: return "SEV0"
    d = float(abs(delta_mm))
    if d <= 0.6: return "SEV1"
    if d <= 1.0: return "SEV2"
    if d <= 3.0: return "SEV3"
    if d <= 5.0: return "SEV4"
    return "SEV5"

def severity_color(sev: str) -> str:
    palette = {"SEV0": "#c6efce", "SEV1": "#a9d18e", "SEV2": "#ffd966", "SEV3": "#f4b183", "SEV4": "#ff8c00", "SEV5": "#ff0000"}
    return palette.get(sev, "#ffffff")

def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty: return None
    cols = [str(c) for c in df.columns]
    lower_map = {c: c.lower().strip() for c in cols}
    exact_priority = ["data", "date", "data_coleta", "data de coleta", "inspection_date", "collection_date", "data_inspecao", "data inspeção", "data_inspeção"]
    for c in cols:
        if lower_map[c] in exact_priority: return c
    for c in cols:
        lc = lower_map[c]
        if ("data" in lc or "date" in lc) and all(bad not in lc for bad in ["update", "atual", "criado", "created", "modified"]): return c
    return None

def severity_recommendation(sev: str) -> Tuple[str, dt.timedelta]:
    recs = {
        "SEV0": ("12 Months", dt.timedelta(days=365)), "SEV1": ("6 Months", dt.timedelta(days=182)),
        "SEV2": ("3 Months", dt.timedelta(days=91)), "SEV3": ("1 Month", dt.timedelta(days=30)),
        "SEV4": ("15 Days", dt.timedelta(days=15)), "SEV5": ("Stop Turbine", dt.timedelta(days=0)),
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
    y = pd.to_numeric(g[ycol], errors="coerce")
    y2 = y.interpolate(limit_area="inside") 
    valid = y2.notna() & (y2.abs() > eps)
    if not valid.any(): return None
    start_pos = int(np.argmax(valid.values))
    end_pos = int(len(valid) - 1 - np.argmax(valid.values[::-1]))
    g = g.iloc[start_pos:end_pos + 1].copy()
    g[ycol] = y2.iloc[start_pos:end_pos + 1].values
    g[xcol] = pd.to_numeric(g[xcol], errors="coerce") - float(pd.to_numeric(g[xcol], errors="coerce").iloc[0])
    return g

def process_data_core(df_target: pd.DataFrame) -> pd.DataFrame:
    frames = []
    date_col = detect_date_column(df_target)

    for col, dial in zip(reading_cols, dial_names):
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

    long_df = pd.concat(frames, ignore_index=True)
    long_df.rename(columns={"SN_da_Pa": "Blade"}, inplace=True)

    if enable_hampel:
        def _hampel(g):
            s = g["Valor_mm"]
            if s.notna().sum() < 3: return g
            med = s.rolling(window=hampel_window, center=True, min_periods=1).median()
            mad = (s - med).abs().rolling(window=hampel_window, center=True, min_periods=1).median() * 1.4826
            mask = (s - med).abs() > (hampel_n_sigma * mad)
            g.loc[mask, "Valor_mm"] = np.nan
            return g
        long_df = long_df.groupby(["Turbina", "Blade", "Casca", "Regiao", "Inspecao"], group_keys=False).apply(_hampel)

    if enable_deriv:
        def _deriv(g):
            g = g.sort_values("Ponto")
            dy = g["Valor_mm"].diff().abs()
            mask = dy > deriv_threshold
            g.loc[mask, "Valor_mm"] = np.nan
            return g
        long_df = long_df.groupby(["Turbina", "Blade", "Casca", "Regiao", "Inspecao"], group_keys=False).apply(_deriv)

    if enable_peak_filter:
        def _peak(g):
            med = g["Valor_mm"].rolling(window=5, center=True, min_periods=1).median()
            mask = (g["Valor_mm"] - med).abs() > peak_threshold_mm
            g.loc[mask, "Valor_mm"] = np.nan
            return g
        long_df = long_df.groupby(["Turbina", "Blade", "Casca", "Regiao", "Inspecao"], group_keys=False).apply(_peak)

    if usar_align:
        long_df["Valor_mm"] = long_df.groupby(["Turbina", "Blade", "Casca", "Regiao", "Inspecao"])["Valor_mm"].transform(lambda x: x - x.mean())

    return long_df

def compute_cycle_delta(g: pd.DataFrame) -> Tuple[float, int]:
    g = g.sort_values("Ponto").copy()
    s = pd.to_numeric(g["Valor_mm"], errors="coerce")
    s = s.interpolate(limit_area="inside")
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
    turning_points = np.concatenate([peaks, valleys])
    turning_points.sort()
    for i in range(len(turning_points) - 1):
        idx1, idx2 = turning_points[i], turning_points[i + 1]
        amp = abs(smooth[idx1] - smooth[idx2])
        if amp > 0.25 * global_amp: amplitudes.append(amp)
    if not amplitudes: return global_amp, 1
    return float(np.mean(amplitudes)), max(1, int(len(amplitudes) // 2))

# Modificado para aceitar parâmetros explícitos
def run_analysis(df_in: pd.DataFrame, full_process=False, t_sel=None, b_sel=None, i_sel=None):
    _turb_sel = t_sel if t_sel is not None else turb_sel
    _blades_sel = b_sel if b_sel is not None else blades_sel
    _insps_sel = i_sel if i_sel is not None else insps_sel

    if not full_process:
        df_subset = df_in[
            (df_in["Turbina"].astype(str).isin([str(t) for t in _turb_sel])) &
            (df_in["SN_da_Pa"].astype(str).isin([str(b) for b in _blades_sel])) &
            (df_in["Inspecao"].astype(str).isin([str(i) for i in _insps_sel]))
        ].copy()
    else:
        df_subset = df_in.copy()

    if df_subset.empty: return None

    long_df = process_data_core(df_subset)

    delta_rows = []
    verify_items = []

    groups = long_df.groupby(["Turbina", "Blade", "Casca", "Regiao", "Relogio", "Inspecao"])
    for keys, g in groups:
        turb, blade, casca, reg, relogio, insp = keys
        d, n = compute_cycle_delta(g)
        dmax = pd.to_datetime(g["Data"], errors="coerce", dayfirst=True).max() if "Data" in g.columns else pd.NaT
        delta_rows.append({
            "Turbina": turb, "Blade": blade, "Casca": casca, "Regiao": reg, "Relogio": relogio, "Inspecao": insp,
            "Delta_medio_ciclo_mm": d, "N_ciclos": n, "Data": dmax
        })
        if not full_process:
            verify_items.append({
                "turbina": turb, "blade": blade, "casca": casca, "regiao": reg, "relogio": relogio, "inspecao": insp,
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
            has_data = False
            idx_color = 0

            for (tb, isp), g_trace in g_sensor.groupby(["Turbina", "Inspecao"]):
                g_trim = trim_and_rebase(g_trace, xcol="Ponto", ycol="Valor_mm")
                if g_trim is None or g_trim.empty: continue
                lbl = f"{isp}" if len(_turb_sel) == 1 else f"{tb}-{isp}"
                c = colors_list[idx_color % len(colors_list)]
                ax.plot(g_trim["Ponto"], g_trim["Valor_mm"], label=lbl, linewidth=1.5, color=c)
                val_d = delta_summary[
                    (delta_summary["Turbina"].astype(str) == str(tb)) & (delta_summary["Blade"].astype(str) == str(blade)) &
                    (delta_summary["Casca"].astype(str) == str(casca)) & (delta_summary["Regiao"].astype(str) == str(reg)) &
                    (delta_summary["Relogio"].astype(str) == str(relogio)) & (delta_summary["Inspecao"].astype(str) == str(isp))
                ]["Delta_medio_ciclo_mm"].max()
                stats_rows.append({"Campanha": lbl, "Gap (mm)": val_d})
                idx_color += 1
                has_data = True

            if has_data:
                sensor_name = f"{casca}-{reg} ({relogio})"
                ax.set_title(sensor_name, fontsize=10, fontweight='bold', pad=8)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.legend(fontsize=7, loc='best')
                plt.tight_layout()
                sensors_data_list.append({"sensor": sensor_name, "fig": fig, "stats": stats_rows})
            else:
                plt.close(fig)

        pdf_detailed_data.append((blade, sensors_data_list))

    grouped_all = delta_summary.groupby(["Turbina", "Blade"], as_index=False)["Delta_medio_ciclo_mm"].max()
    grouped_all.rename(columns={"Delta_medio_ciclo_mm": "Delta_max_mm"}, inplace=True)
    grouped_all["Severity"] = grouped_all["Delta_max_mm"].apply(classify_severity)
    grouped_all["Color"] = grouped_all["Severity"].apply(severity_color)

    latest_sensors = pick_latest_rows(delta_summary, ["Turbina", "Blade", "Casca", "Regiao", "Relogio"])
    blade_latest = latest_sensors.groupby(["Turbina", "Blade"], as_index=False).agg(Delta_latest_max_mm=("Delta_medio_ciclo_mm", "max"), Last_Date=("Data", "max"))
    blade_latest["Severity"] = blade_latest["Delta_latest_max_mm"].apply(classify_severity)

    rec_txt, next_dates = [], []
    for sev, d_last in zip(blade_latest["Severity"], blade_latest["Last_Date"]):
        r_txt, delta_t = severity_recommendation(sev)
        rec_txt.append(r_txt)
        next_dates.append(pd.to_datetime(d_last) + delta_t if pd.notna(d_last) else pd.NaT)
    blade_latest["Recommendation"] = rec_txt
    blade_latest["Next_Inspection"] = next_dates
    blade_latest["Color"] = blade_latest["Severity"].apply(severity_color)

    turbine_latest = blade_latest.groupby(["Turbina"], as_index=False).agg(Delta_latest_max_mm=("Delta_latest_max_mm", "max"), Last_Date=("Last_Date", "max"))
    turbine_latest["Severity"] = turbine_latest["Delta_latest_max_mm"].apply(classify_severity)
    turbine_latest["Recommendation"] = turbine_latest["Severity"].apply(lambda s: severity_recommendation(s)[0])
    turbine_latest["Color"] = turbine_latest["Severity"].apply(severity_color)

    reinspection = delta_summary.groupby(["Turbina", "Blade"], as_index=False).agg(Reinspections=("Inspecao", "nunique"), First_Date=("Data", "min"), Last_Date=("Data", "max"))
    reinspection = reinspection.merge(blade_latest[["Turbina", "Blade", "Delta_latest_max_mm", "Severity", "Recommendation", "Next_Inspection"]], on=["Turbina", "Blade"], how="left")

    critical_blades = blade_latest.sort_values("Delta_latest_max_mm", ascending=False).head(15).copy()
    critical_turbines = turbine_latest.sort_values("Delta_latest_max_mm", ascending=False).head(15).copy()

    return {
        "meta": {"turb": ", ".join(map(str, _turb_sel)) if _turb_sel else "All", "blades": _blades_sel, "insps": _insps_sel, "date": cover_date, "campaign_dates": campaign_dates_str},
        "delta_summary": delta_summary,
        "severity_by_blade": grouped_all,
        "severity_by_blade_latest": blade_latest,
        "severity_by_turbine_latest": turbine_latest,
        "reinspection_table": reinspection,
        "critical_blades": critical_blades,
        "critical_turbines": critical_turbines,
        "pdf_detailed_data": pdf_detailed_data,
        "verify_data": verify_items,
        "latest_sensors": latest_sensors 
    }

# ---------------------------------------------------------------------
# Helpers de Desenho Matplotlib (Reutilizáveis Streamlit/PDF)
# ---------------------------------------------------------------------
def create_dual_line_chart(blade, df_viz_cli, colors_list, data_str):
    """Gera o gráfico Matplotlib com PS e SS lado a lado (Usado no PDF Cliente)"""
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
                
    ax_ps.set_title(f"PS Curvas - {data_str}", fontsize=9, fontweight='bold', pad=8)
    ax_ps.grid(True, linestyle='--', alpha=0.5)
    ax_ps.legend(fontsize=7, loc='upper right')
    
    ax_ss.set_title(f"SS Curvas - {data_str}", fontsize=9, fontweight='bold', pad=8)
    ax_ss.grid(True, linestyle='--', alpha=0.5)
    ax_ss.legend(fontsize=7, loc='upper right')
    
    plt.tight_layout()
    img_io = io.BytesIO()
    fig.savefig(img_io, format='png', dpi=150, bbox_inches='tight')
    img_io.seek(0)
    plt.close(fig)
    return img_io

def create_radar_chart_and_table(blade, latest_sensors):
    """Gera gráfico Radar (Matplotlib) e os dados da tabela (Usado nos PDFs)"""
    fig_polar, ax_polar = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    
    theta_circle = np.linspace(0, 2 * math.pi, 200)
    ax_polar.plot(theta_circle, [1.1] * 200, color='black', linewidth=1) 
    ax_polar.plot(theta_circle, [0.9] * 200, color='black', linewidth=1) 
    
    # ---- LINHA DE DIVISÃO DA CASCA (LE ao TE) ----
    ax_polar.plot([0, 0], [0, 1.1], color='black', linewidth=1.5, linestyle='--') # Centro ao LE
    ax_polar.plot([math.pi, math.pi], [0, 1.1], color='black', linewidth=1.5, linestyle='--') # Centro ao TE
    # ----------------------------------------------
    
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

        angulo_graus = calculate_angle(furo)
        angulo_rad = angulo_graus * math.pi / 180.0
        
        val_gap = row["Delta_medio_ciclo_mm"]
        sev = classify_severity(val_gap)
        cor_hex = severity_color(sev)
        
        ax_polar.scatter(angulo_rad, 1, color=cor_hex, s=150, edgecolors='black', zorder=10)
        
        info_text = f"{val_gap:.1f}mm\n{sev}\n{sens_key}"
        ax_polar.text(angulo_rad, 1.35, info_text, ha='center', va='center', fontsize=9.5, fontweight='bold', zorder=11)

        gap_str = f"{val_gap:.1f}".replace(".", ",") if pd.notna(val_gap) else "-"
        table_data.append([sens_key, gap_str, sev])

    ax_polar.text(90 * math.pi / 180.0, 1.65, "PS", ha='center', va='center', fontsize=12, fontweight='bold')
    ax_polar.text(270 * math.pi / 180.0, 1.65, "SS", ha='center', va='center', fontsize=12, fontweight='bold')
    ax_polar.text(180 * math.pi / 180.0, 1.65, "TE", ha='right', va='center', fontsize=12, fontweight='bold')
    ax_polar.text(0 * math.pi / 180.0, 1.65, "LE", ha='left', va='center', fontsize=12, fontweight='bold')

    ax_polar.set_ylim(0, 1.8)
    ax_polar.axis('off') 
    plt.tight_layout()
    
    img_io_polar = io.BytesIO()
    fig_polar.savefig(img_io_polar, format='png', dpi=150, bbox_inches='tight', transparent=True)
    img_io_polar.seek(0)
    plt.close(fig_polar)

    return img_io_polar, table_data

# ---------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------
def get_proportional_image(img_bytes, max_w, max_h):
    try:
        img_bytes.seek(0)
        img_reader = ImageReader(img_bytes)
        iw, ih = img_reader.getSize()
        aspect = ih / float(iw)
        w = max_w
        h = w * aspect
        if h > max_h:
            h = max_h
            w = h / aspect
        img_bytes.seek(0)
        return Image(img_bytes, width=w, height=h)
    except Exception:
        img_bytes.seek(0)
        return Image(img_bytes, width=max_w, height=max_h)

def _draw_wrapped(canvas, text: str, x: float, y: float, max_chars: int, line_h_pt: float):
    if not text: return
    lines = textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)
    for i, line in enumerate(lines):
        canvas.drawString(x, y - i * line_h_pt, line)

def draw_image_cover(canvas, img_reader, x: float, y: float, w: float, h: float):
    try:
        iw, ih = img_reader.getSize()
        if iw <= 0 or ih <= 0: return
        scale = max(w / float(iw), h / float(ih))
        sw, sh = float(iw) * scale, float(ih) * scale
        dx, dy = x - (sw - w) / 2.0, y - (sh - h) / 2.0
        p = canvas.beginPath()
        p.rect(x, y, w, h)
        canvas.saveState()
        canvas.clipPath(p, stroke=0, fill=0)
        canvas.drawImage(img_reader, dx, dy, width=sw, height=sh, mask='auto')
        canvas.restoreState()
    except Exception: pass

def _create_cover_and_intro(doc, results, h1, normal):
    meta = results["meta"]
    sev_df = results.get("severity_by_blade_latest", results.get("severity_by_blade"))
    turbina_txt = meta.get("turb", "-")
    blades_list_txt = ", ".join(map(str, meta.get("blades", [])))
    camp_dates_txt = " | ".join(meta.get("campaign_dates", [])) or "-"

    def draw_cover_full(canvas, _doc):
        canvas.saveState()
        page_w, page_h = A4
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, page_w, page_h, stroke=0, fill=1)
        canvas.setFillColor(colors.HexColor(COVER_BELOW_IMAGE_BG_COLOR))
        canvas.rect(0, 0, page_w, page_h, stroke=0, fill=1)

        cover_bytes = load_image_for_pdf("COVER")
        img_h = page_h * COVER_IMG_H_RATIO
        y = page_h - img_h - (COVER_IMG_TOP_PAD_CM * cm)
        if cover_bytes:
            try:
                img_reader = ImageReader(cover_bytes)
                draw_image_cover(canvas, img_reader, 0, y, page_w, img_h)
            except Exception: pass
        else:
            canvas.setFillColor(colors.HexColor("#dbeef3"))
            canvas.rect(0, y, page_w, img_h, stroke=0, fill=1)

        logo_bytes = load_image_for_pdf("LOGO")
        if logo_bytes:
            try:
                logo_reader = ImageReader(logo_bytes)
                canvas.drawImage(logo_reader, COVER_LOGO_X_CM * cm, page_h - (COVER_LOGO_Y_FROM_TOP_CM * cm), width=COVER_LOGO_W_CM * cm, height=COVER_LOGO_H_CM * cm, mask='auto', preserveAspectRatio=True)
            except Exception: pass

        img_bottom_y = page_h - (page_h * COVER_IMG_H_RATIO) - (COVER_IMG_TOP_PAD_CM * cm)
        bar_h = COVER_TITLE_BAR_H_CM * cm
        bar_y = img_bottom_y + (COVER_TITLE_BAR_Y_FROM_IMG_BOTTOM_CM * cm)
        bar_x = _doc.leftMargin
        bar_w = page_w - _doc.leftMargin - _doc.rightMargin

        canvas.setFillColor(colors.HexColor(COVER_TITLE_BAR_COLOR))
        canvas.rect(bar_x, bar_y, bar_w, bar_h, stroke=0, fill=1)

        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 18)
        canvas.drawCentredString(page_w / 2, bar_y + bar_h * 0.62, "ROOT GAP MEASUREMENT INSPECTION")
        canvas.drawCentredString(page_w / 2, bar_y + bar_h * 0.22, "REPORT")

        label_x, value_x = COVER_META_LABEL_X_CM * cm, COVER_META_VALUE_X_CM * cm
        y0, line_h = COVER_META_START_Y_FROM_BOTTOM_CM * cm, COVER_META_LINE_H_CM * cm

        labels = ["Wtg Serial Number:", "Windfarm:", "Campaign Dates:", "Date (Last):", "Blade Model:", "Customer:"]
        values = [turbina_txt, "COMPLEXO EÓLICO SERRA AZUL", camp_dates_txt, meta.get("date", "-"), "LM47.6", "ENEL"]

        canvas.setFont("Helvetica-Bold", COVER_META_LABEL_SIZE)
        canvas.setFillColor(colors.HexColor("#1F4E79"))
        for i, lab in enumerate(labels): canvas.drawString(label_x, y0 - i * line_h, lab)

        canvas.setFont("Helvetica", COVER_META_VALUE_SIZE)
        canvas.setFillColor(colors.black)
        for i, val in enumerate(values):
            yy = y0 - i * line_h
            if i == 2: _draw_wrapped(canvas, str(val), value_x, yy, max_chars=65, line_h_pt=10)
            else: canvas.drawString(value_x, yy, str(val))
        canvas.restoreState()

    story = [PageBreak()]
    story.append(Paragraph("1. Summary", h1))
    t_toc = Table([
        ["Section", "Page"], 
        ["2. Introduction", "3"], 
        ["3. Conclusion", "3"], 
        ["4. Methodology", "4"], 
        ["5. Scope", "5"], 
        ["6. Damages categorization", "5"], 
        ["7. Inspection Evidence", "6+"]
    ], colWidths=[14 * cm, 2 * cm])
    t_toc.setStyle(TableStyle([('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.lightgrey)]))
    story.append(t_toc)
    story.append(PageBreak())

    story.append(Paragraph("2. Introduction", h1))
    intro_txt = f"On {meta.get('date','-')}, a gap measurement inspection was performed on LM47.6 model blades, serial numbers {blades_list_txt}, installed on the {turbina_txt} wind turbine located at the COMPLEXO EÓLICO SERRA AZUL."
    story.append(Paragraph(intro_txt, normal))
    story.append(Spacer(1, 1 * cm))

    story.append(Paragraph("3. Conclusion", h1))
    if sev_df is not None and not sev_df.empty:
        conc_data = [["Trb-Blade", "Max Gap (mm)", "Severity"]]
        worst_sev_idx = 0
        for _, row in sev_df.iterrows():
            delta_val = row.get("Delta_latest_max_mm", row.get("Delta_max_mm", np.nan))
            gap_fmt = f"{float(delta_val):.1f}".replace(".", ",") if pd.notna(delta_val) else "-"
            lbl = f"{row.get('Turbina','')}-{row.get('Blade','')}" if (',' in turbina_txt or 'Selected' in turbina_txt) else str(row.get("Blade", ""))
            conc_data.append([lbl, gap_fmt, row.get("Severity", "-")])
            s_map = {"SEV0": 0, "SEV1": 1, "SEV2": 2, "SEV3": 3, "SEV4": 4, "SEV5": 5}
            worst_sev_idx = max(worst_sev_idx, s_map.get(row.get("Severity", "SEV0"), 0))

        t_conc = Table(conc_data, colWidths=[6 * cm, 3 * cm, 4 * cm])
        t_conc.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F4E79")), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white), ('GRID', (0, 0), (-1, -1), 0.5, colors.black)]))
        story.append(t_conc)
        recs = ["12 Months", "6 Months", "3 Months", "1 Month", "15 Days", "Stop Turbine"]
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(f"We recommend a new inspection within {recs[worst_sev_idx]}", normal))
    story.append(PageBreak())

    story.append(Paragraph("4. Methodology", h1))
    story.append(Paragraph("The operation developed by Arthwind consists on inspecting the root gap of wind turbine blades using dial indicators, with the equipment installed externally around the blade root.", normal))
    
    m1_bytes = load_image_for_pdf("METOD_ROTOR")
    if m1_bytes: 
        story.append(Table([[get_proportional_image(m1_bytes, max_w=12*cm, max_h=5.5*cm)]], colWidths=[A4[0] - doc.leftMargin - doc.rightMargin], style=[('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Dial indicators are mounted at specific points around the blade root using suction bases and rods. The blade is rotated 360°while the dial indicators remain in position.", normal))
    
    m2_bytes = load_image_for_pdf("METOD_MAPA")
    if m2_bytes: 
        story.append(Table([[get_proportional_image(m2_bytes, max_w=12*cm, max_h=5.5*cm)]], colWidths=[A4[0] - doc.leftMargin - doc.rightMargin], style=[('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("After the rotation, the maximum and minimum displacement values are analyzed to check the total variation at each point. This procedure is repeated for all blades on the turbine.", normal))
    
    m3_bytes = load_image_for_pdf("METOD_BASE")
    if m3_bytes: 
        story.append(Table([[get_proportional_image(m3_bytes, max_w=12*cm, max_h=5.5*cm)]], colWidths=[A4[0] - doc.leftMargin - doc.rightMargin], style=[('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    
    story.append(PageBreak())

    story.append(Paragraph("5. Scope", h1))
    scope_text = "This report presents the findings of the root gap inspection performed on the wind turbine blades. The scope encompasses the analysis of displacement data collected via dial indicators during a full rotor rotation. The primary objective is to evaluate the gap variation at multiple specific points around the circumference (PS, SS, LE, TE), classify the severity of any deviations according to the client standards, and provide actionable maintenance recommendations to ensure the structural integrity and safe operation of the equipment."
    story.append(Paragraph(scope_text, normal))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("6. Damages categorization", h1))
    cat_data = [
        ["Severity", "Description", "Recommendation"],
        ["SEV0", "No gaps detected", "12 Months"], ["SEV1", "Gap up to 0,6mm", "6 Months"],
        ["SEV2", "Gap between 0,6 and 1mm", "3 Months"], ["SEV3", "Gap between 1 and 3mm", "1 Month"],
        ["SEV4", "Gap between 3 and 5mm", "15 Days"], ["SEV5", "Gap higher than 5mm", "Stop Turbine"],
    ]
    t_cat = Table(cat_data, colWidths=[3 * cm, 7 * cm, 4 * cm])
    t_cat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F4E79")), ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#c6efce")), ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#a9d18e")),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#ffd966")), ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor("#f4b183")),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor("#ff8c00")), ('BACKGROUND', (0, 6), (-1, 6), colors.HexColor("#ff0000")),
    ]))
    story.append(t_cat)
    story.append(PageBreak())

    return story, draw_cover_full

def draw_header_footer(canvas, _doc):
    canvas.saveState()
    canvas.setFont("Helvetica-Bold", 10)
    canvas.setFillColor(colors.black)
    canvas.drawString(2 * cm, A4[1] - 1.5 * cm, "Wind Blade Inspection Report - Preventive Maintenance")
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.HexColor("#1F4E79"))
    canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.5 * cm, "Arthwind Visibility and Prediction")
    canvas.setFont("Helvetica-Oblique", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(A4[0] - 2 * cm, A4[1] - 1.9 * cm, "Manufacturing - Construction - Warrant - [Preventive]")
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.black)
    canvas.drawCentredString(A4[0] / 2, 1 * cm, f"{canvas.getPageNumber()}")
    logo_bytes = load_image_for_pdf("LOGO")
    if logo_bytes:
        try:
            logo_reader = ImageReader(logo_bytes)
            canvas.drawImage(logo_reader, A4[0] - 5 * cm, 0.8 * cm, width=3 * cm, height=1.2 * cm, mask='auto', preserveAspectRatio=True)
        except Exception: pass
    canvas.restoreState()

# PDF Engenharia (Detalhado)
def generate_pdf(results: Dict[str, Any], progress_callback=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2.5 * cm, bottomMargin=2 * cm, leftMargin=1.5 * cm, rightMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, textColor=colors.HexColor("#1F4E79"), spaceAfter=10)
    normal = ParagraphStyle("Norm", parent=styles["Normal"], fontSize=10, leading=12, alignment=TA_JUSTIFY)

    story, draw_cover_full = _create_cover_and_intro(doc, results, h1, normal)

    story.append(Paragraph("7. Inspection Evidence", h1))
    usable_w = A4[0] - doc.leftMargin - doc.rightMargin
    pdf_data = results["pdf_detailed_data"]
    latest_sensors = results["latest_sensors"]

    total_blades = len(pdf_data)
    for idx, (blade, sensors_data) in enumerate(pdf_data):
        if progress_callback: progress_callback(idx + 1, total_blades, f"Processando Pá {blade}...")
        
        story.append(Paragraph(f"BLADE {blade}", h1))
        
        img_io_polar, table_data_radar = create_radar_chart_and_table(blade, latest_sensors)
        rl_polar = Image(img_io_polar, width=9*cm, height=9*cm)
        
        t_sev = Table(table_data_radar, colWidths=[3.5*cm, 2*cm, 2*cm])
        table_styles = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]
        for i in range(1, len(table_data_radar)):
            sev_str = table_data_radar[i][2]
            bg_color = severity_color(sev_str)
            table_styles.append(('BACKGROUND', (2, i), (2, i), colors.HexColor(bg_color)))
        t_sev.setStyle(TableStyle(table_styles))
        
        t_bottom = Table([[rl_polar, Spacer(1,1), t_sev]], colWidths=[9 * cm, usable_w * 0.05, usable_w * 0.45])
        t_bottom.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'CENTER'), ('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        
        story.append(t_bottom)
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph("Detailed Sensor Curves:", normal))
        story.append(Spacer(1, 0.3 * cm))

        for item in sensors_data:
            fig = item["fig"]
            img_io = io.BytesIO()
            fig.savefig(img_io, format='png', dpi=100, bbox_inches='tight')
            img_io.seek(0)

            img_w = usable_w * 0.70
            tbl_w = usable_w - img_w
            rl_img = Image(img_io, width=img_w, height=6.5 * cm)

            table_data = [["Campanha", "Gap(mm)"]]
            for stat in item["stats"]:
                gap_val = stat["Gap (mm)"]
                table_data.append([stat["Campanha"], f"{gap_val:.1f}".replace(".", ",") if pd.notna(gap_val) else "-"])

            t_stats = Table(table_data, colWidths=[tbl_w * 0.65, tbl_w * 0.35])
            t_stats.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e0e0e0")), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('FONTSIZE', (0, 0), (-1, -1), 8), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('GRID', (0, 0), (-1, -1), 0.5, colors.grey), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
            t_master = Table([[rl_img, t_stats]], colWidths=[img_w, tbl_w])
            t_master.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'CENTER')]))

            story.append(KeepTogether(t_master))
            story.append(Spacer(1, 0.5 * cm))
        story.append(PageBreak())

    doc.build(story, onFirstPage=draw_cover_full, onLaterPages=draw_header_footer)
    buffer.seek(0)
    return buffer.getvalue()

# PDF Cliente (Resumido / Dashboard)
def generate_client_pdf(results: Dict[str, Any], progress_callback=None):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=2.5 * cm, bottomMargin=2 * cm, leftMargin=1.5 * cm, rightMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, textColor=colors.HexColor("#1F4E79"), spaceAfter=10)
    normal = ParagraphStyle("Norm", parent=styles["Normal"], fontSize=10, leading=12, alignment=TA_JUSTIFY)

    story, draw_cover_full = _create_cover_and_intro(doc, results, h1, normal)
    
    story.append(Paragraph("7. Inspection Evidence", h1))
    usable_w = A4[0] - doc.leftMargin - doc.rightMargin

    verify_data = results["verify_data"]
    latest_sensors = results["latest_sensors"]
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    flat_data = []
    for item in verify_data:
        sub = item["data"].copy()
        sub["Sensor"] = f"{item['casca']}-{item['regiao']}"
        sub["Blade"] = item["blade"]
        sub["Inspecao"] = item["inspecao"]
        flat_data.append(sub)

    df_viz_cli = pd.concat(flat_data, ignore_index=True) if flat_data else pd.DataFrame()

    total_blades = len(results["meta"]["blades"])
    for idx, blade in enumerate(results["meta"]["blades"]):
        if progress_callback: progress_callback(idx + 1, total_blades, f"Montando vista da Pá {blade}...")
        
        story.append(Paragraph(f"BLADE {blade}", h1))
        
        if not df_viz_cli.empty:
            df_b_cli = df_viz_cli[df_viz_cli["Blade"].astype(str) == str(blade)]
            insp_max = df_b_cli["Inspecao"].max() if not df_b_cli.empty else "-"
            
            delta_sum = results["delta_summary"]
            dt_insp = delta_sum[(delta_sum["Blade"].astype(str) == str(blade)) & (delta_sum["Inspecao"] == insp_max)]["Data"].max()
            data_str = pd.to_datetime(dt_insp).strftime('%d/%m/%Y') if pd.notna(dt_insp) else str(insp_max)

            img_io_line = create_dual_line_chart(blade, df_viz_cli, colors_list, data_str)
            story.append(Image(img_io_line, width=usable_w, height=5 * cm))
            story.append(Spacer(1, 0.5 * cm))

        # Gráfico Polar e Tabela
        img_io_polar, table_data = create_radar_chart_and_table(blade, latest_sensors)
        rl_polar = Image(img_io_polar, width=9*cm, height=9*cm)

        # Configura a Tabela do lado direito reduzida
        t_sev = Table(table_data, colWidths=[3.5*cm, 2*cm, 2*cm])
        table_styles = [
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]
        for i in range(1, len(table_data)):
            sev_str = table_data[i][2]
            bg_color = severity_color(sev_str)
            table_styles.append(('BACKGROUND', (2, i), (2, i), colors.HexColor(bg_color)))
        t_sev.setStyle(TableStyle(table_styles))

        t_bottom = Table([[rl_polar, Spacer(1,1), t_sev]], colWidths=[9 * cm, usable_w * 0.05, usable_w * 0.45])
        t_bottom.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'CENTER'), ('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
        
        story.append(t_bottom)
        story.append(PageBreak())

    doc.build(story, onFirstPage=draw_cover_full, onLaterPages=draw_header_footer)
    buffer.seek(0)
    return buffer.getvalue()

# Excel (mantido)
def generate_excel_report(delta_summary: pd.DataFrame):
    if delta_summary is None or delta_summary.empty: return None
    df_pivot = delta_summary.copy()
    df_pivot['Data'] = pd.to_datetime(df_pivot['Data'], errors='coerce')
    df_pivot['Data_Inspeção'] = df_pivot.groupby(['Turbina', 'Blade', 'Inspecao'])['Data'].transform('min')
    # Formata para o Excel e preenche vazios para não sumir no pivot
    df_pivot['Data_Inspeção'] = df_pivot['Data_Inspeção'].dt.strftime('%d/%m/%Y').fillna("-")
    
    # Prepara o Pivot
    df_pivot['Sensor_Key'] = df_pivot['Casca'].astype(str) + '-' + df_pivot['Regiao'].astype(str)
    
    # Adicionamos 'Data_Inspeção' no índice do pivot (dropna=False garante que nada suma)
    pivot = df_pivot.pivot_table(
        index=['Turbina', 'Blade', 'Inspecao', 'Data_Inspeção'], 
        columns='Sensor_Key', 
        values='Delta_medio_ciclo_mm',
        aggfunc='first'
    )
    pivot['Media'] = pivot.mean(axis=1)
    pivot['Maximo'] = pivot.max(axis=1)
    pivot['Severidade'] = pivot['Maximo'].apply(classify_severity)
    pivot = pivot.reset_index()

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pivot.to_excel(writer, sheet_name='Base_Consolidada', index=False)
        wb = writer.book
        ws = writer.sheets['Base_Consolidada']
        fmt_border = wb.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})
        fmt_head = wb.add_format({'bold': True, 'align': 'center', 'border': 1, 'bg_color': '#D3D3D3'})
        hex_colors = {"SEV0": "#c6efce", "SEV1": "#a9d18e", "SEV2": "#ffd966", "SEV3": "#f4b183", "SEV4": "#ff8c00", "SEV5": "#ff0000"}
        sev_fmts = {k: wb.add_format({'bg_color': v, 'border': 1, 'align': 'center'}) for k, v in hex_colors.items()}
        (max_r, max_c) = pivot.shape
        for c, val in enumerate(pivot.columns.values): ws.write(0, c, val, fmt_head)
        col_sev = pivot.columns.get_loc('Severidade')
        for r in range(max_r):
            sev_val = pivot.iloc[r, col_sev]
            for c in range(max_c):
                val = pivot.iloc[r, c]
                ws.write(r + 1, c, val if c == col_sev else ("" if pd.isna(val) else val), sev_fmts.get(sev_val, fmt_border) if c == col_sev else fmt_border)
    output.seek(0)
    return output.getvalue()

# ---------------------------------------------------------------------
# UI Principal
# ---------------------------------------------------------------------
if st.button("Calcular Análise"):
    with st.spinner("Calculando Visualização..."):
        results = run_analysis(df_raw, full_process=False)
        st.session_state["results"] = results

if "results" in st.session_state and st.session_state["results"] is not None:
    results = st.session_state["results"]

    # --- Criação das Abas ---
    tab_resumo, tab_eng, tab_cli = st.tabs(["📊 Resumo Executivo", "⚙️ Visão Engenharia", "🎯 Visão Cliente"])

    # =========================================================================
    # ABA 1: RESUMO EXECUTIVO (Global Dash)
    # =========================================================================
    with tab_resumo:
        st.subheader("📊 Resumo Global")
        
        qtd_turbs = len(turb_sel) if turb_sel else df_raw['Turbina'].nunique()
        qtd_blades = len(blades_sel) if blades_sel else df_raw['SN_da_Pa'].nunique()
        qtd_insps = len(insps_sel) if insps_sel else df_raw['Inspecao'].nunique()

        c_metric1, c_metric2, c_metric3 = st.columns(3)
        c_metric1.metric("Turbinas Inspecionadas", qtd_turbs)
        c_metric2.metric("Pás Inspecionadas", qtd_blades)
        c_metric3.metric("Inspeções Realizadas", qtd_insps)

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Pás mais críticas (Top 15)**")
            cb = results.get("critical_blades", pd.DataFrame())
            if not cb.empty:
                cb_show = cb.copy()
                cb_show["Delta_latest_max_mm"] = cb_show["Delta_latest_max_mm"].round(1)
                if "Last_Date" in cb_show.columns: cb_show["Last_Date"] = pd.to_datetime(cb_show["Last_Date"], errors="coerce").dt.strftime("%d-%m-%Y")
                if "Next_Inspection" in cb_show.columns: cb_show["Next_Inspection"] = pd.to_datetime(cb_show["Next_Inspection"], errors="coerce").dt.strftime("%d-%m-%Y")
                st.dataframe(cb_show, use_container_width=True)

        with c2:
            st.markdown("**Turbinas mais críticas (Top 15)**")
            ct = results.get("critical_turbines", pd.DataFrame())
            if not ct.empty:
                ct_show = ct.copy()
                ct_show["Delta_latest_max_mm"] = ct_show["Delta_latest_max_mm"].round(1)
                if "Last_Date" in ct_show.columns: ct_show["Last_Date"] = pd.to_datetime(ct_show["Last_Date"], errors="coerce").dt.strftime("%d-%m-%Y")
                st.dataframe(ct_show, use_container_width=True)

        st.markdown("---")
        
        # --- PAINEL DE GESTÃO DE PRÓXIMAS INSPEÇÕES ---
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
            
            df_gestao["Última Inspeção"] = pd.to_datetime(df_gestao["Last_Date"], errors="coerce").dt.strftime("%d/%m/%Y")
            df_gestao["Próxima Inspeção"] = df_gestao["Next_Inspection_dt"].dt.strftime("%d/%m/%Y")
            
            cols_show = ["Turbina", "Blade", "Severity", "Última Inspeção", "Recommendation", "Próxima Inspeção", "Dias Restantes", "Status"]
            
            # Ordena com as mais atrasadas primeiro
            df_gestao_show = df_gestao[cols_show].sort_values(by="Dias Restantes", ascending=True, na_position="last")
            
            def color_status(val):
                if val == "🚨 Vencida": return 'background-color: #ff4c4c; color: white; font-weight: bold;'
                if val == "⚠️ Próxima (30d)": return 'background-color: #ffc107; color: black; font-weight: bold;'
                if val == "✅ No Prazo": return 'background-color: #28a745; color: white;'
                return ''
            
            # Aplica a cor na coluna 'Status'
            st.dataframe(df_gestao_show.style.map(color_status, subset=["Status"]), use_container_width=True, hide_index=True)
        else:
            st.warning("Dados insuficientes para gerar a gestão de próximas inspeções.")
        
        st.markdown("---")
        st.markdown("### 📈 Distribuição por Severidade")
        sev_order = ["SEV0", "SEV1", "SEV2", "SEV3", "SEV4", "SEV5"]
        g1, g2 = st.columns(2)
        with g1:
            bl = results.get("severity_by_blade_latest", pd.DataFrame())
            if not bl.empty and "Severity" in bl.columns:
                counts = bl["Severity"].value_counts().reindex(sev_order, fill_value=0).reset_index()
                counts.columns = ["Severity", "Count"]
                st.plotly_chart(px.bar(counts, x="Severity", y="Count", title="Pás por Severidade"), use_container_width=True)

        with g2:
            tl = results.get("severity_by_turbine_latest", pd.DataFrame())
            if not tl.empty and "Severity" in tl.columns:
                counts = tl["Severity"].value_counts().reindex(sev_order, fill_value=0).reset_index()
                counts.columns = ["Severity", "Count"]
                st.plotly_chart(px.bar(counts, x="Severity", y="Count", title="Turbinas por Severidade"), use_container_width=True)

    # =========================================================================
    # ABA 2: VISÃO ENGENHARIA 
    # =========================================================================
    with tab_eng:
        st.subheader("🔍 Detalhe por Sensor (Comparativo - Analítico)")
        verify_data = results["verify_data"]
        latest_sensors = results["latest_sensors"]
        
        flat_data = []
        for item in verify_data:
            sub = item["data"].copy()
            sub["Legenda"] = f"{item['turbina']} - {item['inspecao']}"
            sub["Sensor"] = f"{item['casca']} - {item['regiao']} - {item.get('relogio', '')}"
            sub["Blade"] = item["blade"]
            sub["Delta_Calc_mm"] = item["delta"]
            flat_data.append(sub)

        chart_counter = 0
        if flat_data:
            df_viz = pd.concat(flat_data, ignore_index=True)
            for blade, g_blade in df_viz.groupby("Blade"):
                st.markdown(f"### ➡️ Pá {blade}")
                
                df_latest_b = latest_sensors[latest_sensors["Blade"].astype(str) == str(blade)].copy()
                fig_polar_eng = go.Figure()
                
                # ---- LINHA DE DIVISÃO DA CASCA (LE ao TE) ----
                fig_polar_eng.add_trace(go.Scatterpolar(
                    r=[1.1, 0, 1.1], theta=[0, 0, 180], mode='lines',
                    line=dict(color='black', width=1.5, dash='dash'), hoverinfo='none', showlegend=False
                ))
                # ----------------------------------------------
                
                for f in range(84):
                    ang_furo = calculate_angle(f)
                    fig_polar_eng.add_trace(go.Scatterpolar(r=[0.9, 1.1], theta=[ang_furo, ang_furo], mode='lines', line=dict(color='grey', width=0.5), showlegend=False, hoverinfo='none'))
                
                for sens_key in MAPA_FUROS.keys():
                    row = df_latest_b[ (df_latest_b["Casca"] + "-" + df_latest_b["Regiao"]) == sens_key]
                    gap = row["Delta_medio_ciclo_mm"].max() if not row.empty else np.nan
                    if pd.isna(gap): continue
                    
                    sev = classify_severity(gap)
                    cor = severity_color(sev)
                    ang = calculate_angle(MAPA_FUROS[sens_key])
                    
                    fig_polar_eng.add_trace(go.Scatterpolar(
                        r=[1], theta=[ang], mode='markers',
                        marker=dict(size=15, color=cor, line=dict(color='black', width=1)),
                        text=[f"<b>{sens_key}</b><br>Gap: {gap:.1f}mm<br>Sev: {sev}"], hoverinfo='text', name=sens_key
                    ))
                
                fig_polar_eng.update_layout(
                    polar=dict(
                        angularaxis=dict(direction="counterclockwise", rotation=0, tickmode='array', tickvals=np.linspace(0, 360, 12, endpoint=False), showticklabels=False),
                        radialaxis=dict(showticklabels=False, range=[0, 1.3])
                    ),
                    showlegend=False, height=350, margin=dict(l=30, r=30, t=30, b=30)
                )
                fig_polar_eng.add_annotation(x=0.5, y=1.05, text="<b>PS</b>", showarrow=False, xref="paper", yref="paper")
                fig_polar_eng.add_annotation(x=0.5, y=-0.05, text="<b>SS</b>", showarrow=False, xref="paper", yref="paper")
                fig_polar_eng.add_annotation(x=-0.05, y=0.5, text="<b>TE</b>", showarrow=False, xref="paper", yref="paper")
                fig_polar_eng.add_annotation(x=1.05, y=0.5, text="<b>LE</b>", showarrow=False, xref="paper", yref="paper")

                col_rad, col_blank = st.columns([1, 2])
                with col_rad:
                    st.plotly_chart(fig_polar_eng, use_container_width=True, key=f"eng_polar_{blade}")

                sensors = sorted(g_blade["Sensor"].unique())
                for sens in sensors:
                    g_sens = g_blade[g_blade["Sensor"] == sens].copy()
                    c_graph, c_table = st.columns([3, 1])

                    plots = []
                    for leg, g_leg in g_sens.groupby("Legenda"):
                        g_leg = g_leg.copy()
                        g_leg["Valor_mm"] = pd.to_numeric(g_leg["Valor_mm"], errors="coerce")
                        g_trim = trim_and_rebase(g_leg, xcol="Ponto", ycol="Valor_mm")
                        if g_trim is not None and not g_trim.empty:
                            g_trim["Legenda"] = leg
                            plots.append(g_trim)

                    with c_graph:
                        chart_counter += 1
                        if plots:
                            g_plot = pd.concat(plots, ignore_index=True)
                            fig = px.line(g_plot, x="Ponto", y="Valor_mm", color="Legenda", title=f"Sensor: {sens}", height=350)
                            st.plotly_chart(fig, use_container_width=True, key=f"eng_{blade}_{sens}_{chart_counter}")
                    with c_table:
                        stats = g_sens.groupby("Legenda")["Delta_Calc_mm"].mean().reset_index()
                        stats.rename(columns={"Legenda": "Campanha", "Delta_Calc_mm": "Gap (mm)"}, inplace=True)
                        st.dataframe(stats.style.format({"Gap (mm)": "{:.1f}"}), hide_index=True, use_container_width=True)
                    st.divider()

        # --- OTIMIZAÇÃO: PDF INDIVIDUAL SÓ RODA SE PEDIR ---
        st.markdown("---")
        if st.checkbox("🖨️ Preparar PDF da Engenharia para download"):
            progress_bar_eng = st.progress(0, text="Iniciando montagem do PDF...")
            def update_bar_eng(current, total, msg):
                pct = int((current / total) * 100) if total > 0 else 0
                progress_bar_eng.progress(current / total if total > 0 else 0, text=f"{msg} ({pct}%)")
            
            pdf_bytes_eng = generate_pdf(results, progress_callback=update_bar_eng)
            progress_bar_eng.progress(1.0, text="✅ PDF de Engenharia pronto!")
            st.download_button("📥 Liberado! Baixar PDF (Engenharia)", data=pdf_bytes_eng, file_name="Relatorio_Engenharia.pdf", mime="application/pdf")

    # =========================================================================
    # ABA 3: VISÃO CLIENTE
    # =========================================================================
    with tab_cli:
        st.subheader("🎯 Dashboard Executivo Simplificado (Por Pá)")
        
        flat_data_cli = []
        for item in verify_data:
            sub = item["data"].copy()
            sub["Sensor"] = f"{item['casca']}-{item['regiao']}"
            sub["Blade"] = item["blade"]
            sub["Inspecao"] = item["inspecao"]
            flat_data_cli.append(sub)
        
        df_viz_cli = pd.concat(flat_data_cli, ignore_index=True) if flat_data_cli else pd.DataFrame()
        latest_sensors = results["latest_sensors"]

        for blade in blades_sel:
            st.markdown(f"### ➡️ Análise da Pá SN: {blade}")
            
            if not df_viz_cli.empty:
                df_b_cli = df_viz_cli[df_viz_cli["Blade"].astype(str) == str(blade)]
                insp_max = df_b_cli["Inspecao"].max() if not df_b_cli.empty else "-"
                
                delta_sum = results["delta_summary"]
                dt_insp = delta_sum[(delta_sum["Blade"].astype(str) == str(blade)) & (delta_sum["Inspecao"] == insp_max)]["Data"].max()
                data_str = pd.to_datetime(dt_insp).strftime('%d/%m/%Y') if pd.notna(dt_insp) else str(insp_max)
                
                col_ps, col_ss = st.columns(2)
                
                plots_ps, plots_ss = [], []
                for sens in df_b_cli["Sensor"].unique():
                    g_leg = df_b_cli[(df_b_cli["Sensor"] == sens) & (df_b_cli["Inspecao"] == insp_max)].copy()
                    g_leg["Valor_mm"] = pd.to_numeric(g_leg["Valor_mm"], errors="coerce")
                    g_trim = trim_and_rebase(g_leg, xcol="Ponto", ycol="Valor_mm")
                    if g_trim is not None and not g_trim.empty:
                        g_trim["Sensor"] = sens
                        if sens.startswith("PS"): plots_ps.append(g_trim)
                        else: plots_ss.append(g_trim)
                
                with col_ps:
                    if plots_ps:
                        g_plot_ps = pd.concat(plots_ps, ignore_index=True)
                        fig_ps = px.line(g_plot_ps, x="Ponto", y="Valor_mm", color="Sensor", height=350, title=f"PS Curvas - {data_str}")
                        st.plotly_chart(fig_ps, use_container_width=True, key=f"cli_line_ps_{blade}")

                with col_ss:
                    if plots_ss:
                        g_plot_ss = pd.concat(plots_ss, ignore_index=True)
                        fig_ss = px.line(g_plot_ss, x="Ponto", y="Valor_mm", color="Sensor", height=350, title=f"SS Curvas - {data_str}")
                        st.plotly_chart(fig_ss, use_container_width=True, key=f"cli_line_ss_{blade}")

            col_radar, col_table = st.columns([3, 1])
            
            df_latest_b = latest_sensors[latest_sensors["Blade"].astype(str) == str(blade)].copy()
            radar_data = []
            
            for sens_key in MAPA_FUROS.keys():
                row = df_latest_b[ (df_latest_b["Casca"] + "-" + df_latest_b["Regiao"]) == sens_key]
                gap = row["Delta_medio_ciclo_mm"].max() if not row.empty else np.nan
                sev = classify_severity(gap)
                cor = severity_color(sev)
                furo = MAPA_FUROS[sens_key]
                angulo = calculate_angle(furo) 

                text_html = f"<b>{sens_key}</b><br>Gap: {gap:.1f}mm<br>Sev: {sev}"
                radar_data.append({"Sensor": sens_key, "Ângulo": angulo, "Gap_mm": gap, "Severidade": sev, "Cor": cor, "Text": text_html})
            
            with col_radar:
                fig_polar = go.Figure()
                
                # ---- LINHA DE DIVISÃO DA CASCA (LE ao TE) ----
                fig_polar.add_trace(go.Scatterpolar(
                    r=[1.1, 0, 1.1], theta=[0, 0, 180], mode='lines',
                    line=dict(color='black', width=1.5, dash='dash'), hoverinfo='none', showlegend=False
                ))
                # ----------------------------------------------
                
                for f in range(84):
                    ang_furo = calculate_angle(f)
                    fig_polar.add_trace(go.Scatterpolar(
                        r=[0.9, 1.1], theta=[ang_furo, ang_furo], mode='lines',
                        line=dict(color='grey', width=0.5), hoverinfo='none', showlegend=False
                    ))
                
                df_radar = pd.DataFrame(radar_data).dropna(subset=['Gap_mm'])
                fig_polar.add_trace(go.Scatterpolar(
                    r=[1] * len(df_radar), theta=df_radar["Ângulo"], mode='markers',
                    marker=dict(size=18, color=df_radar["Cor"], line=dict(color='black', width=1)),
                    text=df_radar["Text"], hoverinfo='text', name='Sensores'
                ))
                
                fig_polar.update_layout(
                    polar=dict(
                        angularaxis=dict(direction="counterclockwise", rotation=0, tickmode='array', tickvals=np.linspace(0, 360, 12, endpoint=False), showticklabels=False),
                        radialaxis=dict(showticklabels=False, range=[0, 1.3]),
                        bgcolor='rgba(0,0,0,0)' 
                    ),
                    showlegend=False, height=500, margin=dict(l=30, r=30, t=30, b=30), paper_bgcolor='rgba(0,0,0,0)'
                )
                
                fig_polar.add_annotation(x=0.5, y=1.05, text="<b>PS</b>", showarrow=False, xref="paper", yref="paper", font=dict(size=14))
                fig_polar.add_annotation(x=0.5, y=-0.05, text="<b>SS</b>", showarrow=False, xref="paper", yref="paper", font=dict(size=14))
                fig_polar.add_annotation(x=-0.05, y=0.5, text="<b>TE</b>", showarrow=False, xref="paper", yref="paper", font=dict(size=14))
                fig_polar.add_annotation(x=1.05, y=0.5, text="<b>LE</b>", showarrow=False, xref="paper", yref="paper", font=dict(size=14))
                
                st.plotly_chart(fig_polar, use_container_width=True, key=f"cli_polar_{blade}")

            with col_table:
                if not df_radar.empty:
                    st.markdown("<br><br><br>", unsafe_allow_html=True) 
                    st.dataframe(df_radar[["Sensor", "Gap_mm", "Severidade"]].style.format({"Gap_mm": "{:.1f}"}), hide_index=True, use_container_width=True)

            st.divider()

        # --- OTIMIZAÇÃO: PDF INDIVIDUAL SÓ RODA SE PEDIR ---
        st.markdown("---")
        if st.checkbox("🖨️ Preparar PDF do Cliente para download"):
            progress_bar_cli = st.progress(0, text="Iniciando montagem do PDF Executivo...")
            def update_bar_cli(current, total, msg):
                pct = int((current / total) * 100) if total > 0 else 0
                progress_bar_cli.progress(current / total if total > 0 else 0, text=f"{msg} ({pct}%)")
            
            pdf_bytes_cli = generate_client_pdf(results, progress_callback=update_bar_cli)
            progress_bar_cli.progress(1.0, text="✅ PDF do Cliente pronto!")
            
            if len(turb_sel) == 1 and len(insps_sel) == 1:
                tb_name = str(turb_sel[0]).replace("/", "-").replace("\\", "-").strip()
                dt_sum = results["delta_summary"]
                dt_val = dt_sum["Data"].max() if not dt_sum.empty else pd.NaT
                dt_str_file = pd.to_datetime(dt_val).strftime('%d-%m-%y') if pd.notna(dt_val) else str(insps_sel[0]).replace("/", "-")
                cli_filename = f"ATW-{tb_name}-{dt_str_file}.pdf"
            else:
                cli_filename = "Relatorio_Cliente_Consolidado.pdf"
                
            st.download_button("📥 Liberado! Baixar PDF (Cliente / Executivo)", data=pdf_bytes_cli, file_name=cli_filename, mime="application/pdf", key="btn_pdf_cli")

    # NOVO: Painel de Engenharia (Dashboard Temporal)
    st.markdown("---")
    st.subheader("🛠️ Painel de Engenharia - Progressão de Deslocamento")
    st.info("Acompanhe a evolução do Gap máximo por sensor ao longo das inspeções.")

    df_eng = results.get("delta_summary", pd.DataFrame()).copy()
    
    if not df_eng.empty:
        df_eng["Sensor"] = df_eng["Casca"].astype(str) + "-" + df_eng["Regiao"].astype(str)
        # Garante que a ordenação seja pela Data e, em caso de empate/falta, pela Inspeção
        df_eng = df_eng.sort_values(by=["Data", "Inspecao"])

        c_eng1, c_eng2 = st.columns(2)
        with c_eng1:
            eng_turb = st.selectbox("Selecione a Turbina (Painel Eng.):", sorted(df_eng["Turbina"].unique()), key="eng_t")
        with c_eng2:
            eng_blades_opts = sorted(df_eng[df_eng["Turbina"] == eng_turb]["Blade"].unique())
            eng_blade = st.selectbox("Selecione a Pá (Painel Eng.):", eng_blades_opts, key="eng_b")

        df_eng_plot = df_eng[(df_eng["Turbina"] == eng_turb) & (df_eng["Blade"] == eng_blade)]
        df_eng_ps = df_eng_plot[df_eng_plot["Casca"] == "PS"]
        df_eng_ss = df_eng_plot[df_eng_plot["Casca"] == "SS"]

        g_eng1, g_eng2 = st.columns(2)
        with g_eng1:
            if not df_eng_ps.empty:
                fig_ps = px.line(df_eng_ps, x="Data", y="Delta_medio_ciclo_mm", color="Sensor", 
                                 hover_data=["Inspecao"], markers=True, 
                                 title=f"Resultados - Deslocamento - PS (Pá {eng_blade})")
                fig_ps.add_hline(y=5.0, line_dash="dash", line_color="red", annotation_text="Limite SEV5 (5.0mm)", annotation_position="top left")
                fig_ps.update_layout(yaxis_title="Deslocamento (mm)", xaxis_title="Data da Inspeção")
                
                # ADICIONE ESTA LINHA PARA FORÇAR O FORMATO DA DATA:
                fig_ps.update_xaxes(tickformat="%d/%m/%Y") 
                
                st.plotly_chart(fig_ps, use_container_width=True, key="fig_ps_eng")
            else:
                st.warning("Sem dados PS para esta pá.")

        with g_eng2:
            if not df_eng_ss.empty:
                fig_ss = px.line(df_eng_ss, x="Data", y="Delta_medio_ciclo_mm", color="Sensor", 
                                 hover_data=["Inspecao"], markers=True, 
                                 title=f"Resultados - Deslocamento - SS (Pá {eng_blade})")
                fig_ss.add_hline(y=5.0, line_dash="dash", line_color="red", annotation_text="Limite SEV5 (5.0mm)", annotation_position="top left")
                fig_ss.update_layout(yaxis_title="Deslocamento (mm)", xaxis_title="Data da Inspeção")
                
                # ADICIONE ESTA LINHA PARA FORÇAR O FORMATO DA DATA:
                fig_ss.update_xaxes(tickformat="%d/%m/%Y")
                
                st.plotly_chart(fig_ss, use_container_width=True, key="fig_ss_eng")
            else:
                st.warning("Sem dados SS para esta pá.")

    # =========================================================================
    # EXPORTAÇÕES GLOBAIS
    # =========================================================================
    st.markdown("---")
    st.markdown("### 📦 Exportações Globais")
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        st.info("Gera Excel com a base consolidada.")
        if "excel_bytes" not in st.session_state: st.session_state["excel_bytes"] = None
        if st.button("🚀 Gerar Base Consolidada (Excel)"):
            with st.spinner("Processando..."):
                global_res = run_analysis(df_raw, full_process=True)
                st.session_state["excel_bytes"] = generate_excel_report(global_res["delta_summary"])
        if st.session_state["excel_bytes"] is not None:
            st.download_button("📥 Download Excel Completo", data=st.session_state["excel_bytes"], file_name="Base_Consolidada.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    with col_exp2:
        st.info("ZIP: PDFs Completos (Engenharia).")
        if st.button("🚀 Gerar ZIP com PDFs (Engenharia)"):
            turbs_to_run = sorted(df_raw["Turbina"].dropna().unique().tolist())
            if turbs_to_run:
                zip_buffer, errors = io.BytesIO(), []
                total = len(turbs_to_run)
                
                # BARRA DE PROGRESSO DO LOTE
                my_bar_zip_eng = st.progress(0, text="Preparando lote de Engenharia...")

                with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for i, tb in enumerate(turbs_to_run, start=1):
                        pct_zip = int((i / total) * 100)
                        my_bar_zip_eng.progress(i / total, text=f"Gerando Engenharia: Turbina {tb} ({pct_zip}%) - {i}/{total}")
                        
                        df_tb = df_raw[df_raw["Turbina"] == tb].copy()
                        b_sel = sorted(df_tb["SN_da_Pa"].dropna().unique().tolist())
                        i_sel = sorted(df_tb["Inspecao"].dropna().unique().tolist())

                        if b_sel and i_sel:
                            try:
                                res_tb = run_analysis(df_raw, full_process=False, t_sel=[tb], b_sel=b_sel, i_sel=i_sel)
                                pdf_tb = generate_pdf(res_tb)
                                safe_name = str(tb).replace("/", "-").replace("\\", "-").strip()
                                zf.writestr(f"ATW-{safe_name}-GAP-ENG.pdf", pdf_tb)
                            except Exception as e:
                                errors.append((tb, str(e)))

                my_bar_zip_eng.progress(1.0, text="✅ ZIP de Engenharia concluído!")
                zip_buffer.seek(0)
                
                st.download_button("📥 Baixar ZIP (Engenharia)", data=zip_buffer.getvalue(), file_name="Relatorios_Engenharia.zip", mime="application/zip")
                if errors: st.warning("Falhas:"); st.dataframe(pd.DataFrame(errors, columns=["Turbina", "Erro"]))

    with col_exp3:
        st.info("ZIP: PDFs Executivos INDIVIDUAIS por Campanha.")
        if st.button("🚀 Gerar ZIP com PDFs (Cliente)"):
            # Identifica todas as combinações únicas de Turbina e Campanha
            combos = df_raw[['Turbina', 'Inspecao']].dropna().drop_duplicates().values.tolist()
            if combos:
                zip_buffer_cli = io.BytesIO()
                errors_cli = []
                total_cli = len(combos)
                
                # BARRA DE PROGRESSO DO LOTE CLIENTE
                my_bar_zip_cli = st.progress(0, text="Preparando lote de relatórios Individuais (Cliente)...")
                
                with zipfile.ZipFile(zip_buffer_cli, mode="w", compression=zipfile.ZIP_DEFLATED) as zf_cli:
                    for i, (tb, insp) in enumerate(combos, start=1):
                        pct_zip_cli = int((i / total_cli) * 100)
                        my_bar_zip_cli.progress(i / total_cli, text=f"Gerando Cliente: {tb} | {insp} ({pct_zip_cli}%) - {i}/{total_cli}")
                        
                        df_tb_insp = df_raw[(df_raw["Turbina"] == tb) & (df_raw["Inspecao"] == insp)].copy()
                        b_sel_cli = sorted(df_tb_insp["SN_da_Pa"].dropna().unique().tolist())
                        
                        if b_sel_cli:
                            try:
                                res_cli = run_analysis(df_raw, full_process=False, t_sel=[tb], b_sel=b_sel_cli, i_sel=[insp])
                                
                                ds = res_cli["delta_summary"]
                                dt_max = ds["Data"].max() if not ds.empty else pd.NaT
                                data_str_cli = pd.to_datetime(dt_max).strftime("%d-%m-%y") if pd.notna(dt_max) else str(insp).replace("/", "-")
                                
                                pdf_tb_cli = generate_client_pdf(res_cli)
                                safe_tb_cli = str(tb).replace("/", "-").replace("\\", "-").strip()
                                
                                filename_cli = f"ATW-{safe_tb_cli}-{data_str_cli}.pdf"
                                zf_cli.writestr(filename_cli, pdf_tb_cli)
                            except Exception as e:
                                errors_cli.append((f"{tb} - {insp}", str(e)))
                                
                my_bar_zip_cli.progress(1.0, text="✅ ZIP do Cliente concluído!")
                zip_buffer_cli.seek(0)
                
                st.download_button("📥 Baixar ZIP (Cliente)", data=zip_buffer_cli.getvalue(), file_name="Relatorios_Cliente_Por_Campanha.zip", mime="application/zip")
                if errors_cli:
                    st.warning("Falhas ao gerar:")
                    st.dataframe(pd.DataFrame(errors_cli, columns=["Turbina/Campanha", "Erro"]))
