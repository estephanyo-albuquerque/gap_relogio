# app_curvas_relogios_v4_46_FINAL_PDF_SIDE_BY_SIDE_EXEC.py
# v4.46 — Ajuste de CAPA (canvas absoluto) + Plotly estável no Streamlit (keys) + mantém datas reais e curvas rebased
# - Capa agora é desenhada 100% no canvas (posições fixas): faixa branca à esquerda + imagem à direita + barra azul sobre a imagem.
# - Datas reais das coletas: por campanha (Inspecao) usa a data mais recente; "Date (Last)" = mais recente geral.
# - Curvas: remove trecho inicial sem leitura e rebasa X (Streamlit + PDF).
# - Streamlit: adiciona keys únicas para plotly_chart/dataframe dentro de loops (evita “só aparece depois de abrir/fechar” e “só últimos gráficos”).

import io
import os
import datetime as dt
import zipfile
import textwrap
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
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
# CONFIGURAÇÃO DE IMAGENS
# ---------------------------------------------------------------------
IMG_DIR = "imagens"

FILES_CONFIG = {
    "LOGO": "logo",
    "COVER": "capa",
    "METOD_ROTOR": "metodologia_rotor",
    "METOD_MAPA": "metodologia_2",
    "METOD_BASE": "metodologia_3"
}

# ---------------------------------------------------------------------
# PARÂMETROS DA CAPA (AJUSTE AQUI)
# ---------------------------------------------------------------------
COVER_LEFT_STRIP_W_CM = 4.8      # largura da faixa branca à esquerda (logo)
COVER_IMG_H_RATIO = 0.7         # altura da imagem (fração da altura da página)
COVER_IMG_TOP_PAD_CM = 0.0       # ajuste fino vertical do topo da imagem
COVER_TITLE_BAR_H_CM = 2.0       # altura da barra do título
COVER_TITLE_BAR_COLOR = "#1F4E79"  # cor da barra do título
COVER_BELOW_IMAGE_BG_COLOR = "#E0EFF1"  # cor do fundo abaixo da imagem (onde ficam as infos)
COVER_IMAGE_MODE = "cover"
COVER_IMAGE_CROP_ANCHOR = "top"
COVER_TITLE_BAR_Y_FROM_IMG_BOTTOM_CM = 1.7  # distância da barra a partir do "bottom" da imagem (para ficar sobre a imagem)
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
    if not os.path.exists(IMG_DIR):
        return None
    try:
        for f in os.listdir(IMG_DIR):
            fl = f.lower()
            if fl.startswith(base_name.lower()) and fl.endswith(('.png', '.jpg', '.jpeg')):
                return os.path.join(IMG_DIR, f)
    except Exception:
        pass
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
        except Exception:
            return None
    return None

# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Relatório GAP - Arthwind", layout="wide")
st.title("Medição de Gap - Relógio Comparador")

if not os.path.exists(IMG_DIR):
    st.sidebar.error(f"❌ Pasta '{IMG_DIR}' não encontrada!")
else:
    st.sidebar.success(f"📂 Pasta '{IMG_DIR}' encontrada.")

st.sidebar.header("Entrada de Dados")

@st.cache_data
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return pd.DataFrame()
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file)
        return pd.read_excel(uploaded_file)
    except Exception:
        return pd.DataFrame()

uploaded = st.sidebar.file_uploader("Upload Base (Excel/CSV)", type=["csv", "xlsx", "xls"])
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

# estado da seleção
if "turb_sel" not in st.session_state:
    st.session_state["turb_sel"] = [turbinas[0]] if turbinas else []

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

turb_sel = st.sidebar.multiselect(
    "Turbinas",
    turbinas,
    key="turb_sel",
)

df_turb = df_raw[df_raw["Turbina"].isin(turb_sel)].copy() if turb_sel else df_raw.copy()

blades = sorted(df_turb["SN_da_Pa"].dropna().unique().tolist())
blades_sel = st.sidebar.multiselect("Pás", blades, default=blades)

insps = sorted(df_turb["Inspecao"].dropna().unique().tolist())
insps_sel = st.sidebar.multiselect("Campanhas", insps, default=insps)

# ---------------------------------------------------------------------
# Lógica
# ---------------------------------------------------------------------
def classify_severity(delta_mm: float) -> str:
    if delta_mm is None or np.isnan(delta_mm) or delta_mm <= 0:
        return "SEV0"
    d = float(abs(delta_mm))
    if d <= 0.6: return "SEV1"
    if d <= 1.0: return "SEV2"
    if d <= 3.0: return "SEV3"
    if d <= 5.0: return "SEV4"
    return "SEV5"

def severity_color(sev: str) -> str:
    palette = {
        "SEV0": "#c6efce", "SEV1": "#a9d18e", "SEV2": "#ffd966",
        "SEV3": "#f4b183", "SEV4": "#ff8c00", "SEV5": "#ff0000",
    }
    return palette.get(sev, "#ffffff")

def detect_date_column(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = [str(c) for c in df.columns]
    lower_map = {c: c.lower().strip() for c in cols}
    exact_priority = [
        "data", "date", "data_coleta", "data de coleta", "inspection_date", "collection_date",
        "data_inspecao", "data inspeção", "data_inspeção"
    ]
    for c in cols:
        if lower_map[c] in exact_priority:
            return c
    for c in cols:
        lc = lower_map[c]
        if ("data" in lc or "date" in lc) and all(bad not in lc for bad in ["update", "atual", "criado", "created", "modified"]):
            return c
    return None

def severity_recommendation(sev: str) -> Tuple[str, dt.timedelta]:
    recs = {
        "SEV0": ("12 Months", dt.timedelta(days=365)),
        "SEV1": ("6 Months", dt.timedelta(days=182)),
        "SEV2": ("3 Months", dt.timedelta(days=91)),
        "SEV3": ("1 Month", dt.timedelta(days=30)),
        "SEV4": ("15 Days", dt.timedelta(days=15)),
        "SEV5": ("Stop Turbine", dt.timedelta(days=0)),
    }
    return recs.get(sev, ("Review", dt.timedelta(days=90)))

def _insp_num(x) -> float:
    s = str(x)
    m = pd.Series([s]).str.extract(r'(\d+)')[0].iloc[0]
    try:
        return float(m)
    except Exception:
        return np.nan

def pick_latest_rows(df: pd.DataFrame, group_keys: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    tmp = df.copy()
    tmp["_Data"] = pd.to_datetime(tmp.get("Data"), errors="coerce", dayfirst=True)
    tmp["_HasDate"] = tmp["_Data"].notna().astype(int)
    tmp["_InspNum"] = tmp["Inspecao"].map(_insp_num)
    tmp["_InspecaoStr"] = tmp["Inspecao"].astype(str)
    tmp = tmp.sort_values(group_keys + ["_HasDate", "_Data", "_InspNum", "_InspecaoStr"])
    latest = tmp.groupby(group_keys, as_index=False).tail(1)
    return latest.drop(columns=["_Data", "_HasDate", "_InspNum", "_InspecaoStr"], errors="ignore")

def trim_and_rebase(g: pd.DataFrame, xcol="Ponto", ycol="Valor_mm", eps=1e-12) -> Optional[pd.DataFrame]:
    """
    Remove trechos iniciais/finais sem leitura (NaN/0) e rebasa o eixo X para iniciar em 0.
    - Interpola somente dentro da área com dados (limit_area="inside") para NÃO criar rampa no começo/fim.
    - Corta início no primeiro ponto válido e final no último ponto válido.
    """
    if g is None or g.empty:
        return None
    g = g.sort_values(xcol).copy()

    y = pd.to_numeric(g[ycol], errors="coerce")
    y2 = y.interpolate(limit_area="inside")  # não preenche fora do intervalo válido

    valid = y2.notna() & (y2.abs() > eps)
    if not valid.any():
        return None

    start_pos = int(np.argmax(valid.values))  # primeiro True
    end_pos = int(len(valid) - 1 - np.argmax(valid.values[::-1]))  # último True

    g = g.iloc[start_pos:end_pos + 1].copy()
    g[ycol] = y2.iloc[start_pos:end_pos + 1].values

    # rebasa X
    g[xcol] = pd.to_numeric(g[xcol], errors="coerce") - float(pd.to_numeric(g[xcol], errors="coerce").iloc[0])
    return g

def process_data_core(df_target: pd.DataFrame) -> pd.DataFrame:
    frames = []
    date_col = detect_date_column(df_target)

    for col, dial in zip(reading_cols, dial_names):
        base_cols = ["Turbina", "SN_da_Pa", "Casca", "Inspecao"]
        if date_col is not None and date_col in df_target.columns:
            base_cols.append(date_col)

        base = df_target[base_cols].copy()
        if date_col is not None and date_col in base.columns:
            base["Data"] = pd.to_datetime(base[date_col], errors="coerce", dayfirst=True)
            base.drop(columns=[date_col], inplace=True, errors="ignore")
        elif "Data" in df_target.columns:
            base["Data"] = pd.to_datetime(df_target["Data"], errors="coerce", dayfirst=True)
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
            if s.notna().sum() < 3:
                return g
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
    if s.notna().sum() < 10:
        return float("nan"), 0
    y = s.values.astype(float)
    global_amp = np.nanmax(y) - np.nanmin(y)
    if global_amp < 0.05:
        return 0.0, 0
    win = max(5, int(len(y) * 0.02)) | 1
    smooth = pd.Series(y).rolling(window=win, center=True, min_periods=1).mean().values
    peaks, _ = find_peaks(smooth, prominence=global_amp * 0.1)
    valleys, _ = find_peaks(-smooth, prominence=global_amp * 0.1)
    if len(peaks) == 0 or len(valleys) == 0:
        return global_amp, 1
    amplitudes = []
    turning_points = np.concatenate([peaks, valleys])
    turning_points.sort()
    for i in range(len(turning_points) - 1):
        idx1, idx2 = turning_points[i], turning_points[i + 1]
        amp = abs(smooth[idx1] - smooth[idx2])
        if amp > 0.25 * global_amp:
            amplitudes.append(amp)
    if not amplitudes:
        return global_amp, 1
    return float(np.mean(amplitudes)), max(1, int(len(amplitudes) // 2))

def run_analysis(df_in: pd.DataFrame, full_process=False):
    if not full_process:
        df_subset = df_in[
            (df_in["Turbina"].isin(turb_sel)) &
            (df_in["SN_da_Pa"].isin(blades_sel)) &
            (df_in["Inspecao"].isin(insps_sel))
        ].copy()
    else:
        df_subset = df_in.copy()

    if df_subset.empty:
        return None

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

    if full_process:
        return {"delta_summary": delta_summary}

    # Datas por campanha (mais recente)
    camp_dates = (
        delta_summary.groupby("Inspecao", as_index=False)["Data"]
        .max()
        .dropna(subset=["Data"])
    )
    camp_dates = camp_dates.sort_values("Inspecao", key=lambda s: s.map(_insp_num).fillna(1e9))
    campaign_dates_str = [
        f"{row['Inspecao']}: {pd.to_datetime(row['Data']).strftime('%d-%m-%Y')}"
        for _, row in camp_dates.iterrows()
    ]
    last_overall = camp_dates["Data"].max() if not camp_dates.empty else pd.NaT
    cover_date = pd.to_datetime(last_overall).strftime("%d-%m-%Y") if pd.notna(last_overall) else dt.datetime.now().strftime("%d-%m-%Y")

    # PDF data
    pdf_detailed_data = []
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for blade in blades_sel:
        df_b = long_df[long_df["Blade"] == blade]
        if df_b.empty:
            continue
        sensors_data_list = []
        for (casca, reg, relogio), g_sensor in df_b.groupby(["Casca", "Regiao", "Relogio"]):
            stats_rows = []
            fig, ax = plt.subplots(figsize=(6, 3.2))
            has_data = False
            idx_color = 0

            for (tb, isp), g_trace in g_sensor.groupby(["Turbina", "Inspecao"]):
                g_trim = trim_and_rebase(g_trace, xcol="Ponto", ycol="Valor_mm")
                if g_trim is None or g_trim.empty:
                    continue
                lbl = f"{isp}" if len(turb_sel) == 1 else f"{tb}-{isp}"
                c = colors_list[idx_color % len(colors_list)]
                ax.plot(g_trim["Ponto"], g_trim["Valor_mm"], label=lbl, linewidth=1.5, color=c)

                val_d = delta_summary[
                    (delta_summary["Turbina"] == tb) &
                    (delta_summary["Blade"] == blade) &
                    (delta_summary["Casca"] == casca) &
                    (delta_summary["Regiao"] == reg) &
                    (delta_summary["Relogio"] == relogio) &
                    (delta_summary["Inspecao"] == isp)
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

    # Severidade geral
    grouped_all = delta_summary.groupby(["Turbina", "Blade"], as_index=False)["Delta_medio_ciclo_mm"].max()
    grouped_all.rename(columns={"Delta_medio_ciclo_mm": "Delta_max_mm"}, inplace=True)
    grouped_all["Severity"] = grouped_all["Delta_max_mm"].apply(classify_severity)
    grouped_all["Color"] = grouped_all["Severity"].apply(severity_color)

    # Severidade última coleta
    latest_sensors = pick_latest_rows(delta_summary, ["Turbina", "Blade", "Casca", "Regiao", "Relogio"])
    blade_latest = (
        latest_sensors.groupby(["Turbina", "Blade"], as_index=False)
        .agg(Delta_latest_max_mm=("Delta_medio_ciclo_mm", "max"), Last_Date=("Data", "max"))
    )
    blade_latest["Severity"] = blade_latest["Delta_latest_max_mm"].apply(classify_severity)

    rec_txt, next_dates = [], []
    for sev, d_last in zip(blade_latest["Severity"], blade_latest["Last_Date"]):
        r_txt, delta_t = severity_recommendation(sev)
        rec_txt.append(r_txt)
        next_dates.append(pd.to_datetime(d_last) + delta_t if pd.notna(d_last) else pd.NaT)
    blade_latest["Recommendation"] = rec_txt
    blade_latest["Next_Inspection"] = next_dates
    blade_latest["Color"] = blade_latest["Severity"].apply(severity_color)

    turbine_latest = (
        blade_latest.groupby(["Turbina"], as_index=False)
        .agg(Delta_latest_max_mm=("Delta_latest_max_mm", "max"), Last_Date=("Last_Date", "max"))
    )
    turbine_latest["Severity"] = turbine_latest["Delta_latest_max_mm"].apply(classify_severity)
    turbine_latest["Recommendation"] = turbine_latest["Severity"].apply(lambda s: severity_recommendation(s)[0])
    turbine_latest["Color"] = turbine_latest["Severity"].apply(severity_color)

    # Reinspeções
    reinspection = (
        delta_summary.groupby(["Turbina", "Blade"], as_index=False)
        .agg(Reinspections=("Inspecao", "nunique"), First_Date=("Data", "min"), Last_Date=("Data", "max"))
    )
    reinspection = reinspection.merge(
        blade_latest[["Turbina", "Blade", "Delta_latest_max_mm", "Severity", "Recommendation", "Next_Inspection"]],
        on=["Turbina", "Blade"], how="left"
    )

    critical_blades = blade_latest.sort_values("Delta_latest_max_mm", ascending=False).head(15).copy()
    critical_turbines = turbine_latest.sort_values("Delta_latest_max_mm", ascending=False).head(15).copy()

    return {
        "meta": {
            "turb": ", ".join(map(str, turb_sel)) if turb_sel else "All",
            "blades": blades_sel,
            "insps": insps_sel,
            "date": cover_date,
            "campaign_dates": campaign_dates_str
        },
        "delta_summary": delta_summary,
        "severity_by_blade": grouped_all,
        "severity_by_blade_latest": blade_latest,
        "severity_by_turbine_latest": turbine_latest,
        "reinspection_table": reinspection,
        "critical_blades": critical_blades,
        "critical_turbines": critical_turbines,
        "pdf_detailed_data": pdf_detailed_data,
        "verify_data": verify_items
    }

# ---------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------
def _draw_wrapped(canvas, text: str, x: float, y: float, max_chars: int, line_h_pt: float):
    """Quebra simples por número de caracteres (pra cover Campaign Dates)."""
    if not text:
        return
    lines = textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)
    for i, line in enumerate(lines):
        canvas.drawString(x, y - i * line_h_pt, line)


def draw_image_cover(canvas, img_reader, x: float, y: float, w: float, h: float):
    """Desenha imagem em modo 'cover': preenche totalmente o retângulo (pode cortar bordas)."""
    try:
        iw, ih = img_reader.getSize()
        if iw <= 0 or ih <= 0:
            return
        scale = max(w / float(iw), h / float(ih))
        sw = float(iw) * scale
        sh = float(ih) * scale
        dx = x - (sw - w) / 2.0
        dy = y - (sh - h) / 2.0

        p = canvas.beginPath()
        p.rect(x, y, w, h)
        canvas.saveState()
        canvas.clipPath(p, stroke=0, fill=0)
        canvas.drawImage(img_reader, dx, dy, width=sw, height=sh, mask='auto')
        canvas.restoreState()
    except Exception:
        pass

def draw_image_contain(canvas, img_reader, x: float, y: float, w: float, h: float):
    """Desenha imagem em modo 'contain': mantém proporção, pode sobrar bordas brancas."""
    try:
        canvas.drawImage(img_reader, x=x, y=y, width=w, height=h, preserveAspectRatio=True, anchor='n')
    except Exception:
        pass


def generate_pdf(results: Dict[str, Any]):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=2.5 * cm, bottomMargin=2 * cm, leftMargin=1.5 * cm, rightMargin=1.5 * cm
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=14, textColor=colors.HexColor("#1F4E79"), spaceAfter=10)
    normal = ParagraphStyle("Norm", parent=styles["Normal"], fontSize=10, leading=12, alignment=TA_JUSTIFY)

    meta = results["meta"]
    sev_df = results.get("severity_by_blade_latest", results.get("severity_by_blade"))
    pdf_data = results["pdf_detailed_data"]

    turbina_txt = meta.get("turb", "-")
    blades_list_txt = ", ".join(map(str, meta.get("blades", [])))
    camp_dates_txt = " | ".join(meta.get("campaign_dates", [])) or "-"

    def draw_cover_full(canvas, _doc):
        canvas.saveState()
        page_w, page_h = A4

        strip_w = COVER_LEFT_STRIP_W_CM * cm

        # fundo branco (faixa esquerda é o próprio branco)
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, page_w, page_h, stroke=0, fill=1)

        # fundo abaixo da imagem (onde ficam as infos)
        canvas.setFillColor(colors.HexColor(COVER_BELOW_IMAGE_BG_COLOR))
        canvas.rect(0, 0, page_w, page_h, stroke=0, fill=1)

        # imagem da capa (full width)
        cover_bytes = load_image_for_pdf("COVER")
        img_h = page_h * COVER_IMG_H_RATIO
        y = page_h - img_h - (COVER_IMG_TOP_PAD_CM * cm)

        if cover_bytes:
            try:
                img_reader = ImageReader(cover_bytes)
                if str(COVER_IMAGE_MODE).lower().strip() == "contain":
                    draw_image_contain(canvas, img_reader, 0, y, page_w, img_h)
                else:
                    draw_image_cover(canvas, img_reader, 0, y, page_w, img_h)
            except Exception:
                pass
        else:
            # fallback: retângulo azul claro
            canvas.setFillColor(colors.HexColor("#dbeef3"))
            canvas.rect(0, y, page_w, img_h, stroke=0, fill=1)

        # logo
        logo_bytes = load_image_for_pdf("LOGO")
        if logo_bytes:
            try:
                logo_reader = ImageReader(logo_bytes)
                canvas.drawImage(
                    logo_reader,
                    COVER_LOGO_X_CM * cm,
                    page_h - (COVER_LOGO_Y_FROM_TOP_CM * cm),
                    width=COVER_LOGO_W_CM * cm,
                    height=COVER_LOGO_H_CM * cm,
                    mask='auto',
                    preserveAspectRatio=True
                )
            except Exception:
                pass

        # barra azul do título (sobre a imagem)
        img_bottom_y = page_h - (page_h * COVER_IMG_H_RATIO) - (COVER_IMG_TOP_PAD_CM * cm)
        bar_h = COVER_TITLE_BAR_H_CM * cm
        bar_y = img_bottom_y + (COVER_TITLE_BAR_Y_FROM_IMG_BOTTOM_CM * cm)
        bar_x = _doc.leftMargin
        bar_w = page_w - _doc.leftMargin - _doc.rightMargin

        canvas.setFillColor(colors.HexColor(COVER_TITLE_BAR_COLOR))
        canvas.rect(bar_x, bar_y, bar_w, bar_h, stroke=0, fill=1)

        # título em 2 linhas
        canvas.setFillColor(colors.white)
        canvas.setFont("Helvetica-Bold", 18)
        canvas.drawCentredString(page_w / 2, bar_y + bar_h * 0.62, "ROOT GAP MEASUREMENT INSPECTION")
        canvas.drawCentredString(page_w / 2, bar_y + bar_h * 0.22, "REPORT")

        # metadados (abaixo)
        label_x = COVER_META_LABEL_X_CM * cm
        value_x = COVER_META_VALUE_X_CM * cm
        y0 = COVER_META_START_Y_FROM_BOTTOM_CM * cm
        line_h = COVER_META_LINE_H_CM * cm

        labels = ["Wtg Serial Number:", "Windfarm:", "Campaign Dates:", "Date (Last):", "Blade Model:", "Customer:"]
        values = [
            turbina_txt,
            "COMPLEXO EÓLICO SERRA AZUL",
            camp_dates_txt,
            meta.get("date", "-"),
            "LM47.6",
            "ENEL"
        ]

        canvas.setFont("Helvetica-Bold", COVER_META_LABEL_SIZE)
        canvas.setFillColor(colors.HexColor("#1F4E79"))
        for i, lab in enumerate(labels):
            canvas.drawString(label_x, y0 - i * line_h, lab)

        canvas.setFont("Helvetica", COVER_META_VALUE_SIZE)
        canvas.setFillColor(colors.black)
        for i, val in enumerate(values):
            yy = y0 - i * line_h
            if i == 2:  # campaign dates (quebra linha)
                _draw_wrapped(canvas, str(val), value_x, yy, max_chars=65, line_h_pt=10)
            else:
                canvas.drawString(value_x, yy, str(val))

        canvas.restoreState()

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
            except Exception:
                pass
        canvas.restoreState()

    # STORY — página 1 é só capa (desenhada no canvas), então começamos já com PageBreak
    story = [PageBreak()]

    # SUMMARY
    story.append(Paragraph("1. Summary", h1))
    t_toc = Table([["Section", "Page"], ["2. Introduction", "3"], ["3. Conclusion", "3"], ["4. Methodology", "4"], ["5. Scope", "5"], ["6. Damages", "5"], ["7. Evidence", "6+"]], colWidths=[14 * cm, 2 * cm])
    t_toc.setStyle(TableStyle([('LINEBELOW', (0, 0), (-1, -1), 0.5, colors.lightgrey)]))
    story.append(t_toc)
    story.append(PageBreak())

    # INTRO
    story.append(Paragraph("2. Introduction", h1))
    intro_txt = f"On {meta.get('date','-')}, a gap measurement inspection was performed on LM47.6 model blades, serial numbers {blades_list_txt}, installed on the {turbina_txt} wind turbine located at the COMPLEXO EÓLICO SERRA AZUL."
    story.append(Paragraph(intro_txt, normal))
    story.append(Spacer(1, 1 * cm))

    # CONCLUSION
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
        t_conc.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        story.append(t_conc)
        recs = ["12 Months", "6 Months", "3 Months", "1 Month", "15 Days", "Stop Turbine"]
        story.append(Spacer(1, 0.5 * cm))
        story.append(Paragraph(f"We recommend a new inspection within {recs[worst_sev_idx]}", normal))

    story.append(PageBreak())

    # METHODOLOGY (igual)
    story.append(Paragraph("4. Methodology", h1))
    story.append(Paragraph("The operation developed by Arthwind consists on inspecting the root gap of wind turbine blades using dial indicators, with the equipment installed externally around the blade root.", normal))
    m1_bytes = load_image_for_pdf("METOD_ROTOR")
    if m1_bytes:
        story.append(Table([[Image(m1_bytes, width=8 * cm, height=5.5 * cm)]], colWidths=[A4[0] - doc.leftMargin - doc.rightMargin], style=[('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("Dial indicators are mounted at specific points around the blade root using suction bases and rods. The blade is rotated 360°while the dial indicators remain in position.", normal))
    m2_bytes = load_image_for_pdf("METOD_MAPA")
    if m2_bytes:
        story.append(Table([[Image(m2_bytes, width=12 * cm, height=10 * cm)]], colWidths=[A4[0] - doc.leftMargin - doc.rightMargin], style=[('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("After the rotation, the maximum and minimum displacement values are analyzed to check the total variation at each point. This procedure is repeated for all blades on the turbine.", normal))
    m3_bytes = load_image_for_pdf("METOD_BASE")
    if m3_bytes:
        story.append(Table([[Image(m3_bytes, width=14 * cm, height=14 * cm)]], colWidths=[A4[0] - doc.leftMargin - doc.rightMargin], style=[('ALIGN', (0, 0), (-1, -1), 'CENTER')]))
    story.append(PageBreak())

    # DAMAGES
    story.append(Paragraph("6. Damages categorization", h1))
    cat_data = [
        ["Severity", "Description", "Recommendation"],
        ["SEV0", "No gaps detected", "12 Months"],
        ["SEV1", "Gap up to 0,6mm", "6 Months"],
        ["SEV2", "Gap between 0,6 and 1mm", "3 Months"],
        ["SEV3", "Gap between 1 and 3mm", "1 Month"],
        ["SEV4", "Gap between 3 and 5mm", "15 Days"],
        ["SEV5", "Gap higher than 5mm", "Stop Turbine"],
    ]
    t_cat = Table(cat_data, colWidths=[3 * cm, 7 * cm, 4 * cm])
    t_cat.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#c6efce")),
        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#a9d18e")),
        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#ffd966")),
        ('BACKGROUND', (0, 4), (-1, 4), colors.HexColor("#f4b183")),
        ('BACKGROUND', (0, 5), (-1, 5), colors.HexColor("#ff8c00")),
        ('BACKGROUND', (0, 6), (-1, 6), colors.HexColor("#ff0000")),
    ]))
    story.append(t_cat)
    story.append(PageBreak())

    # EVIDENCE (lado a lado)
    story.append(Paragraph("7. Inspection Evidence", h1))
    usable_w = A4[0] - doc.leftMargin - doc.rightMargin

    for (blade, sensors_data) in pdf_data:
        story.append(Paragraph(f"BLADE {blade}", h1))
        for item in sensors_data:
            fig = item["fig"]
            stats_rows = item["stats"]

            img_io = io.BytesIO()
            fig.savefig(img_io, format='png', dpi=100, bbox_inches='tight')
            img_io.seek(0)

            img_w = usable_w * 0.70
            tbl_w = usable_w - img_w
            rl_img = Image(img_io, width=img_w, height=6.5 * cm)

            table_data = [["Campanha", "Gap(mm)"]]
            for stat in stats_rows:
                gap_val = stat["Gap (mm)"]
                gap_str = f"{gap_val:.1f}".replace(".", ",") if pd.notna(gap_val) else "-"
                table_data.append([stat["Campanha"], gap_str])

            t_stats = Table(table_data, colWidths=[tbl_w * 0.65, tbl_w * 0.35])
            t_stats.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e0e0e0")),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
            ]))

            t_master = Table([[rl_img, t_stats]], colWidths=[img_w, tbl_w])
            t_master.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ]))

            story.append(KeepTogether(t_master))
            story.append(Spacer(1, 0.5 * cm))
        story.append(PageBreak())

    doc.build(story, onFirstPage=draw_cover_full, onLaterPages=draw_header_footer)
    buffer.seek(0)
    return buffer.getvalue()

# ---------------------------------------------------------------------
# Excel (mantido)
# ---------------------------------------------------------------------
def generate_excel_report(delta_summary: pd.DataFrame):
    if delta_summary is None or delta_summary.empty:
        return None
    df_pivot = delta_summary.copy()
    df_pivot['Sensor_Key'] = df_pivot['Casca'].astype(str) + '-' + df_pivot['Regiao'].astype(str)
    pivot = df_pivot.pivot_table(index=['Turbina', 'Blade', 'Inspecao'], columns='Sensor_Key', values='Delta_medio_ciclo_mm')
    pivot['Media'] = pivot.mean(axis=1)
    pivot['Maximo'] = pivot.max(axis=1)
    pivot['Severidade'] = pivot['Maximo'].apply(classify_severity)
    pivot = pivot.reset_index()

    hex_colors = {"SEV0": "#c6efce", "SEV1": "#a9d18e", "SEV2": "#ffd966", "SEV3": "#f4b183", "SEV4": "#ff8c00", "SEV5": "#ff0000"}
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        pivot.to_excel(writer, sheet_name='Base_Consolidada', index=False)
        wb = writer.book
        ws = writer.sheets['Base_Consolidada']
        fmt_border = wb.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})
        fmt_head = wb.add_format({'bold': True, 'align': 'center', 'border': 1, 'bg_color': '#D3D3D3'})
        sev_fmts = {k: wb.add_format({'bg_color': v, 'border': 1, 'align': 'center'}) for k, v in hex_colors.items()}
        (max_r, max_c) = pivot.shape
        for c, val in enumerate(pivot.columns.values):
            ws.write(0, c, val, fmt_head)
        col_sev = pivot.columns.get_loc('Severidade')
        # Substitua o bloco de escrita atual por este:
        for r in range(max_r):
            sev_val = pivot.iloc[r, col_sev]
            for c in range(max_c):
                val = pivot.iloc[r, c]
        
            if c == col_sev:
                ws.write(r + 1, c, val, sev_fmts.get(sev_val, fmt_border))
            else:
            # Tratamento de segurança: tenta converter, se falhar, escreve como string
                try:
                # Se for NaN, escreve vazio
                    if pd.isna(val):
                        ws.write(r + 1, c, "", fmt_border)
                    else:
                    # Tenta converter para float, se for string numérica
                        ws.write(r + 1, c, float(val), fmt_border)
                except (ValueError, TypeError):
                # Se for um texto que não vira número, escreve o próprio texto
                    ws.write(r + 1, c, str(val), fmt_border)
    output.seek(0)
    return output.getvalue()

# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
if st.button("Calcular Análise"):
    with st.spinner("Calculando Visualização..."):
        results = run_analysis(df_raw, full_process=False)
        st.session_state["results"] = results

if "results" in st.session_state and st.session_state["results"] is not None:
    results = st.session_state["results"]

    st.markdown("---")
    st.subheader("📌 Dashboard Executivo (Baseado na Última Coleta)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Pás mais críticas (Top 15)**")
        cb = results.get("critical_blades", pd.DataFrame())
        if not cb.empty:
            cb_show = cb.copy()
            cb_show["Delta_latest_max_mm"] = cb_show["Delta_latest_max_mm"].round(2)
            if "Last_Date" in cb_show.columns:
                cb_show["Last_Date"] = pd.to_datetime(cb_show["Last_Date"], errors="coerce").dt.strftime("%d-%m-%Y")
            if "Next_Inspection" in cb_show.columns:
                cb_show["Next_Inspection"] = pd.to_datetime(cb_show["Next_Inspection"], errors="coerce").dt.strftime("%d-%m-%Y")
            st.dataframe(cb_show, use_container_width=True, key="df_critical_blades")
        else:
            st.info("Sem dados suficientes para ranking de pás.")

    with c2:
        st.markdown("**Turbinas mais críticas (Top 15)**")
        ct = results.get("critical_turbines", pd.DataFrame())
        if not ct.empty:
            ct_show = ct.copy()
            ct_show["Delta_latest_max_mm"] = ct_show["Delta_latest_max_mm"].round(2)
            if "Last_Date" in ct_show.columns:
                ct_show["Last_Date"] = pd.to_datetime(ct_show["Last_Date"], errors="coerce").dt.strftime("%d-%m-%Y")
            st.dataframe(ct_show, use_container_width=True, key="df_critical_turbines")
        else:
            st.info("Sem dados suficientes para ranking de turbinas.")

    st.markdown("### 📈 Distribuição por Severidade")
    sev_order = ["SEV0", "SEV1", "SEV2", "SEV3", "SEV4", "SEV5"]

    g1, g2 = st.columns(2)
    with g1:
        bl = results.get("severity_by_blade_latest", pd.DataFrame())
        if not bl.empty and "Severity" in bl.columns:
            counts = bl["Severity"].value_counts().reindex(sev_order, fill_value=0).reset_index()
            counts.columns = ["Severity", "Count"]
            figb = px.bar(counts, x="Severity", y="Count", title="Pás por Severidade")
            figb.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(figb, use_container_width=True, key="bar_blades_sev")

    with g2:
        tl = results.get("severity_by_turbine_latest", pd.DataFrame())
        if not tl.empty and "Severity" in tl.columns:
            counts = tl["Severity"].value_counts().reindex(sev_order, fill_value=0).reset_index()
            counts.columns = ["Severity", "Count"]
            figt = px.bar(counts, x="Severity", y="Count", title="Turbinas por Severidade")
            figt.update_layout(margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(figt, use_container_width=True, key="bar_turbines_sev")

    st.markdown("### 🔁 Reinspeções com Relógio (por Turbina e Pá)")
    rt = results.get("reinspection_table", pd.DataFrame())
    if not rt.empty:
        rt_show = rt.copy()
        for c in ["First_Date", "Last_Date", "Next_Inspection"]:
            if c in rt_show.columns:
                rt_show[c] = pd.to_datetime(rt_show[c], errors="coerce").dt.strftime("%d-%m-%Y")
        if "Delta_latest_max_mm" in rt_show.columns:
            rt_show["Delta_latest_max_mm"] = rt_show["Delta_latest_max_mm"].round(2)
        st.dataframe(rt_show, use_container_width=True, key="df_reinspection")
    else:
        st.info("Sem dados suficientes para consolidar reinspeções.")

    st.markdown("### 🗓️ Datas das Campanhas (mais recente por campanha)")
    camp_dates = results.get("meta", {}).get("campaign_dates", [])
    st.write(" | ".join(camp_dates) if camp_dates else "Não foi possível detectar datas na base.")

    st.markdown("---")
    st.subheader("📊 Resumo de Severidade (Campanha Mais Recente)")
    sev_df = results.get("severity_by_blade_latest", results["severity_by_blade"])
    st.dataframe(sev_df, use_container_width=True, key="df_sev_summary")

    st.markdown("---")
    st.subheader("🔍 Detalhe por Sensor (Comparativo)")

    verify_data = results["verify_data"]
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
            sensors = sorted(g_blade["Sensor"].unique())
            for sens in sensors:
                g_sens = g_blade[g_blade["Sensor"] == sens].copy()
                c_graph, c_table = st.columns([3, 1])

                plots = []
                for leg, g_leg in g_sens.groupby("Legenda"):
                    g_leg = g_leg.copy()
                    g_leg["Valor_mm"] = pd.to_numeric(g_leg["Valor_mm"], errors="coerce")
                    g_trim = trim_and_rebase(g_leg, xcol="Ponto", ycol="Valor_mm")
                    if g_trim is None or g_trim.empty:
                        continue
                    g_trim["Legenda"] = leg
                    plots.append(g_trim)

                with c_graph:
                    chart_counter += 1
                    if plots:
                        g_plot = pd.concat(plots, ignore_index=True)
                        fig = px.line(g_plot, x="Ponto", y="Valor_mm", color="Legenda",
                                      title=f"Sensor: {sens}", height=350)
                        fig.update_traces(connectgaps=True)
                        fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True, key=f"curve_{blade}_{sens}_{chart_counter}")
                    else:
                        st.info("Sem dados válidos para plotar esse sensor.")

                with c_table:
                    stats = g_sens.groupby("Legenda")["Delta_Calc_mm"].mean().reset_index()
                    stats.rename(columns={"Legenda": "Campanha", "Delta_Calc_mm": "Gap (mm)"}, inplace=True)
                    st.dataframe(stats.style.format({"Gap (mm)": "{:.1f}"}), hide_index=True, use_container_width=True, key=f"tbl_{blade}_{sens}_{chart_counter}")

                st.divider()

    with st.spinner("Gerando PDF..."):
        pdf_bytes = generate_pdf(results)
    st.download_button("📥 Baixar PDF (Lado a Lado)", data=pdf_bytes, file_name="Relatorio.pdf", mime="application/pdf")
    # -----------------------------------------------------------------
# EXPORTAÇÃO EM LOTE: PDFs de TODAS as turbinas em um ZIP (com progresso)
# -----------------------------------------------------------------
st.markdown("### 📦 Exportação em Lote (ZIP)")
st.info("Gera um PDF por turbina (incluindo todas as pás e todas as campanhas disponíveis na base), e entrega um arquivo .zip. Pode demorar.")

export_all = st.checkbox("Gerar para TODAS as turbinas da base", value=True)
only_selected = st.checkbox("Gerar apenas para as turbinas selecionadas no filtro", value=False)

if st.button("🚀 Gerar ZIP com PDFs"):
    # Decide a lista de turbinas
    if only_selected and len(turb_sel) > 0:
        turbs_to_run = list(turb_sel)
    elif export_all:
        turbs_to_run = sorted(df_raw["Turbina"].dropna().unique().tolist())
    else:
        turbs_to_run = list(turb_sel) if len(turb_sel) > 0 else sorted(df_raw["Turbina"].dropna().unique().tolist())

    if not turbs_to_run:
        st.error("Não há turbinas para processar.")
    else:
        # backup das seleções atuais (para restaurar depois)
        _orig_turb_sel = list(turb_sel)
        _orig_blades_sel = list(blades_sel) if isinstance(blades_sel, list) else blades_sel
        _orig_insps_sel = list(insps_sel) if isinstance(insps_sel, list) else insps_sel

        zip_buffer = io.BytesIO()
        errors = []

        prog = st.progress(0)
        status = st.empty()

        with zipfile.ZipFile(zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            total = len(turbs_to_run)
            for i, tb in enumerate(turbs_to_run, start=1):
                status.markdown(f"**Gerando {i}/{total}** — Turbina: `{tb}`")

                # Define seleção local (1 turbina + todas as pás + todas as campanhas daquela turbina)
                turb_sel = [tb]
                df_tb = df_raw[df_raw["Turbina"] == tb].copy()

                blades_sel = sorted(df_tb["SN_da_Pa"].dropna().unique().tolist())
                insps_sel = sorted(df_tb["Inspecao"].dropna().unique().tolist())

                # Se faltar dado mínimo, pula
                if not blades_sel or not insps_sel:
                    errors.append((tb, "Sem blades/inspeções suficientes nesta turbina."))
                    prog.progress(i / total)
                    continue

                try:
                    res_tb = run_analysis(df_raw, full_process=False)
                    pdf_tb = generate_pdf(res_tb)

                    safe_name = str(tb).replace("/", "-").replace("\\", "-").strip()
                    zf.writestr(f"ATW-{safe_name}-GAP.pdf", pdf_tb)

                except Exception as e:
                    errors.append((tb, str(e)))

                prog.progress(i / total)

        # Restaura seleções do usuário
        turb_sel = _orig_turb_sel
        blades_sel = _orig_blades_sel
        insps_sel = _orig_insps_sel

        status.empty()
        prog.progress(1.0)

        zip_buffer.seek(0)
        st.download_button(
            "📥 Baixar ZIP com PDFs",
            data=zip_buffer.getvalue(),
            file_name="Relatorios_GAP.zip",
            mime="application/zip",
        )

        if errors:
            st.warning("Algumas turbinas falharam e foram puladas:")
            st.dataframe(pd.DataFrame(errors, columns=["Turbina", "Erro"]), use_container_width=True)


st.markdown("### Exportação Global")
st.info("O botão abaixo processa TODAS as turbinas/pás da base com os filtros atuais.")

# Inicializa o estado se não existir
if "excel_bytes" not in st.session_state:
    st.session_state["excel_bytes"] = None

# Botão de processamento
if st.button("🚀 Gerar Base Consolidada (Excel)"):
    with st.spinner("Processando base completa (pode demorar)..."):
        global_res = run_analysis(df_raw, full_process=True)
        # Armazena os bytes no session_state para persistir
        st.session_state["excel_bytes"] = generate_excel_report(global_res["delta_summary"])

# Renderiza o botão de download apenas se os dados existirem no estado
if st.session_state["excel_bytes"] is not None:
    st.download_button(
        "📥 Download Excel Completo", 
        data=st.session_state["excel_bytes"], 
        file_name="Base_Consolidada.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
