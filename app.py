"""Dashboard Streamlit - Línea Inbound Tuboleta (agente virtual + agentes humanos).

Ejecutar:  streamlit run app.py
"""
import base64
import itertools
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from generar_dashboard import TIP

BASE_DIR = Path(__file__).resolve().parent
CSV = BASE_DIR / "data" / "base_llamadas_prueba.csv"
CSV_AG = BASE_DIR / "data" / "base_agentes_prueba.csv"
BANNER = BASE_DIR / "banner_tuboleta.png"
LOGO = BASE_DIR / "Logo.png"

AZUL, AZUL_OSC, NAVY = "#0162BA", "#01407A", "#0B2B55"
AMARILLO, NARANJA, CELESTE, CIELO = "#F1D11A", "#F1A01A", "#A9CDF0", "#5BA3E6"
VERDE, ROJO, AMBAR, GRIS = "#1E9E5A", "#D64545", "#E8A317", "#9AA8B8"
PALETA = [AZUL, NARANJA, NAVY, AMARILLO, CIELO, "#7F8FA6", "#C27C0E", "#3D7FCB"]
VIRTUAL, HUMANO = "Agente Virtual", "Agente Humano"
METAS = {"ns": .80, "asa": 20, "tmoH": 300, "abp": .05, "occ": (.80, .90), "adh": .90, "cr": .95,
         "tmoV": 120, "cont": round(1 - TIP.p_hum.sum(), 3)}
# meta IA % por tipificación (tabla de negocio) y ponderada por skill
META_IA = {**dict(zip(TIP.tipificacion, TIP.ia)),
           **(1 - TIP.groupby("skill").p_hum.sum() / TIP.groupby("skill").p.sum()).to_dict()}
DIR = {"ns": 1, "adh": 1, "cr": 1, "cont": 1, "abp": -1, "asa": -1, "tmoH": -1, "tmoV": -1, "occ": 0}
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
DIAS_L = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
FLAGS = ["ofr", "ate", "aba", "abc", "atu", "h_ofr", "h_ate", "v_ate", "esp_ate", "tmo_h", "tmo_v",
         "tiempo_conversacion_s", "tiempo_hold_s", "tiempo_acw_s"]
AFLAGS = ["seg_conversacion", "seg_hold", "seg_acw", "seg_disponible", "min_en_estado_planificado", "min_planificados"]

st.set_page_config(page_title="Tuboleta | Dashboard Línea Inbound", page_icon=str(LOGO), layout="wide")

st.markdown(f"""
<style>
.stApp {{background:#F3F5F8}}
[data-testid="stHeader"], [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{display:none}}
.block-container {{padding-top:1rem; padding-bottom:1rem; max-width:1640px}}
/* encabezado */
.hdr {{display:flex; align-items:center; gap:18px; height:100%}}
.hdr img {{height:62px; border-radius:10px}}
.hdr .sep {{width:3px; height:58px; background:{AZUL}; border-radius:2px}}
.hdr h1 {{margin:0; padding:0; font-size:24px; font-weight:800; color:{NAVY}; line-height:1.15; white-space:nowrap}}
.hdr h1 span {{color:{AZUL}}}
.hdr p {{margin:2px 0 0; font-size:15px; font-weight:600; color:#5B6B80}}
.st-key-filtros {{background:#fff; border:1px solid #DDE3EA; border-radius:10px; padding:8px 14px 10px; margin:10px 0 4px}}
.ft {{display:flex; align-items:center; gap:8px; background:{NAVY}; color:#fff; font-weight:700; font-size:14px;
        border-radius:8px; padding:9px 12px; margin-bottom:2px}}
[data-testid="stWidgetLabel"] p {{font-size:11px !important; color:#6B7A90 !important; font-weight:600}}
[data-baseweb="select"] > div {{background:#fff; border-color:#D5DCE5; border-radius:8px; min-height:38px}}
[data-baseweb="tag"] {{background:{AZUL} !important}}
[data-baseweb="tag"] span {{color:#fff !important}}
/* pestañas */
.stTabs [data-baseweb="tab-list"], .stTabs [role="tablist"] {{gap:6px; border-bottom:none}}
.stTabs [data-baseweb="tab"], [data-testid="stTab"] {{background:#fff !important; border:1px solid #DDE3EA !important;
        border-radius:8px !important; padding:6px 16px !important; color:{NAVY} !important; height:auto}}
[data-testid="stTab"] p {{font-weight:700; font-size:13px}}
.stTabs [aria-selected="true"] {{background:{NAVY} !important; border-color:{NAVY} !important}}
.stTabs [aria-selected="true"] p {{color:#fff !important}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{display:none}}
/* tarjetas KPI */
.kgrid {{display:grid; grid-template-columns:repeat(10,1fr); gap:8px; margin:4px 0 14px}}
@media (max-width:1300px) {{.kgrid {{grid-template-columns:repeat(5,1fr)}}}}
.kc {{background:#fff; border:1px solid #DDE3EA; border-radius:10px; padding:9px 10px 8px; text-align:center}}
.kc .top {{display:flex; align-items:center; justify-content:center; gap:6px; font-size:12px; font-weight:700;
        color:{NAVY}; white-space:nowrap; overflow:hidden}}
.kc .v {{font-size:25px; font-weight:800; line-height:1.2; margin-top:4px}}
.kc .s {{font-size:10.5px; color:#7A8799; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}}
/* paneles */
[class*="st-key-pnl_"] {{background:#fff; border:1px solid #DDE3EA; border-radius:10px; padding:0 10px 8px !important;
        overflow:hidden; gap:.35rem}}
.ph {{display:flex; justify-content:space-between; align-items:center; background:{NAVY}; color:#fff;
        margin:0 -10px 4px; padding:8px 14px; font-size:14px; font-weight:700}}
.ph .x {{font-size:11px; font-weight:600; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.35);
        border-radius:6px; padding:2px 10px}}
/* estado de agentes y cola */
.ag-top {{display:flex; justify-content:space-around; align-items:baseline; margin:6px 0 10px}}
.ag-top b {{font-size:30px; color:{NAVY}}} .ag-top span {{font-size:12px; color:#6B7A90; margin-left:6px}}
.ag-bar {{display:flex; height:46px; border-radius:6px; overflow:hidden; font-size:12px; font-weight:700}}
.ag-bar div {{display:flex; align-items:center; justify-content:center; color:#fff}}
.ag-leg {{display:grid; grid-template-columns:repeat(5,1fr); margin-top:10px; text-align:center; font-size:11px; color:#5B6B80}}
.ag-leg b {{display:block; font-size:16px; color:{NAVY}}}
.dot {{display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px}}
.ql {{display:flex; align-items:center; justify-content:space-between; padding:9px 4px; border-bottom:1px solid #EEF1F5;
        font-size:13px; color:#41536B}}
.ql:last-child {{border-bottom:none}} .ql span {{display:flex; align-items:center; gap:10px}}
.ql b {{font-size:20px; color:{AZUL}}}
/* desempeño por hora */
table.dh {{width:100%; border-collapse:separate; border-spacing:2px; font-size:12px}}
table.dh th {{color:#5B6B80; font-weight:600; padding:4px 2px; text-align:center}}
table.dh td {{text-align:center; padding:6px 2px; border-radius:3px}}
table.dh td.r {{text-align:left; font-weight:700; color:{NAVY}; background:none; white-space:nowrap; padding-right:8px}}
td.ok {{background:#D7F0E1}} td.warn {{background:#FBEBC4}} td.bad {{background:#F6CFCF}} td.na {{background:#F2F4F7}}
/* alertas */
.al {{display:flex; gap:12px; padding:9px 2px; border-bottom:1px solid #EEF1F5}}
.al:last-child {{border-bottom:none}}
.al .ic {{flex:0 0 30px; height:30px; border-radius:50%; display:flex; align-items:center; justify-content:center}}
.al .tx {{flex:1; font-size:12px; color:#5B6B80}} .al .tx b {{display:block; font-size:13px; color:{NAVY}}}
.al .tm {{font-size:12px; color:#5B6B80; white-space:nowrap}}
/* flujo de datos */
.flow {{display:flex; align-items:stretch; gap:6px; background:#fff; border:1px solid #DDE3EA; border-radius:10px;
        padding:10px; margin-top:14px}}
.flow .tt {{background:{NAVY}; color:#fff; border-radius:8px; padding:8px 14px; font-weight:800; font-size:14px;
        display:flex; align-items:center; min-width:150px}}
.flow .st {{flex:1; display:flex; align-items:center; gap:8px; border:1px solid #DDE3EA; border-radius:8px; padding:6px 8px}}
.flow .st b {{display:block; font-size:12px; color:{NAVY}}} .flow .st small {{font-size:10.5px; color:#7A8799}}
.flow .ar {{display:flex; align-items:center; color:{AZUL}; font-weight:800}}
.flow .st.fin {{background:{AZUL}; border-color:{AZUL}}} .flow .st.fin b, .flow .st.fin small {{color:#fff}}
/* hallazgos */
.ins {{background:#fff; border:1px solid #DDE3EA; border-radius:10px; padding:12px 16px; margin-bottom:12px; border-left:6px solid {AZUL}}}
.ins.ok {{border-left-color:{VERDE}}} .ins.bad {{border-left-color:{ROJO}}} .ins.warn {{border-left-color:{AMBAR}}} .ins.opp {{border-left-color:{AMARILLO}}}
.ins .tag {{font-size:11px; font-weight:700; text-transform:uppercase; color:{AZUL}}}
.ins.ok .tag {{color:{VERDE}}} .ins.bad .tag {{color:{ROJO}}} .ins.warn .tag {{color:#9A6A00}} .ins.opp .tag {{color:#B58B00}}
.ins h4 {{margin:2px 0 4px; color:{NAVY}; font-size:15px}} .ins p {{margin:0; font-size:13.5px; line-height:1.5}}
.rec li {{margin:6px 0; font-size:13.5px; color:#2C3E55}}
</style>""", unsafe_allow_html=True)

ICONOS = {
    "call": "M6.62 10.79c1.44 2.83 3.76 5.14 6.59 6.59l2.2-2.2c.27-.27.67-.36 1.02-.24 1.12.37 2.33.57 3.57.57.55 0 1 .45 1 1V20c0 .55-.45 1-1 1-9.39 0-17-7.61-17-17 0-.55.45-1 1-1h3.5c.55 0 1 .45 1 1 0 1.25.2 2.45.57 3.57.11.35.03.74-.25 1.02l-2.2 2.2z",
    "check": "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z",
    "target": "M12 7c-2.76 0-5 2.24-5 5s2.24 5 5 5 5-2.24 5-5-2.24-5-5-5zm0-5C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8z",
    "done": "M18 7l-1.41-1.41-6.34 6.34 1.41 1.41L18 7zm4.24-1.41L11.66 16.17 7.48 12l-1.41 1.41L11.66 19l12-12-1.42-1.41zM.41 13.41L6 19l1.41-1.41L1.83 12 .41 13.41z",
    "end": "M12 9c-1.6 0-3.15.25-4.6.72v3.1c0 .39-.23.74-.56.9-.98.49-1.87 1.12-2.66 1.85-.18.18-.43.28-.7.28-.28 0-.53-.11-.71-.29L.29 13.08c-.18-.17-.29-.42-.29-.7 0-.28.11-.53.29-.71C3.34 8.78 7.46 7 12 7s8.66 1.78 11.71 4.67c.18.18.29.43.29.71 0 .28-.11.53-.29.71l-2.48 2.48c-.18.18-.43.29-.71.29-.27 0-.52-.11-.7-.28-.79-.74-1.69-1.36-2.67-1.85-.33-.16-.56-.5-.56-.9v-3.1C15.15 9.25 13.6 9 12 9z",
    "down": "M16 18l2.29-2.29-4.88-4.88-4 4L2 7.41 3.41 6l6 6 4-4 6.3 6.29L22 12v6z",
    "up": "M16 6l2.29 2.29-4.88 4.88-4-4L2 16.59 3.41 18l6-6 4 4 6.3-6.29L22 12V6z",
    "timer": "M9 1h6v2H9zm10.03 6.39l1.42-1.42c-.43-.51-.9-.99-1.41-1.41l-1.42 1.42C16.07 4.84 14.12 4 12 4c-4.97 0-9 4.03-9 9s4.02 9 9 9 9-4.03 9-9c0-2.12-.74-4.07-1.97-5.61zM13 14h-2V8h2v6z",
    "clock": "M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z",
    "hourglass": "M6 2v6h.01L6 8.01 10 12l-4 4 .01.01H6V22h12v-5.99h-.01L18 16l-4-4 4-3.99-.01-.01H18V2H6z",
    "group": "M16 11c1.66 0 2.99-1.34 2.99-3S17.66 5 16 5c-1.66 0-3 1.34-3 3s1.34 3 3 3zm-8 0c1.66 0 2.99-1.34 2.99-3S9.66 5 8 5C6.34 5 5 6.34 5 8s1.34 3 3 3zm0 2c-2.33 0-7 1.17-7 3.5V19h14v-2.5c0-2.33-4.67-3.5-7-3.5zm8 0c-.29 0-.62.02-.97.05 1.16.84 1.97 1.97 1.97 3.45V19h6v-2.5c0-2.33-4.67-3.5-7-3.5z",
    "event": "M16.53 11.06L15.47 10l-4.88 4.88-2.12-2.12-1.06 1.06L10.59 17l5.94-5.94zM19 3h-1V1h-2v2H8V1H6v2H5c-1.11 0-1.99.9-1.99 2L3 19c0 1.1.89 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm0 16H5V8h14v11z",
    "warn": "M1 21h22L12 2 1 21zm12-3h-2v-2h2v2zm0-4h-2v-4h2v4z",
    "info": "M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-6h2v6zm0-8h-2V7h2v2z",
    "db": "M2 20h20v-4H2v4zm2-3h2v2H4v-2zM2 4v4h20V4H2zm4 3H4V5h2v2zm-4 7h20v-4H2v4zm2-3h2v2H4v-2z",
    "chat": "M20 2H4c-1.1 0-1.99.9-1.99 2L2 22l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zM6 9h12v2H6V9zm8 5H6v-2h8v2zm4-6H6V6h12v2z",
    "sync": "M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46C19.54 15.03 20 13.57 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74C4.46 8.97 4 10.43 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z",
    "bars": "M5 9.2h3V19H5zM10.6 5h2.8v14h-2.8zm5.6 8H19v6h-2.8z",
    "screen": "M21 2H3c-1.1 0-2 .9-2 2v12c0 1.1.9 2 2 2h7v2H8v2h8v-2h-2v-2h7c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H3V4h18v12z",
    "bolt": "M7 2v11h3v9l7-12h-4l4-8z",
}


def ico(nombre, color=AZUL, tam=18):
    return (f'<svg viewBox="0 0 24 24" width="{tam}" height="{tam}" fill="{color}">'
            f'<path d="{ICONOS[nombre]}"/></svg>')


_n_panel = itertools.count()


@contextmanager
def panel(titulo, extra=""):
    with st.container(key=f"pnl_{next(_n_panel)}"):
        x = f'<span class="x">{extra}</span>' if extra else ""
        st.markdown(f'<div class="ph"><span>{titulo}</span>{x}</div>', unsafe_allow_html=True)
        yield


# ---------------------------------------------------------------- datos
@st.cache_data(show_spinner="Cargando bases...")
def cargar() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not CSV.exists() or not CSV_AG.exists() or not BANNER.exists():
        import generar_dashboard as gd
        if not CSV.exists() or not CSV_AG.exists():
            CSV.parent.mkdir(exist_ok=True)
            llamadas, agentes = gd.generar_base()
            llamadas.to_csv(CSV, index=False, sep=";", encoding="utf-8-sig")
            agentes.to_csv(CSV_AG, index=False, sep=";", encoding="utf-8-sig")
        if not BANNER.exists():
            gd.crear_banner()
    ag = pd.read_csv(CSV_AG, sep=";", encoding="utf-8-sig")
    ag["fecha"] = pd.to_datetime(ag["fecha"])
    ag["dow"] = ag["fecha"].dt.dayofweek
    ag["mes"] = ag["fecha"].dt.to_period("M").dt.to_timestamp()
    df = pd.read_csv(CSV, sep=";", encoding="utf-8-sig", dtype={"agente_id": str})
    df["fecha"] = pd.to_datetime(df["fecha"])
    df["dow"] = df["fecha"].dt.dayofweek
    df["mes"] = df["fecha"].dt.to_period("M").dt.to_timestamp()
    hum = df["canal"].eq(HUMANO)
    ate = df["estado"].eq("Atendida")
    df["ofr"] = 1
    df["ate"] = ate.astype(int)
    df["aba"] = (~ate).astype(int)
    df["abc"] = (~ate & (df["tiempo_espera_s"] < 5)).astype(int)
    df["atu"] = (hum & ate & (df["tiempo_espera_s"] <= 20)).astype(int)
    df["h_ofr"] = hum.astype(int)
    df["h_ate"] = (hum & ate).astype(int)
    df["v_ate"] = (~hum & ate).astype(int)
    df["esp_ate"] = df["tiempo_espera_s"].where(hum & ate, 0)
    df["tmo_h"] = df["tmo_s"].where(hum & ate, 0)
    df["tmo_v"] = df["tmo_s"].where(~hum & ate, 0)
    return df, ag


def derivar(s: pd.DataFrame, con_cont: bool) -> pd.DataFrame:
    s = s.astype(float)
    div = lambda a, b: a / b.where(b > 0)
    s["abp"] = div(s["aba"], s["ofr"])
    s["ns"] = div(s["atu"], s["h_ofr"])
    s["cr"] = div(s["ate"], s["ofr"] - s["abc"])
    s["asa"] = div(s["esp_ate"], s["h_ate"])
    s["tmoH"] = div(s["tmo_h"], s["h_ate"])
    s["tmoV"] = div(s["tmo_v"], s["v_ate"])
    s["tmo"] = div(s["tmo_h"] + s["tmo_v"], s["ate"])
    s["cont"] = div(s["v_ate"], s["ofr"]) if con_cont else np.nan
    s["transf"] = div(s["h_ofr"], s["ofr"]) if con_cont else np.nan
    for c, t in (("conv", "tiempo_conversacion_s"), ("hold", "tiempo_hold_s"), ("acw", "tiempo_acw_s")):
        s[c] = div(s[t], s["h_ate"])
    return s


def resumen(d: pd.DataFrame, by=None):
    if by is None:
        return derivar(d[FLAGS].sum().to_frame().T, CON_CONT).iloc[0]
    return derivar(d.groupby(by)[FLAGS].sum(), CON_CONT)


def resumen_ag(a: pd.DataFrame, by=None):
    s = a[AFLAGS].sum().to_frame().T if by is None else a.groupby(by)[AFLAGS].sum()
    s = s.astype(float)
    manejo = s["seg_conversacion"] + s["seg_hold"] + s["seg_acw"]
    total = manejo + s["seg_disponible"]
    s["occ"] = manejo / total.where(total > 0)
    s["adh"] = s["min_en_estado_planificado"] / s["min_planificados"].where(s["min_planificados"] > 0)
    if not AG_OK:
        s[["occ", "adh"]] = np.nan
    return s.iloc[0] if by is None else s


# ---------------------------------------------------------------- formato
nulo = lambda v: v is None or (isinstance(v, float) and np.isnan(v))
num = lambda v: "—" if nulo(v) else f"{v:,.0f}".replace(",", ".")
pct = lambda v: "—" if nulo(v) else f"{v * 100:.1f}".replace(".", ",") + "%"
spct = lambda v: f"{v * 100:+.1f}".replace(".", ",") + "%"
seg = lambda v: "—" if nulo(v) else f"{v:.0f} s"
mmss = lambda v: "—" if nulo(v) else f"{int(round(v)) // 60}:{int(round(v)) % 60:02d}"
mes_lbl = lambda m: f"{MESES[m.month - 1]} {m.year}"
hh = lambda h: f"{int(h):02d}:00"
CSS_EST = {"ok": f"background-color:#DDF3E7;color:{VERDE};font-weight:600",
           "warn": "background-color:#FCEFD3;color:#9A6A00;font-weight:600",
           "bad": f"background-color:#F9DEDE;color:{ROJO};font-weight:600", "": ""}


def estado(k, v, meta_k=None):
    if k not in DIR or nulo(v):
        return ""
    m = METAS[k] if meta_k is None else meta_k
    if k == "occ":
        return "ok" if m[0] <= v <= m[1] else "warn" if m[0] - .05 <= v <= m[1] + .03 else "bad"
    if DIR[k] > 0:
        return "ok" if v >= m else "warn" if v >= m * .95 else "bad"
    return "ok" if v <= m else "warn" if v <= m * 1.15 else "bad"


color_est = lambda e: {"ok": VERDE, "warn": AMBAR, "bad": ROJO}.get(e, AZUL)


def estilo(fig, alto=300):
    fig.update_layout(height=alto, margin=dict(l=6, r=6, t=30, b=6),
                      plot_bgcolor="#fff", paper_bgcolor="#fff",
                      font=dict(family="Segoe UI, Arial", color="#41536B", size=11),
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, font=dict(size=11)),
                      hovermode="x unified" if fig.layout.hovermode is None else fig.layout.hovermode)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EDF1F6")
    return fig


def meta(fig, valor, texto, **kw):
    fig.add_hline(y=valor, line_dash="dash", line_color=ROJO, line_width=1.5,
                  annotation_text=texto, annotation_font_color=ROJO, **kw)


def grafico(fig, alto=300):
    st.plotly_chart(estilo(fig, alto), width="stretch", config={"displaylogo": False})


def tabla(disp: pd.DataFrame, est: pd.DataFrame):
    sty = disp.style.apply(lambda _: est.replace(CSS_EST), axis=None)
    st.dataframe(sty, hide_index=True, width="stretch")


COLS_KPI = [("Nivel de Servicio", "ns", pct), ("Nivel de Atención", "cr", pct), ("% Abandono", "abp", pct),
            ("TMO / AHT", "tmoH", mmss), ("ASA", "asa", seg), ("IA %", "cont", pct),
            ("Asesor %", "transf", pct), ("TMO Virtual", "tmoV", mmss)]
COLS_AG = [("Ocupación", "occ", pct), ("Adherencia", "adh", pct)]


def tabla_dim(d: pd.DataFrame, by: str, etiqueta: str, a: pd.DataFrame | None = None):
    r = resumen(d, by).sort_values("ofr", ascending=False)
    r.loc["Total"] = resumen(d)
    if a is not None:
        ra = resumen_ag(a, by)
        ra.loc["Total"] = resumen_ag(a)
        r = r.join(ra[["occ", "adh"]])
    disp = pd.DataFrame({etiqueta: r.index, "Ofrecidas": r["ofr"].map(num),
                         "% Part.": (r["ofr"] / r.loc["Total", "ofr"]).map(pct),
                         "Atendidas": r["ate"].map(num), "Abandonadas": r["aba"].map(num)})
    if by == "tipificacion":
        disp.insert(0, "Skill", disp[etiqueta].map(SKILL_DE).fillna(""))
    metas_ia = r.index.map(lambda x: META_IA.get(x, METAS["cont"]))
    est = pd.DataFrame("", index=disp.index, columns=disp.columns)
    for col, k, f in COLS_KPI + (COLS_AG if a is not None else []):
        disp[col] = r[k].map(f).to_numpy()
        est[col] = [estado(k, v, mi if k == "cont" else None) for v, mi in zip(r[k], metas_ia)]
        if k == "cont":
            disp["Meta IA %"] = [pct(mi) for mi in metas_ia]
            est["Meta IA %"] = ""
    orden = [c for c in disp.columns if c not in ("Abandonadas", "Ocupación", "Adherencia")]
    orden.insert(orden.index("Nivel de Atención") + 1, "Abandonadas")
    if a is not None:
        orden[orden.index("ASA") + 1:orden.index("ASA") + 1] = ["Ocupación", "Adherencia"]
    disp, est = disp[orden], est[orden]
    disp.index = est.index = range(len(disp))
    tabla(disp, est)


# ---------------------------------------------------------------- carga y filtros
df, ag = cargar()
SKILL_DE = df.groupby("tipificacion")["skill"].first()
meses = list(pd.DatetimeIndex(df["mes"].unique()).sort_values())
logo_b64 = base64.b64encode(LOGO.read_bytes()).decode()

st.markdown(f'<div class="hdr"><img src="data:image/png;base64,{logo_b64}"><div class="sep"></div><div>'
            '<h1>Analítica de Datos &amp; <span>Visibilidad Operativa</span></h1>'
            '<p>Control 360° de la línea inbound · Agente Virtual + Asesores</p></div></div>',
            unsafe_allow_html=True)
with st.container(key="filtros"):
    h0, h1, h2, h3, h4 = st.columns([.9, 1.2, 1.6, 2.6, 1.6], vertical_alignment="bottom")
    h0.markdown(f'<div class="ft">{ico("bars", "#fff", 18)}Filtros</div>', unsafe_allow_html=True)
    f_mes = h1.selectbox("Mes", ["Todos"] + meses, format_func=lambda m: m if m == "Todos" else mes_lbl(m))
    f_skill = h2.multiselect("Skill", ["SAC", "PQRS", "Venta"], placeholder="Todos")
    opc_tip = df[df["skill"].isin(f_skill)] if f_skill else df
    f_tip = h3.multiselect("Tipificación", opc_tip["tipificacion"].value_counts().index.tolist(),
                           placeholder="Todas")
    f_canal = h4.selectbox("Canal", ["Todos", VIRTUAL, HUMANO])
periodo = "Jul – Sep 2026" if f_mes == "Todos" else mes_lbl(f_mes)

CON_CONT = f_canal == "Todos"
AG_OK = f_canal != VIRTUAL
a_sin_mes = ag[ag["skill"].isin(f_skill)] if f_skill else ag
a = a_sin_mes if f_mes == "Todos" else a_sin_mes[a_sin_mes["mes"].eq(f_mes)]
m = pd.Series(True, index=df.index)
if f_skill:
    m &= df["skill"].isin(f_skill)
if f_tip:
    m &= df["tipificacion"].isin(f_tip)
if f_canal != "Todos":
    m &= df["canal"].eq(f_canal)
d_sin_mes = df[m]
d = d_sin_mes if f_mes == "Todos" else d_sin_mes[d_sin_mes["mes"].eq(f_mes)]
if d.empty:
    st.warning("No hay datos para la combinación de filtros seleccionada.")
    st.stop()
o = pd.concat([resumen(d), resumen_ag(a)[["occ", "adh"]]])
# días calendario del periodo filtrado por día de semana (para promedios por día)
dias_dow = d.drop_duplicates("fecha")["dow"].value_counts().reindex(range(7), fill_value=0)
# TMO según el canal filtrado: asesor, agente virtual o global (ambos)
TK = {"Todos": "tmo", VIRTUAL: "tmoV", HUMANO: "tmoH"}[f_canal]
TMETA = {"Todos": None, VIRTUAL: METAS["tmoV"], HUMANO: METAS["tmoH"]}[f_canal]
TSUB = {"Todos": f"Virtual {mmss(o['tmoV'])} · Asesor {mmss(o['tmoH'])}",
        VIRTUAL: "duración IA · meta ≤ 2:00", HUMANO: "mm:ss · meta ≤ 5:00"}[f_canal]


def fig_tmo(r: pd.DataFrame, horizontal=False):
    series = [("tmoV", "Agente virtual", AMARILLO), ("tmoH", "Asesor", AZUL)] if TK == "tmo" else [(TK, "TMO", None)]
    fig = go.Figure()
    for k, n, cl in series:
        v = r[k]
        ejes = dict(y=r.index, x=v, orientation="h") if horizontal else dict(x=r.index, y=v)
        fig.add_bar(name=n, text=v.map(mmss), marker_color=cl or [color_est(estado(k, x)) for x in v], **ejes)
    if TMETA:
        linea = fig.add_vline if horizontal else fig.add_hline
        linea(TMETA, line_dash="dash", line_color=ROJO, annotation_text=f"Meta {mmss(TMETA)}",
              annotation_font_color=ROJO)
    return fig.update_layout(barmode="group", showlegend=TK == "tmo")

# ---------------------------------------------------------------- TARJETAS KPI (siempre visibles)
kpis = [("Ofrecidas", "ofr", num, "call", "llamadas"),
        ("Atendidas", "ate", num, "check", f"Virtual {num(o['v_ate'])} · Asesor {num(o['h_ate'])}"),
        ("Nivel Servicio", "ns", pct, "target", "meta ≥ 80% en 20 s"),
        ("Nivel Atención", "cr", pct, "done", "meta ≥ 95%"),
        ("Abandonadas", "aba", num, "end", "llamadas"),
        ("% Abandono", "abp", pct, "down", "meta ≤ 5%"),
        ("TMO", TK, mmss, "timer", TSUB),
        ("ASA", "asa", seg, "hourglass", "meta ≤ 20 s"),
        ("Ocupación", "occ", pct, "group", "meta 80% – 90%"),
        ("Adherencia", "adh", pct, "event", "meta ≥ 90%")]
st.markdown('<div class="kgrid">' + "".join(
    f'<div class="kc"><div class="top">{ico(ic)}{lbl}</div>'
    f'<div class="v" style="color:{color_est(estado(k, o[k])) if k in DIR else NAVY}">{f(o[k])}</div>'
    f'<div class="s">{sub}</div></div>' for lbl, k, f, ic, sub in kpis) + "</div>", unsafe_allow_html=True)

tabs = st.tabs(["Resumen", "Línea de tiempo", "Tipificaciones", "Skills", "Intradía", "Hallazgos", "Metodología"])

# ---------------------------------------------------------------- RESUMEN
with tabs[0]:
    n_dias = max(int(dias_dow.sum()), 1)
    rh = resumen(d, "hora")
    rha = resumen_ag(a, "hora") if AG_OK else None
    c1, c2, c3, c4 = st.columns([1.15, 1.15, 1.1, .85])
    with c1, panel("Volumen por intervalo", "Promedio día"):
        fig = go.Figure()
        for col, n, c in (("v_ate", "Agente virtual", AMARILLO), ("h_ate", "Asesor", AZUL), ("aba", "Abandonadas", ROJO)):
            fig.add_bar(x=[f"{h}:00" for h in rh.index], y=(rh[col] / n_dias).round(1), name=n, marker_color=c)
        grafico(fig.update_layout(barmode="stack", bargap=.25), 270)
    with c2, panel("Tendencia de TMO (mm:ss)", "Diario · " + ("Global" if TK == "tmo" else f_canal)):
        td = resumen(d, "fecha")[TK].dropna()
        fig = go.Figure(go.Scatter(x=td.index, y=td, name="TMO", mode="lines", line=dict(color=AZUL, width=2),
                                   fill="tozeroy", fillcolor="rgba(1,98,186,.12)"))
        if TMETA:
            fig.add_hline(y=TMETA, line_dash="dash", line_color=NAVY, line_width=1.2)
        if len(td):
            fig.add_annotation(x=td.index[-1], y=td.iloc[-1], text=f"<b>{mmss(o[TK])}</b><br>TMO periodo",
                               bgcolor=AZUL, font=dict(color="#fff", size=11), showarrow=True, arrowcolor=AZUL, ax=-40, ay=-30)
            tope = max(td.max(), TMETA or 0) * 1.15
            paso = 30 if tope < 300 else 60 if tope < 600 else 120
            ticks = list(range(0, int(tope) + paso, paso))
            fig.update_yaxes(range=[0, tope], tickvals=ticks, ticktext=[mmss(t) for t in ticks])
        grafico(fig.update_layout(showlegend=False, hovermode="closest"), 270)
    with c3, panel("Estado de agentes / tiempo", periodo):
        if AG_OK and not a.empty:
            sa = a[AFLAGS].sum()
            no_adh = (sa["min_planificados"] - sa["min_en_estado_planificado"]) * 60
            partes = [("Conversación", sa["seg_conversacion"], AZUL), ("Hold", sa["seg_hold"], CIELO),
                      ("ACW", sa["seg_acw"], NAVY), ("Disponible", sa["seg_disponible"], AMARILLO),
                      ("Fuera de turno", no_adh, GRIS)]
            total = sum(v for _, v, _ in partes)
            barra = "".join(f'<div style="width:{v / total * 100:.1f}%;background:{c}">'
                            f'{pct(v / total) if v / total >= .06 else ""}</div>' for _, v, c in partes)
            leyenda = "".join(f'<div><span class="dot" style="background:{c}"></span>{n}<b>{num(v / 3600)} h</b></div>'
                              for n, v, c in partes)
            st.markdown(f'<div class="ag-top"><div><b>{a["agente_id"].nunique()}</b><span>agentes</span></div>'
                        f'<div><b>{num(sa["min_planificados"] / 60)}</b><span>horas planificadas</span></div></div>'
                        f'<div class="ag-bar">{barra}</div><div class="ag-leg">{leyenda}</div>', unsafe_allow_html=True)
        else:
            st.info("Aplica a asesores (Canal ≠ Agente Virtual).")
    with c4, panel("Cola de asesores", "Periodo"):
        hum = d[d["canal"].eq(HUMANO)]
        ag_int = a.groupby(["fecha", "hora"]).size().mean() if AG_OK and not a.empty else np.nan
        filas_q = [("call", "Transferidas a asesor", num(o["h_ofr"])),
                   ("clock", "Espera más larga", mmss(hum["tiempo_espera_s"].max()) if len(hum) else "—"),
                   ("group", "Asesores por intervalo", "—" if nulo(ag_int) else f"{ag_int:.1f}".replace(".", ",")),
                   ("end", "Abandono corto (&lt; 5 s)", num(o["abc"])),
                   ("bolt", "Resuelto por IA", pct(o["cont"]))]
        st.markdown("".join(f'<div class="ql"><span>{ico(i, AZUL, 20)}{t}</span><b>{v}</b></div>' for i, t, v in filas_q),
                    unsafe_allow_html=True)

    c1, c2 = st.columns([2.2, 1])
    with c1, panel("Desempeño por hora (KPIs operativos)", periodo):
        horas = list(rh.index)
        filas_h = [("TMO", TK, mmss, rh), ("Nivel Servicio", "ns", pct, rh), ("Nivel Atención", "cr", pct, rh),
                   ("Abandono", "abp", pct, rh)]
        if rha is not None:
            filas_h.append(("Ocupación", "occ", pct, rha))
        cuerpo = "".join(
            f'<tr><td class="r">{lbl}</td>' + "".join(
                f'<td class="{estado(k, src[k].get(h, np.nan)) or "na"}">{f(src[k].get(h, np.nan))}</td>' for h in horas)
            + "</tr>" for lbl, k, f, src in filas_h)
        st.markdown(f'<table class="dh"><tr><th></th>{"".join(f"<th>{h}:00</th>" for h in horas)}</tr>{cuerpo}</table>',
                    unsafe_allow_html=True)
    with c2, panel("Alertas operativas", periodo):
        alertas = []
        if TMETA and rh[TK].notna().any():
            ht = rh[TK].idxmax()
            if rh.loc[ht, TK] > TMETA:
                alertas.append(("warn", ROJO, "TMO por encima del objetivo",
                                f"A las {hh(ht)} es {mmss(rh.loc[ht, TK])} (objetivo {mmss(TMETA)})", hh(ht)))
        if o["h_ofr"] > 0:
            hn = rh["ns"].idxmin()
            if rh.loc[hn, "ns"] < METAS["ns"]:
                alertas.append(("warn", NARANJA, "Nivel de servicio bajo el objetivo",
                                f"A las {hh(hn)} es {pct(rh.loc[hn, 'ns'])} (objetivo 80%)", hh(hn)))
        rdia = resumen(d, "fecha")
        if len(rdia) > 3:
            dp = rdia["ofr"].idxmax()
            alertas.append(("up", AMBAR, "Pico de volumen",
                            f"{num(rdia.loc[dp, 'ofr'])} llamadas ({spct(rdia.loc[dp, 'ofr'] / rdia['ofr'].median() - 1)} "
                            "vs mediana diaria)", dp.strftime("%d/%m")))
        if rha is not None and AG_OK:
            occ_sk = resumen_ag(a, "skill")["occ"].dropna()
            if len(occ_sk) and occ_sk.min() < METAS["occ"][0]:
                alertas.append(("group", AZUL, "Ocupación fuera de rango",
                                f"{occ_sk.idxmin()} con {pct(occ_sk.min())} (rango 80% – 90%)", occ_sk.idxmin()))
            if o["adh"] < METAS["adh"] + .01:
                alertas.append(("info", AZUL, "Adherencia en el límite", f"{pct(o['adh'])} de cumplimiento (meta ≥ 90%)",
                                "Periodo"))
        if not alertas:
            alertas.append(("check", VERDE, "Sin alertas", "Todos los indicadores dentro de meta", ""))
        st.markdown("".join(
            f'<div class="al"><div class="ic" style="background:{c}1F">{ico(i, c, 18)}</div>'
            f'<div class="tx"><b>{t}</b>{s}</div><div class="tm">{tm}</div></div>' for i, c, t, s, tm in alertas[:5]),
            unsafe_allow_html=True)

    rm = resumen(d_sin_mes, "mes").join(resumen_ag(a_sin_mes, "mes")[["occ", "adh"]])
    c1, c2, c3 = st.columns([1, 1.4, 2.2])
    with c1, panel("Distribución de llamadas"):
        fig = go.Figure(go.Pie(labels=["Agente virtual", "Asesor", "Abandonadas"],
                               values=[o["v_ate"], o["h_ate"], o["aba"]], hole=.62, sort=False,
                               marker=dict(colors=[AMARILLO, AZUL, ROJO], line=dict(color="#fff", width=2))))
        grafico(fig.update_layout(hovermode="closest", legend=dict(orientation="h", y=-0.1, yanchor="top")), 300)
    with c2, panel("Volumen mensual"):
        fig = go.Figure()
        for col, n, c in (("ofr", "Ofrecidas", NAVY), ("ate", "Atendidas", AZUL), ("aba", "Abandonadas", NARANJA)):
            fig.add_bar(x=[mes_lbl(x) for x in rm.index], y=rm[col], name=n, marker_color=c,
                        text=rm[col].map(num), textposition="outside")
        grafico(fig, 300)
    with c3, panel("Comparativo mensual vs meta"):
        filas = [("Llamadas ofrecidas", "ofr", num, "—"), ("Llamadas atendidas", "ate", num, "—"),
                 ("Nivel de Servicio (20 s)", "ns", pct, "≥ 80%"), ("Nivel de Atención", "cr", pct, "≥ 95%"),
                 ("Llamadas abandonadas", "aba", num, "—"), ("% Abandono", "abp", pct, "≤ 5%"),
                 ("TMO" + (" global" if TK == "tmo" else ""), TK, mmss, f"≤ {mmss(TMETA)}" if TMETA else "—"),
                 ("ASA", "asa", seg, "≤ 20 s"),
                 ("Ocupación", "occ", pct, "80% – 90%"), ("Adherencia", "adh", pct, "≥ 90%"),
                 ("IA % (complementario)", "cont", pct, f"≥ {pct(METAS['cont'])}"),
                 ("TMO virtual (complementario)", "tmoV", mmss, "≤ 2:00")]
        tot = pd.concat([resumen(d_sin_mes), resumen_ag(a_sin_mes)[["occ", "adh"]]])
        cols = [mes_lbl(x) for x in rm.index] + ["Total"]
        disp = pd.DataFrame([[l] + [f(rm.loc[x, k]) for x in rm.index] + [f(tot[k]), mt] for l, k, f, mt in filas],
                            columns=["Indicador"] + cols + ["Meta"])
        est = pd.DataFrame([[""] + [estado(k, rm.loc[x, k]) for x in rm.index] + [estado(k, tot[k]), ""]
                            for _, k, _, _ in filas], columns=disp.columns)
        tabla(disp, est)

    pasos = [("db", "ACD / CDR", "Llamadas y colas"), ("chat", "Agente Virtual", "Logs de IA"),
             ("event", "WFM", "Turnos y adherencia"), ("sync", "Integración", "ETL / consolidación"),
             ("bars", "Modelo de KPIs", "Reglas y metas"), ("screen", "Dashboard", "Visualización")]
    st.markdown('<div class="flow"><div class="tt">Flujo de datos<br>a la acción</div>' + "".join(
        f'<div class="st">{ico(i, AZUL, 22)}<div><b>{t}</b><small>{s}</small></div></div><div class="ar">→</div>'
        for i, t, s in pasos) + f'<div class="st fin">{ico("bolt", AMARILLO, 22)}<div><b>Acción</b>'
        '<small>Decisiones operativas</small></div></div></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- LÍNEA DE TIEMPO
with tabs[1]:
    gran = st.radio("Granularidad", ["Diaria", "Semanal", "Mensual"], horizontal=True)
    if gran == "Diaria":
        clave, fmt_x = d["fecha"], lambda x: x.strftime("%d/%m")
    elif gran == "Semanal":
        clave, fmt_x = d["fecha"] - pd.to_timedelta(d["dow"], unit="D"), lambda x: "Sem " + x.strftime("%d/%m")
    else:
        clave, fmt_x = d["mes"], mes_lbl
    rt = resumen(d.assign(_k=clave), "_k")
    x = [fmt_x(i) for i in rt.index]

    with panel("Volumen de llamadas", gran):
        fig = go.Figure()
        for col, n, c in (("v_ate", "Atendidas agente virtual", AMARILLO), ("h_ate", "Atendidas asesor", AZUL),
                          ("aba", "Abandonadas", ROJO)):
            fig.add_bar(x=x, y=rt[col], name=n, marker_color=c)
        fig.add_scatter(x=x, y=rt["ofr"], name="Ofrecidas", mode="lines", line=dict(color=NAVY, width=1, dash="dot"))
        grafico(fig.update_layout(barmode="stack"), 340)

    c1, c2 = st.columns(2)
    with c1:
        with panel("Nivel de servicio y nivel de atención", gran):
            fig = go.Figure()
            fig.add_scatter(x=x, y=rt["ns"] * 100, name="Nivel de servicio", line=dict(color=AZUL, width=2.5))
            fig.add_scatter(x=x, y=rt["cr"] * 100, name="Nivel de atención", line=dict(color=NARANJA, width=2.5))
            meta(fig, 80, "Meta NS 80%")
            meta(fig, 95, "Meta NA 95%", annotation_position="top left")
            grafico(fig.update_yaxes(ticksuffix="%", range=[0, 102]))
        with panel("TMO asesor vs agente virtual", gran):
            fig = go.Figure()
            fig.add_scatter(x=x, y=rt["tmoH"], name="TMO asesor (s)", line=dict(color=AZUL, width=2.5))
            fig.add_scatter(x=x, y=rt["tmoV"], name="TMO virtual (s)", line=dict(color=NARANJA, width=2.5))
            meta(fig, 300, "Meta asesor 300 s")
            meta(fig, 120, "Meta virtual 120 s")
            grafico(fig.update_yaxes(rangemode="tozero", title="Segundos"))
    with c2:
        with panel("ASA y % de abandono", gran):
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_bar(x=x, y=rt["asa"], name="ASA (s)", marker_color=CELESTE)
            fig.add_scatter(x=x, y=rt["abp"] * 100, name="% Abandono", line=dict(color=ROJO, width=2.5),
                            secondary_y=True)
            meta(fig, 20, "Meta ASA 20 s")
            fig.update_yaxes(title="Segundos", secondary_y=False)
            fig.update_yaxes(ticksuffix="%", secondary_y=True, showgrid=False)
            grafico(fig)
        with panel("IA %: resolución del agente virtual", gran):
            if CON_CONT:
                fig = go.Figure(go.Scatter(x=x, y=rt["cont"] * 100, name="IA %", fill="tozeroy",
                                           line=dict(color=AMARILLO, width=2.5), fillcolor="rgba(241,209,26,.25)"))
                meta(fig, METAS["cont"] * 100, f"Meta {pct(METAS['cont'])}")
                grafico(fig.update_yaxes(ticksuffix="%", rangemode="tozero"))
            else:
                st.info("IA % solo aplica con el filtro Canal = Todos.")

    with panel("Ocupación y adherencia a turno", gran):
        if AG_OK:
            clave_a = {"Diaria": a["fecha"], "Semanal": a["fecha"] - pd.to_timedelta(a["dow"], unit="D"),
                       "Mensual": a["mes"]}[gran]
            rta = resumen_ag(a.assign(_k=clave_a), "_k")
            xa = [fmt_x(i) for i in rta.index]
            fig = go.Figure()
            fig.add_scatter(x=xa, y=rta["occ"] * 100, name="Ocupación", line=dict(color=AZUL, width=2.5))
            fig.add_scatter(x=xa, y=rta["adh"] * 100, name="Adherencia", line=dict(color=NARANJA, width=2.5))
            fig.add_hrect(y0=80, y1=90, fillcolor=VERDE, opacity=.08, line_width=0,
                          annotation_text="Rango meta ocupación 80–90%", annotation_position="bottom left")
            meta(fig, 90, "Meta adherencia 90%", annotation_position="top right")
            grafico(fig.update_yaxes(ticksuffix="%", range=[40, 102]))
        else:
            st.info("Ocupación y adherencia aplican a agentes humanos (filtro Canal ≠ Agente Virtual).")

# ---------------------------------------------------------------- TIPIFICACIONES
with tabs[2]:
    with panel("Indicadores por tipificación", periodo):
        tabla_dim(d, "tipificacion", "Tipificación")
    rtp = resumen(d, "tipificacion").sort_values("ofr", ascending=False)
    alto_tip = max(360, 26 * len(rtp) + 80)

    c1, c2 = st.columns(2)
    with c1, panel("Distribución del volumen: skill → tipificación"):
        tm = d.groupby(["skill", "tipificacion"]).size().reset_index(name="Ofrecidas")
        fig = px.treemap(tm, path=[px.Constant("Línea inbound"), "skill", "tipificacion"], values="Ofrecidas",
                         color="skill", color_discrete_map={"(?)": AZUL_OSC, "SAC": AZUL, "PQRS": NARANJA,
                                                            "Venta": AMARILLO})
        fig.update_traces(texttemplate="%{label}<br>%{percentRoot:.1%}", root_color=AZUL_OSC)
        grafico(fig.update_layout(hovermode="closest", margin=dict(t=6)), alto_tip)
    with c2, panel("Resolución IA % vs Asesor % por tipificación"):
        if CON_CONT:
            ia = rtp.iloc[::-1]
            fig = go.Figure()
            fig.add_bar(y=ia.index, x=ia["cont"] * 100, name="IA %", orientation="h", marker_color=AMARILLO,
                        text=ia["cont"].map(pct), textposition="inside")
            fig.add_bar(y=ia.index, x=ia["transf"] * 100, name="Asesor %", orientation="h", marker_color=AZUL,
                        text=ia["transf"].map(pct), textposition="inside")
            grafico(fig.update_layout(barmode="stack", hovermode="y unified").update_xaxes(ticksuffix="%",
                                                                                         range=[0, 100]), alto_tip)
        else:
            st.info("IA % y Asesor % solo aplican con el filtro Canal = Todos.")

    with panel("Pareto de tipificaciones", "Azul = SAC · Naranja = PQRS · Amarillo = Venta"):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=rtp.index, y=rtp["ofr"], name="Ofrecidas", marker_color=[
            {"SAC": AZUL, "PQRS": NARANJA, "Venta": AMARILLO}.get(SKILL_DE.get(t), AZUL) for t in rtp.index])
        fig.add_scatter(x=rtp.index, y=rtp["ofr"].cumsum() / rtp["ofr"].sum() * 100, name="% acumulado",
                        line=dict(color=NAVY, width=2.5), secondary_y=True)
        fig.add_hline(y=80, line_dash="dash", line_color=GRIS, secondary_y=True)
        fig.update_yaxes(ticksuffix="%", range=[0, 105], secondary_y=True, showgrid=False)
        grafico(fig.update_xaxes(tickangle=-45), 420)

    c1, c2 = st.columns(2)
    with c1, panel("TMO por tipificación", "Global" if TK == "tmo" else f_canal):
        tt = rtp.dropna(subset=[TK]).sort_values(TK)
        grafico(fig_tmo(tt, horizontal=True).update_layout(hovermode="y unified"), alto_tip)
    with c2, panel("Evolución mensual (top 8 tipificaciones)"):
        top8 = rtp.index[:8]
        ev = d.assign(Tipificación=d["tipificacion"].where(d["tipificacion"].isin(top8), "Otras"))
        ev = ev.groupby(["mes", "Tipificación"]).size().reset_index(name="Ofrecidas")
        ev["Mes"] = ev["mes"].map(mes_lbl)
        fig = px.bar(ev, x="Mes", y="Ofrecidas", color="Tipificación", category_orders={"Tipificación": [*top8, "Otras"]},
                     color_discrete_sequence=PALETA + [GRIS])
        grafico(fig, alto_tip)

# ---------------------------------------------------------------- SKILLS
with tabs[3]:
    with panel("Indicadores por skill", periodo):
        tabla_dim(d, "skill", "Skill", a)
    rs = resumen(d, "skill")
    c1, c2, c3 = st.columns(3)
    with c1, panel("Nivel de servicio por skill"):
        fig = go.Figure(go.Bar(x=rs.index, y=rs["ns"] * 100, name="Nivel de servicio", text=rs["ns"].map(pct),
                               marker_color=[color_est(estado("ns", v)) for v in rs["ns"]]))
        meta(fig, 80, "Meta 80%")
        grafico(fig.update_yaxes(ticksuffix="%", range=[0, 100]))
    with c2, panel("ASA y % abandono por skill"):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=rs.index, y=rs["asa"], name="ASA (s)", marker_color=AZUL, offsetgroup=0)
        fig.add_bar(x=rs.index, y=rs["abp"] * 100, name="% Abandono", marker_color=NARANJA, offsetgroup=1,
                    secondary_y=True)
        fig.update_yaxes(title="Segundos", secondary_y=False)
        fig.update_yaxes(ticksuffix="%", secondary_y=True, showgrid=False)
        grafico(fig)
    with c3, panel("TMO por skill", "Global" if TK == "tmo" else f_canal):
        grafico(fig_tmo(rs))
    c1, c2 = st.columns(2)
    with c1, panel("Volumen mensual por skill"):
        ev = d.groupby(["skill", "mes"]).size().reset_index(name="Ofrecidas")
        ev["Mes"] = ev["mes"].map(mes_lbl)
        fig = px.bar(ev, x="skill", y="Ofrecidas", color="Mes", barmode="group",
                     color_discrete_sequence=[CELESTE, AZUL, NAVY, AMARILLO], labels={"skill": "Skill"})
        grafico(fig)
    with c2, panel("Ocupación y adherencia a turno por skill"):
        if AG_OK:
            rsa = resumen_ag(a, "skill")
            fig = go.Figure()
            fig.add_bar(x=rsa.index, y=rsa["occ"] * 100, name="Ocupación", text=rsa["occ"].map(pct),
                        marker_color=[color_est(estado("occ", v)) for v in rsa["occ"]])
            fig.add_bar(x=rsa.index, y=rsa["adh"] * 100, name="Adherencia", text=rsa["adh"].map(pct),
                        marker_color=CELESTE)
            fig.add_hrect(y0=80, y1=90, fillcolor=VERDE, opacity=.08, line_width=0)
            meta(fig, 90, "Meta adherencia 90% · rango ocupación 80–90%")
            grafico(fig.update_yaxes(ticksuffix="%", range=[0, 105]).update_layout(barmode="group"))
        else:
            st.info("Ocupación y adherencia aplican a agentes humanos (filtro Canal ≠ Agente Virtual).")
    with panel("Nivel de servicio semanal por skill"):
        sem = resumen(d.assign(sem=d["fecha"] - pd.to_timedelta(d["dow"], unit="D")),
                      ["sem", "skill"])["ns"].reset_index()
        sem["Semana"] = sem["sem"].dt.strftime("Sem %d/%m")
        sem["ns"] *= 100
        fig = px.line(sem, x="Semana", y="ns", color="skill", markers=True, color_discrete_sequence=PALETA,
                      labels={"ns": "Nivel de servicio", "skill": "Skill"})
        meta(fig, 80, "Meta 80%")
        grafico(fig.update_yaxes(ticksuffix="%"))

# ---------------------------------------------------------------- INTRADÍA
with tabs[4]:
    n_dias = dias_dow.sum()
    rh = resumen(d, "hora")
    with panel("Curva intradía: ofrecidas promedio por día, nivel de servicio y ocupación", periodo):
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=[hh(h) for h in rh.index], y=rh["ofr"] / n_dias, name="Ofrecidas promedio/día",
                    marker_color=AZUL)
        fig.add_scatter(x=[hh(h) for h in rh.index], y=rh["ns"] * 100, name="Nivel de servicio",
                        line=dict(color=NARANJA, width=2.5), secondary_y=True)
        if AG_OK:
            rha = resumen_ag(a, "hora")
            fig.add_scatter(x=[hh(h) for h in rha.index], y=rha["occ"] * 100, name="Ocupación",
                            line=dict(color=NAVY, width=2.5, dash="dot"), secondary_y=True)
        fig.add_hline(y=80, line_dash="dash", line_color=ROJO, secondary_y=True)
        fig.update_yaxes(ticksuffix="%", range=[0, 102], secondary_y=True, showgrid=False)
        grafico(fig, 320)

    rdh = resumen(d, ["dow", "hora"])
    vol = rdh["ofr"].unstack().reindex(range(7)).div(dias_dow.replace(0, np.nan), axis=0)
    ns = rdh["ns"].unstack().reindex(range(7)) * 100
    c1, c2 = st.columns(2)
    with c1, panel("Mapa de calor: promedio de llamadas por intervalo"):
        fig = px.imshow(vol.round(0), x=[f"{h:02d}h" for h in vol.columns], y=DIAS, text_auto=".0f", aspect="auto",
                        color_continuous_scale=["#E8F1FB", AZUL, NAVY])
        grafico(fig.update_layout(hovermode="closest", coloraxis_showscale=False), 320)
    with c2, panel("Mapa de calor: nivel de servicio por intervalo", "Rojo &lt; 60% · Amarillo ≈ 80% · Verde ≥ 90%"):
        fig = px.imshow(ns.round(0), x=[f"{h:02d}h" for h in ns.columns], y=DIAS, text_auto=".0f", aspect="auto",
                        zmin=50, zmax=100,
                        color_continuous_scale=[[0, ROJO], [.2, ROJO], [.6, AMARILLO], [.8, VERDE], [1, VERDE]])
        grafico(fig.update_layout(hovermode="closest", coloraxis_showscale=False), 320)

    with panel("Comportamiento por día de la semana"):
        rd = resumen(d, "dow")
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_bar(x=[DIAS[i] for i in rd.index], y=rd["ofr"] / dias_dow[rd.index].to_numpy(),
                    name="Ofrecidas promedio/día", marker_color=AZUL)
        fig.add_scatter(x=[DIAS[i] for i in rd.index], y=rd["ns"] * 100, name="Nivel de servicio",
                        line=dict(color=NARANJA, width=2.5), secondary_y=True)
        fig.add_scatter(x=[DIAS[i] for i in rd.index], y=rd["abp"] * 100, name="% Abandono",
                        line=dict(color=ROJO, width=2.5), secondary_y=True)
        fig.update_yaxes(ticksuffix="%", range=[0, 102], secondary_y=True, showgrid=False)
        grafico(fig)


# ---------------------------------------------------------------- HALLAZGOS
def hallazgos():
    ins, rec = [], []
    hum = o["h_ofr"] > 0
    ins.append(("ok" if hum and o["ns"] >= METAS["ns"] and o["abp"] <= METAS["abp"] else "bad" if hum else "info",
                "Panorama general del periodo",
                f"Se ofrecieron <b>{num(o['ofr'])}</b> llamadas; se atendieron <b>{num(o['ate'])}</b> (nivel de atención "
                f"<b>{pct(o['cr'])}</b> vs meta ≥ 95%) y se abandonaron <b>{num(o['aba'])}</b> ({pct(o['abp'])} vs meta ≤ 5%)."
                + (f" El nivel de servicio cerró en <b>{pct(o['ns'])}</b> (meta 80% en 20 s), ASA de <b>{seg(o['asa'])}</b> "
                   f"y TMO de <b>{mmss(o['tmoH'])}</b>." if hum else "")))

    rm = resumen(d_sin_mes, "mes")
    if len(rm) > 1 and hum:
        f, l = rm.iloc[0], rm.iloc[-1]
        best, worst, top = rm["ns"].idxmax(), rm["ns"].idxmin(), rm["ofr"].idxmax()
        txt = (f"El volumen pasó de {num(f['ofr'])} llamadas en {mes_lbl(rm.index[0])} a {num(l['ofr'])} en "
               f"{mes_lbl(rm.index[-1])} ({spct(l['ofr'] / f['ofr'] - 1)}). El mes de mayor demanda fue <b>{mes_lbl(top)}</b> "
               f"({num(rm.loc[top, 'ofr'])}). El mejor nivel de servicio se logró en <b>{mes_lbl(best)}</b> "
               f"({pct(rm.loc[best, 'ns'])}) y el más crítico fue <b>{mes_lbl(worst)}</b> ({pct(rm.loc[worst, 'ns'])}, "
               f"abandono {pct(rm.loc[worst, 'abp'])}).")
        if worst == top:
            txt += " La caída coincide con el mes de mayor volumen: la capacidad humana no creció al ritmo de la demanda."
            rec.append(f"Ajustar el dimensionamiento (Erlang C) para meses de alta temporada como {mes_lbl(top)}: el volumen "
                       f"creció {spct(rm.loc[top, 'ofr'] / f['ofr'] - 1)} frente a {mes_lbl(rm.index[0])} sin un aumento "
                       "proporcional de capacidad.")
        ins.append(("info", "Evolución mensual", txt))

    rdia = resumen(d, "fecha")
    if len(rdia) > 3 and hum:
        med = rdia["ofr"].median()
        picos = rdia.index[rdia["ofr"] > 1.5 * med]
        if len(picos):
            pk, rs_ = resumen(d[d["fecha"].isin(picos)]), resumen(d[~d["fecha"].isin(picos)])
            ins.append(("bad", "Picos de demanda (preventas / eventos)",
                        f"Se identificaron <b>{len(picos)}</b> días con volumen superior a 1,5 veces la mediana diaria "
                        f"({num(med)}): {', '.join(p.strftime('%d/%m') for p in picos)}. En esos días el nivel de servicio "
                        f"cayó a <b>{pct(pk['ns'])}</b> y el abandono subió a <b>{pct(pk['abp'])}</b>, frente a {pct(rs_['ns'])} "
                        f"y {pct(rs_['abp'])} en días normales. Estos días explican {pct(pk['aba'] / o['aba'])} de los abandonos."))
            rec.append("Construir con el área comercial un calendario de eventos (preventas, aperturas de taquilla, cambios de "
                       "fecha) y activar un plan de contingencia: refuerzo de turnos, mensajes proactivos en el agente virtual "
                       "y opción de devolución de llamada (callback).")
        bajo = int((rdia["ns"] < METAS["ns"]).sum())
        r_bajo = bajo / len(rdia)
        ins.append(("bad" if r_bajo > .3 else "warn" if r_bajo > .1 else "ok", "Cumplimiento diario del nivel de servicio",
                    f"<b>{bajo}</b> de {len(rdia)} días ({pct(r_bajo)}) cerraron por debajo de la meta 80/20. El referente de "
                    "una operación madura es cumplir la meta en más del 90% de los días."))
        alto = rdia.index[rdia["asa"] > METAS["asa"]]
        if 0 < len(alto) < len(rdia):
            a1, a2 = resumen(d[d["fecha"].isin(alto)]), resumen(d[~d["fecha"].isin(alto)])
            ins.append(("info", "Relación tiempo de espera – abandono",
                        f"En los {len(alto)} días con ASA superior a 20 s el abandono fue de <b>{pct(a1['abp'])}</b>, versus "
                        f"<b>{pct(a2['abp'])}</b> en los días con ASA en meta. Cada segundo adicional de espera se traduce en "
                        "llamadas (y potenciales ventas) perdidas."))

    rtp = resumen(d, "tipificacion").sort_values("ofr", ascending=False)
    acum = rtp["ofr"].cumsum() / rtp["ofr"].sum()
    top_t = list(acum.index[: int((acum < .8).sum()) + 1])
    ins.append(("info", "Concentración por tipificación (Pareto)",
                f"{len(top_t)} de {len(rtp)} tipificaciones concentran el <b>{pct(acum[top_t[-1]])}</b> del volumen "
                f"({', '.join(top_t)}). La principal es <b>{rtp.index[0]}</b> con {pct(rtp['ofr'].iloc[0] / o['ofr'])} "
                "de las llamadas."))

    if CON_CONT:
        if len(rm) > 1:
            dl = rm["cont"].iloc[-1] - rm["cont"].iloc[0]
            ins.append(("ok" if rm["cont"].iloc[-1] >= METAS["cont"] else "warn", "Resolución del agente virtual (IA %)",
                        f"La IA resolvió el {pct(rm['cont'].iloc[0])} de las llamadas en {mes_lbl(rm.index[0])} y el "
                        f"<b>{pct(rm['cont'].iloc[-1])}</b> en {mes_lbl(rm.index[-1])} ({spct(dl)[:-1]} p.p.). "
                        + ("El aumento reduce la carga sobre los asesores y mejora el nivel de servicio."
                           if dl > 0 else "Se recomienda revisar los flujos con mayor tasa de transferencia.")))
        opp = rtp[rtp["cont"] < .5].sort_values("h_ofr", ascending=False).head(3)
        if len(opp):
            ins.append(("opp", "Oportunidad de automatización",
                        ", ".join(f"<b>{t}</b> ({SKILL_DE.get(t, '')}; IA {pct(r['cont'])}, {num(r['h_ofr'])} transferencias)"
                                  for t, r in opp.iterrows())
                        + " generan el mayor volumen transferido a asesores con baja resolución de la IA. Cada 10 p.p. "
                        f"adicionales de IA en estas tipificaciones liberarían ~<b>{num(opp['ofr'].sum() * .1)}</b> llamadas "
                        "de la cola humana."))
            rec.append(f"Priorizar el rediseño de flujos del agente virtual para {', '.join(opp.index)}: integración con "
                       "backend (estado de casos, devoluciones, validación de cobros), autenticación automática y respuestas "
                       "transaccionales; mantener en asesor los casos sensibles (fraude, cliente VIP).")

    if hum:
        sobre = rtp[rtp["tmoH"] > METAS["tmoH"]].sort_values("tmoH", ascending=False)
        ins.append(("warn" if len(sobre) else "ok", "Tiempo medio operativo (TMO)",
                    f"El TMO de agentes humanos es <b>{mmss(o['tmoH'])}</b> (meta ≤ 5:00): conversación {mmss(o['conv'])}, "
                    f"hold {mmss(o['hold'])} y ACW {mmss(o['acw'])}. "
                    + (f"{len(sobre)} tipificaciones superan la meta; las más altas: "
                       f"{', '.join(f'{t} ({mmss(v)})' for t, v in sobre['tmoH'].head(5).items())}."
                       if len(sobre) else "Todas las tipificaciones se mantienen dentro de la meta.")))
        if len(sobre):
            rec.append(f"Revisar guiones y base de conocimiento de {', '.join(sobre.index[:5])}; monitoreo de calidad "
                       "focalizado y reducción de hold/ACW con tipificación asistida.")
    if o["v_ate"] > 0:
        largas = rtp[rtp["tmoV"] > METAS["tmoV"]].sort_values("tmoV", ascending=False)
        ins.append(("warn" if o["tmoV"] > METAS["tmoV"] else "ok", "Duración de las llamadas con el agente virtual",
                    f"Las llamadas resueltas por el agente virtual duran en promedio <b>{mmss(o['tmoV'])}</b> "
                    "(meta ≤ 2:00). " + (f"Las de mayor duración: "
                                         f"{', '.join(f'{t} ({mmss(v)})' for t, v in largas['tmoV'].head(5).items())}."
                                         if len(largas) else "Todas las tipificaciones están dentro de la meta.")))

    if AG_OK and not a.empty:
        ra_m = resumen_ag(a_sin_mes, "mes")
        ra_h = resumen_ag(a, "hora")
        bajas = [hh(h) for h in ra_h.index[ra_h["occ"] < .70]]
        altas = [hh(h) for h in ra_h.index[ra_h["occ"] > .90]]
        txt = (f"La ocupación de los agentes fue <b>{pct(o['occ'])}</b> (rango meta 80%–90%). "
               + ("Por skill: " + ", ".join(f"{k} {pct(v)}" for k, v in resumen_ag(a, "skill")["occ"].items()) + ". "
                  if a["skill"].nunique() > 1 else "")
               + ("Por mes: " + ", ".join(f"{mes_lbl(k)} {pct(v)}" for k, v in ra_m["occ"].items()) + ". "
                  if len(ra_m) > 1 else "")
               + (f"Franjas con subutilización (&lt; 70%): {', '.join(bajas)}. " if bajas else "")
               + (f"Franjas con sobrecarga (&gt; 90%): {', '.join(altas)}." if altas else ""))
        ins.append((estado("occ", o["occ"]) or "info", "Ocupación de agentes", txt))
        occ_sk = resumen_ag(a, "skill")["occ"]
        bajos = occ_sk[occ_sk < .60]
        if len(bajos) and len(occ_sk) > 1:
            rec.append(f"Habilitar multiskill para {', '.join(bajos.index)} (ocupación "
                       f"{', '.join(pct(v) for v in bajos)}): atender colas de {occ_sk.idxmax()} en tiempos libres sin "
                       "perder prioridad sobre su skill principal.")
        if o["occ"] < METAS["occ"][0]:
            rec.append("Reajustar la dotación a la capacidad liberada por el agente virtual: reducir conexiones en franjas "
                       f"de baja ocupación{' (' + ', '.join(bajas) + ')' if bajas else ''} y reasignarlas a franjas pico, "
                       "backoffice o capacitación.")
        peor_adh = ra_m["adh"].idxmin() if len(ra_m) > 1 else None
        ins.append((estado("adh", o["adh"]) or "info", "Adherencia a turno",
                    f"Los agentes cumplieron el <b>{pct(o['adh'])}</b> de los minutos planificados (meta ≥ 90%)."
                    + (f" El mes más bajo fue <b>{mes_lbl(peor_adh)}</b> ({pct(ra_m.loc[peor_adh, 'adh'])}), lo que reduce la "
                       "capacidad efectiva justo cuando la demanda es mayor." if peor_adh is not None else "")))
        if o["adh"] < METAS["adh"] or (peor_adh is not None and ra_m.loc[peor_adh, "adh"] < METAS["adh"]):
            rec.append("Implementar control de adherencia en tiempo real (RTA) con alertas por desvío de estado, y revisar "
                       "ausentismo y puntualidad en los meses de menor cumplimiento.")

    rs = resumen(d, "skill")
    rs = rs[rs["h_ofr"] > 0]
    if hum and len(rs) > 1:
        w, b = rs["ns"].idxmin(), rs["ns"].idxmax()
        ins.append(("warn" if rs.loc[w, "ns"] < METAS["ns"] else "ok", "Desempeño por skill",
                    f"<b>{w}</b> presenta el nivel de servicio más bajo ({pct(rs.loc[w, 'ns'])}) y <b>{b}</b> el más alto "
                    f"({pct(rs.loc[b, 'ns'])}). El mayor abandono está en {rs['abp'].idxmax()} ({pct(rs['abp'].max())}) y el "
                    f"TMO más alto en {rs['tmoH'].idxmax()} ({mmss(rs['tmoH'].max())})."))
        rec.append(f"Evaluar multiskilling y reglas de desborde hacia {b} para absorber picos de {w}.")

    if hum:
        rh = resumen(d, "hora")
        t3 = rh["ofr"].nlargest(3)
        hw = rh["ns"].idxmin()
        prom = resumen(d, "dow")["ofr"] / dias_dow.replace(0, np.nan)
        ins.append(("info", "Comportamiento intradía y semanal",
                    f"La hora de mayor demanda es <b>{hh(t3.index[0])}</b>; las tres horas pico "
                    f"({', '.join(hh(h) for h in sorted(t3.index))}) concentran {pct(t3.sum() / o['ofr'])} del volumen. "
                    f"El nivel de servicio más bajo se presenta a las <b>{hh(hw)}</b> ({pct(rh.loc[hw, 'ns'])}). El "
                    f"{DIAS_L[prom.idxmax()]} es el día de mayor tráfico ({num(prom.max())} llamadas/día) y el "
                    f"{DIAS_L[prom.idxmin()]} el de menor ({num(prom.min())})."))
        rec.append(f"Ajustar la malla de turnos (WFM) a la curva intradía, reforzando la franja {hh(hw)} y los "
                   f"{DIAS_L[prom.idxmax()]}; programar breaks y capacitaciones en horas valle.")
    rec.append("Para producción: conectar el tablero a la fuente real (ACD/CDR + logs del agente virtual), homologar skills "
               "y tipificaciones, y reemplazar las metas de mercado por las metas contractuales.")
    return ins, rec


with tabs[5]:
    ins, rec = hallazgos()
    etiquetas = {"ok": "Positivo", "bad": "Alerta", "warn": "Riesgo", "info": "Hallazgo", "opp": "Oportunidad"}
    cols = st.columns(2)
    for i, (t, h, p) in enumerate(ins):
        cols[i % 2].markdown(f'<div class="ins {t}"><span class="tag">{etiquetas[t]}</span><h4>{h}</h4><p>{p}</p></div>',
                             unsafe_allow_html=True)
    with panel("Recomendaciones", periodo):
        st.markdown('<ol class="rec">' + "".join(f"<li>{r}</li>" for r in rec) + "</ol>", unsafe_allow_html=True)

# ---------------------------------------------------------------- METODOLOGÍA
with tabs[6]:
    with panel("Definición de indicadores y metas de referencia (estándar de mercado)"):
        st.markdown(f"""
| Métrica | Definición | Fórmula de negocio | Meta |
|---|---|---|---|
| Llamadas ofrecidas | Total de llamadas que ingresan a la línea (entran al agente virtual) | Σ llamadas entrantes | — |
| Llamadas atendidas | Resueltas por el agente virtual + contestadas por un agente humano | Contenidas virtual + Atendidas humano | — |
| Llamadas abandonadas | Transferidas a la cola humana en las que el cliente cuelga antes de ser atendido | Σ abandonos en cola | — |
| **Nivel de Servicio** | % de contactos atendidos dentro de X segundos (X = 20 s, cola humana) | (Atendidos ≤ X seg) / (Ofrecidos) | ≥ 80% |
| **ASA (Average Speed of Answer)** | Tiempo promedio de espera antes de ser atendido | Σ Tiempo de espera / Atendidos | ≤ 20 s |
| **TMO / AHT** | Tiempo promedio por interacción (conversación + hold + ACW) | (Σ ATT + Σ Hold + Σ ACW) / Atendidos — segundos | ≤ 300 s (5:00) |
| **Abandono (%)** | % de llamadas que cuelgan antes de ser atendidas | Abandonos / Ofrecidos | ≤ 5% |
| **Ocupación (%)** | Tiempo productivo sobre el tiempo disponible | (Conversación + Hold + ACW) / (Conversación + Hold + ACW + Disponible) | 80% – 90% |
| **Adherencia a turno (%)** | Cumplimiento de horarios planificados | Minutos en estado planificado / Minutos planificados | ≥ 90% |
| **Nivel de Atención (%)** | % de llamadas que conectan con alguien (agente virtual o humano) | Conectados / Total registros aptos (ofrecidas − abandonos &lt; 5 s) | ≥ 95% |
| IA % *(complementario)* | Llamadas resueltas íntegramente por el agente virtual | Resueltas IA / Ofrecidas | ≥ {pct(METAS['cont'])} (ponderado de la tabla de tipificaciones) |
| TMO agente virtual *(complementario)* | Duración promedio de la interacción resuelta por el agente virtual | Σ duración contenidas / Contenidas | ≤ 2:00 |
""")
        st.caption("Nivel de Servicio, ASA y TMO se miden sobre la cola de agentes humanos; Abandono y Nivel de Atención "
                   "sobre el total ofrecido de la línea. Semáforo: verde = cumple; amarillo = desviación ≤ 5% (mayor es "
                   "mejor), ≤ 15% (menor es mejor) o ocupación entre 75–80% / 90–93%; rojo = desviación mayor. Las metas "
                   "son referencias de mercado y deben reemplazarse por las contractuales.")
    with panel("Catálogo de skills y tipificaciones (base de distribución)"):
        cat = TIP.assign(**{"% Vol.": TIP.vol.map(lambda v: f"{v:g}%".replace(".", ",")),
                            "% Vol. normalizado": TIP.p.map(pct), "IA %": TIP.ia.map(pct),
                            "Asesor %": (1 - TIP.ia).map(pct), "Conversación base asesor": TIP.conv.map(mmss)})
        st.dataframe(cat.rename(columns={"skill": "Skill", "tipificacion": "Tipificación"})[
            ["Skill", "Tipificación", "% Vol.", "% Vol. normalizado", "IA %", "Asesor %", "Conversación base asesor"]],
            hide_index=True, width="stretch")
        st.caption(f"La columna % Vol. de la tabla de negocio suma {TIP.vol.sum():g}%".replace(".", ",")
                   + "; se normaliza a 100% para distribuir la base. IA % y Asesor % se aplican por tipificación (con "
                   "una leve curva de madurez mensual del agente virtual cuyo promedio equivale a la tabla).")
    c1, c2 = st.columns(2)
    with c1, panel("Flujo de atención modelado"):
        st.markdown("1. La llamada ingresa y es atendida por el **agente virtual**.\n"
                    "2. Si el agente virtual resuelve, se cierra como **contenida** (canal Agente Virtual).\n"
                    "3. Si no, se **transfiere al skill** y espera en cola (canal Agente Humano).\n"
                    "4. En cola la llamada es **atendida** por un agente o **abandonada** por el cliente.")
    with c2, panel("Base de prueba"):
        st.markdown(f"`data/base_llamadas_prueba.csv` · **{num(len(df))}** llamadas · "
                    f"{df['fecha'].min():%d/%m/%Y} – {df['fecha'].max():%d/%m/%Y} · franja 07:00–22:59.\n\n"
                    f"`data/base_agentes_prueba.csv` · **{num(len(ag))}** registros agente-intervalo (minutos "
                    "planificados, minutos en estado planificado, conversación, hold, ACW y disponible).\n\n"
                    "**Paso a producción:** reemplazar los CSV por el detalle real de la plataforma (CDR/ACD + agente "
                    "virtual + WFM) con la misma estructura de campos, homologar skills y tipificaciones y ajustar metas.")
        st.download_button("Descargar base filtrada (CSV)", d.drop(columns=FLAGS[:11] + ["dow", "mes"]).to_csv(
            index=False, sep=";").encode("utf-8-sig"), "base_filtrada_tuboleta.csv", "text/csv")
