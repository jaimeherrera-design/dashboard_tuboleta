"""Genera las bases de prueba y el banner del dashboard Streamlit de la línea inbound de Tuboleta.

Uso:  python generar_dashboard.py
Salidas:
  data/base_llamadas_prueba.csv     -> detalle llamada a llamada (datos ficticios)
  data/base_agentes_prueba.csv      -> detalle agente-intervalo (ocupación y adherencia)
  banner_tuboleta.png               -> banner del título construido a partir de Logo.png
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
LOGO = BASE_DIR / "Logo.png"
BANNER = BASE_DIR / "banner_tuboleta.png"
SALIDA_CSV = BASE_DIR / "data" / "base_llamadas_prueba.csv"
SALIDA_AGENTES = BASE_DIR / "data" / "base_agentes_prueba.csv"

rng = np.random.default_rng(2026)

# ----------------------------------------------------------------------------------
# Parámetros del modelo de simulación (ajustables)
# ----------------------------------------------------------------------------------
FECHA_INI, FECHA_FIN = "2026-07-01", "2026-09-30"
HORAS = list(range(7, 23))  # franja de atención 07:00 - 22:59
PESO_HORA = np.array([2, 4, 6, 8, 9, 9, 8, 7, 6, 6, 7, 8, 8, 7, 5, 3], dtype=float)
PESO_HORA /= PESO_HORA.sum()
FACTOR_DIA = [1.15, 1.05, 1.00, 1.00, 1.10, 0.80, 0.60]  # lunes..domingo
VOL_BASE_DIA = 2500
TENDENCIA_MES = {7: 1.00, 8: 1.12, 9: 1.05}
STAFF_MES = {7: 1.18, 8: 1.10, 9: 1.20}       # holgura de capacidad humana planeada
ASESOR_MES = {7: 1.10, 8: 1.00, 9: 0.90}      # madurez del agente virtual (promedio = % Asesor de la tabla)
TMO_MES = {7: 1.00, 8: 1.04, 9: 0.95}
EVENTOS = {  # días de preventa / apertura de taquilla (multiplicador de volumen)
    "2026-07-14": 2.3, "2026-07-15": 1.5,
    "2026-08-04": 2.8, "2026-08-05": 1.9, "2026-08-21": 2.1,
    "2026-09-10": 2.4, "2026-09-11": 1.4,
}
# (skill, tipificación, % volumen, % resuelto por IA, segundos base de conversación con asesor)
TIPIFICACIONES = [
    ("SAC", "Necesita cierre de sesión", 18, .95, 150),
    ("SAC", "Confirmación de compra", 15, .95, 160),
    ("SAC", "No se ven boletas", 12, .90, 200),
    ("SAC", "Acceso cuenta TB Pass", 10, .90, 210),
    ("SAC", "Transferencia de boletas", 6, .90, 220),
    ("SAC", "Entrega en taquilla", 5, .90, 180),
    ("SAC", "Como comprar en página", 4, .80, 240),
    ("SAC", "Novedades de evento", 4, .80, 200),
    ("SAC", "Restricciones del evento", 3, .85, 190),
    ("SAC", "Salida a venta", 4, .95, 170),
    ("SAC", "Evento agotado", 2, .90, 160),
    ("SAC", "Addi / CeroPay", 2, .75, 260),
    ("PQRS", "Tiempo respuesta", 3, .95, 200),
    ("PQRS", "Creación casos", 2, .70, 330),
    ("PQRS", "Cliente entregó info", 1, .80, 260),
    ("PQRS", "Actualización datos", 1, .50, 280),
    ("PQRS", "Facturación electrónica", 1, .60, 300),
    ("PQRS", "Solicitud cierre caso", 1, .40, 260),
    ("PQRS", "Insatisfacción solución", 1, .20, 420),
    ("PQRS", "Cobro sin compra", 1, .10, 380),
    ("PQRS", "Fraude", .5, .10, 450),
    ("PQRS", "Devoluciones", 2.5, .30, 360),
    ("Venta", "Venta efectiva", .5, .40, 420),
    ("Venta", "Cotización", .5, .60, 300),
    ("Venta", "Cambio ubicación", .5, .30, 330),
    ("Venta", "Pago reserva", .5, .50, 300),
    ("Venta", "Cliente VIP", .2, .10, 480),
    ("Venta", "Cliente corporativo", .2, .20, 480),
    ("Venta", "Alianzas", .1, .20, 400),
]
TIP = pd.DataFrame(TIPIFICACIONES, columns=["skill", "tipificacion", "vol", "ia", "conv"])
TIP["p"] = TIP.vol / TIP.vol.sum()  # la tabla suma 101,5%: se normaliza a 100%
TIP["p_hum"] = TIP.p * (1 - TIP.ia)
SKILLS = ["SAC", "PQRS", "Venta"]
HUM_SHARE = TIP.p_hum.sum()
# holgura relativa de cada skill (>1 = más espera que el promedio)
ESPERA_SKILL = {"SAC": 1.00, "PQRS": 1.30, "Venta": 0.60}
ADHERENCIA_MES = {7: 0.91, 8: 0.86, 9: 0.93}  # media de minutos en estado planificado / planificados
CANALES = ["Agente Virtual", "Agente Humano"]
DIAS_ES = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]


def lognormal(media, sigma, n):
    return rng.lognormal(np.log(media) - sigma ** 2 / 2, sigma, n)


def generar_base() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (detalle de llamadas, detalle agente-intervalo)."""
    fechas = pd.date_range(FECHA_INI, FECHA_FIN, freq="D")
    slots = pd.DataFrame([(f, h) for f in fechas for h in HORAS], columns=["fecha", "hora"])
    slots["mes"] = slots.fecha.dt.month
    slots["dow"] = slots.fecha.dt.dayofweek
    peso_h = slots.hora.map(dict(zip(HORAS, PESO_HORA)))
    f_dia = slots.dow.map(lambda d: FACTOR_DIA[d])
    evento = slots.fecha.dt.strftime("%Y-%m-%d").map(EVENTOS).fillna(1.0)
    esperado = VOL_BASE_DIA * f_dia * peso_h * slots.mes.map(TENDENCIA_MES) * evento
    slots["n"] = rng.poisson(esperado.to_numpy())
    # capacidad humana planeada sobre volumen "normal" (sin eventos ni tendencia)
    capacidad = (VOL_BASE_DIA * f_dia * peso_h * HUM_SHARE * slots.mes.map(STAFF_MES)).to_numpy()

    idx = np.repeat(slots.index.to_numpy(), slots.n.to_numpy())
    df = slots.loc[idx, ["fecha", "hora", "mes", "dow"]].reset_index(drop=True)
    n = len(df)

    skills = SKILLS
    t_idx = rng.choice(len(TIP), size=n, p=TIP.p.to_numpy())
    df["skill"] = TIP.skill.to_numpy()[t_idx]
    df["tipificacion"] = TIP.tipificacion.to_numpy()[t_idx]

    p_asesor = np.clip((1 - TIP.ia.to_numpy())[t_idx] * df.mes.map(ASESOR_MES).to_numpy(), 0.01, 1)
    humano = rng.random(n) < p_asesor
    contenida = ~humano

    # cola humana: la espera crece exponencialmente con la ocupación del intervalo
    carga = np.bincount(idx[humano], minlength=len(slots))
    rho = carga / capacidad * rng.lognormal(0, 0.08, len(slots))
    media_espera = np.clip(6 * np.exp(3.0 * (rho - 0.8)), 2, 240)
    espera_pot = rng.exponential(media_espera[idx] * df.skill.map(ESPERA_SKILL).to_numpy())
    paciencia = rng.exponential(300, n)
    abandona = humano & (espera_pot > paciencia)
    atendida_h = humano & ~abandona
    espera = np.where(humano, np.where(abandona, paciencia, espera_pot), 0)

    base_conv = TIP.conv.to_numpy()[t_idx] * df.mes.map(TMO_MES).to_numpy()
    conv = lognormal(base_conv, 0.45, n)
    hold = np.where(rng.random(n) < 0.35, rng.gamma(2.0, 22.0, n), 0)
    acw = np.clip(rng.normal(32, 10, n), 8, None)
    bot = np.where(contenida, lognormal(95, 0.35, n), lognormal(45, 0.40, n))

    to_int = lambda a: np.round(a).astype(int)
    df["canal"] = np.where(contenida, CANALES[0], CANALES[1])
    df["transferida_a_humano"] = humano.astype(int)
    df["estado"] = np.where(abandona, "Abandonada", "Atendida")
    df["tiempo_bot_s"] = to_int(bot)
    df["tiempo_espera_s"] = to_int(espera)
    df["tiempo_conversacion_s"] = to_int(np.where(atendida_h, conv, 0))
    df["tiempo_hold_s"] = to_int(np.where(atendida_h, hold, 0))
    df["tiempo_acw_s"] = to_int(np.where(atendida_h, acw, 0))
    df["tmo_s"] = np.where(
        contenida, df.tiempo_bot_s,
        df.tiempo_conversacion_s + df.tiempo_hold_s + df.tiempo_acw_s)
    # agentes programados por intervalo y skill: capacidad planeada / AHT esperado
    sk_idx = df.skill.map({s: i for i, s in enumerate(skills)}).to_numpy()
    g = TIP.groupby("skill")
    share = (g.p_hum.sum() / HUM_SHARE).reindex(skills).to_numpy()
    aht_sk = ((TIP.p_hum * TIP.conv).groupby(TIP.skill).sum() / g.p_hum.sum()).reindex(skills).to_numpy() + 47
    n_ag = np.round(0.85 * capacidad[:, None] * share[None, :] * aht_sk[None, :]
                    * slots.mes.map(TMO_MES).to_numpy()[:, None] / 3600).astype(int)
    n_ag[:, :2] = np.maximum(n_ag[:, :2], 1)
    franja_venta = slots.hora.between(9, 20).to_numpy()
    n_ag[franja_venta, 2] = np.maximum(n_ag[franja_venta, 2], 1)
    # fuera de su franja, Venta desborda a SAC
    k_ag = np.where(n_ag[idx, sk_idx] > 0, sk_idx, 0)
    agente = (rng.random(n) * n_ag[idx, k_ag]).astype(int) + 1
    turno = np.where(df.hora.to_numpy() < 15, 1, 2)
    df["agente_id"] = np.where(atendida_h, [f"AG{k + 1:02d}{t}{a:02d}" for k, t, a in zip(k_ag, turno, agente)], "")

    S, K = np.meshgrid(np.arange(len(slots)), np.arange(len(skills)), indexing="ij")
    N = n_ag.ravel()
    rs, rk = np.repeat(S.ravel(), N), np.repeat(K.ravel(), N)
    ra = np.concatenate([np.arange(1, k + 1) for k in N])
    ag = pd.DataFrame({"slot": rs, "k": rk, "a": ra})
    h = pd.DataFrame({"slot": idx, "k": k_ag, "a": agente, "seg_conversacion": df.tiempo_conversacion_s,
                      "seg_hold": df.tiempo_hold_s, "seg_acw": df.tiempo_acw_s})[atendida_h]
    ag = ag.merge(h.groupby(["slot", "k", "a"]).sum().reset_index(), how="left").fillna(0)
    media_adh = slots.mes.map(ADHERENCIA_MES).to_numpy()[ag.slot]
    adh = rng.beta(media_adh * 40, (1 - media_adh) * 40)
    ag["min_planificados"] = 60
    ag["min_en_estado_planificado"] = np.round(60 * adh).astype(int)
    logueado = ag.min_en_estado_planificado * 60
    manejo = ag.seg_conversacion + ag.seg_hold + ag.seg_acw
    ag["seg_disponible"] = np.maximum(logueado - manejo, 0.03 * logueado).round().astype(int)
    ag["fecha"] = slots.fecha.dt.strftime("%Y-%m-%d").to_numpy()[ag.slot]
    ag["hora"] = slots.hora.to_numpy()[ag.slot]
    ag["mes"] = slots.fecha.dt.strftime("%Y-%m").to_numpy()[ag.slot]
    ag["dia_semana"] = [DIAS_ES[d] for d in slots.dow.to_numpy()[ag.slot]]
    ag["skill"] = np.array(skills)[ag.k]
    ag["agente_id"] = [f"AG{k + 1:02d}{1 if h < 15 else 2}{a:02d}" for k, h, a in zip(ag.k, ag.hora, ag.a)]
    agentes = ag[["fecha", "hora", "mes", "dia_semana", "skill", "agente_id", "min_planificados",
                  "min_en_estado_planificado", "seg_conversacion", "seg_hold", "seg_acw", "seg_disponible"]]
    agentes = agentes.astype({c: int for c in ["seg_conversacion", "seg_hold", "seg_acw"]})

    df["fecha_hora"] = (df.fecha + pd.to_timedelta(df.hora, unit="h")
                        + pd.to_timedelta(rng.integers(0, 3600, n), unit="s"))
    df = df.sort_values("fecha_hora").reset_index(drop=True)
    df.insert(0, "id_llamada", [f"TB{i:07d}" for i in range(1, len(df) + 1)])
    df["dia_semana"] = df.dow.map(lambda d: DIAS_ES[d])
    df["mes"] = df.fecha.dt.strftime("%Y-%m")
    df["fecha"] = df.fecha.dt.strftime("%Y-%m-%d")
    cols = ["id_llamada", "fecha_hora", "fecha", "hora", "mes", "dia_semana", "skill", "tipificacion",
            "canal", "transferida_a_humano", "estado", "tiempo_bot_s", "tiempo_espera_s",
            "tiempo_conversacion_s", "tiempo_hold_s", "tiempo_acw_s", "tmo_s", "agente_id"]
    return df[cols], agentes


def _fuente(nombres, tam):
    for nombre in nombres:
        for ruta in (Path("C:/Windows/Fonts") / nombre, Path(nombre)):
            if ruta.exists():
                return ImageFont.truetype(str(ruta), tam)
    return ImageFont.load_default()


def crear_banner():
    ancho, alto = 1600, 260
    azul, azul_osc = (1, 98, 186), (1, 52, 110)
    amarillo, naranja = (241, 209, 26), (241, 160, 26)
    img = Image.new("RGB", (ancho, alto), azul)
    dr = ImageDraw.Draw(img)
    for x in range(ancho):
        t = max(0.0, (x - 560) / (ancho - 560))
        dr.line([(x, 0), (x, alto)], fill=tuple(int(a + (b - a) * t) for a, b in zip(azul, azul_osc)))

    logo = Image.open(LOGO).convert("RGB").crop((70, 60, 690, 265))
    alto_logo = 190
    logo = logo.resize((int(logo.width * alto_logo / logo.height), alto_logo), Image.LANCZOS)
    img.paste(logo, (30, (alto - alto_logo) // 2 - 6))

    dr.rounded_rectangle([640, 58, 646, 202], radius=3, fill=amarillo)
    negrita = ["segoeuib.ttf", "arialbd.ttf"]
    normal = ["segoeui.ttf", "arial.ttf"]
    dr.text((675, 52), "Dashboard Línea Inbound", font=_fuente(negrita, 56), fill="white")
    dr.text((677, 128), "Agente Virtual + Agentes Humanos  ·  Indicadores de gestión",
            font=_fuente(normal, 28), fill=(220, 234, 250))
    texto = "MODELO DEMOSTRATIVO  ·  DATOS DE PRUEBA  ·  JUL – SEP 2026"
    f_pill = _fuente(negrita, 18)
    x0, y0, x1, y1 = dr.textbbox((0, 0), texto, font=f_pill)
    dr.rounded_rectangle([677, 182, 677 + (x1 - x0) + 32, 182 + (y1 - y0) + 18], radius=16, fill=amarillo)
    dr.text((693 - x0, 191 - y0), texto, font=f_pill, fill=(11, 43, 85))

    for x in range(ancho):
        t = x / ancho
        dr.line([(x, alto - 8), (x, alto)], fill=tuple(int(a + (b - a) * t) for a, b in zip(amarillo, naranja)))
    img.save(BANNER)


def main():
    df, agentes = generar_base()
    SALIDA_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(SALIDA_CSV, index=False, sep=";", encoding="utf-8-sig")
    agentes.to_csv(SALIDA_AGENTES, index=False, sep=";", encoding="utf-8-sig")

    crear_banner()

    # resumen de control en consola
    h = df[df.canal == CANALES[1]]
    res = df.groupby("mes").apply(lambda g: pd.Series({
        "ofrecidas": len(g),
        "contencion": (g.canal == CANALES[0]).mean(),
        "aband_%": (g.estado == "Abandonada").mean(),
    }), include_groups=False)
    res["ns_80_20"] = h.groupby("mes").apply(
        lambda g: ((g.estado == "Atendida") & (g.tiempo_espera_s <= 20)).sum() / len(g), include_groups=False)
    res["tmo_humano"] = h[h.estado == "Atendida"].groupby("mes").tmo_s.mean()
    ga = agentes.groupby("mes")[["seg_conversacion", "seg_hold", "seg_acw", "seg_disponible",
                                  "min_en_estado_planificado", "min_planificados"]].sum()
    manejo = ga.seg_conversacion + ga.seg_hold + ga.seg_acw
    res["ocupacion"] = manejo / (manejo + ga.seg_disponible)
    res["adherencia"] = ga.min_en_estado_planificado / ga.min_planificados
    print(res.round(3).to_string())
    print(f"\nRegistros: {len(df):,} -> {SALIDA_CSV.name} | {len(agentes):,} -> {SALIDA_AGENTES.name}"
          f"\nDashboard: streamlit run app.py")


if __name__ == "__main__":
    main()
