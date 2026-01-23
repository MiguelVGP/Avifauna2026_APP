import os
import ast
import requests
import pandas as pd
import streamlit as st
import io
from datetime import date
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
import plotly.express as px
import plotly.graph_objects as go
import warnings

import streamlit.components.v1 as components

def _restore_scroll():
    components.html("""
    <script>
    const y = sessionStorage.getItem("st_scroll_y");
    if (y !== null) window.scrollTo(0, parseInt(y));
    </script>
    """, height=0)

_restore_scroll()

components.html("""
<script>
document.addEventListener("click", () => {
  sessionStorage.setItem("st_scroll_y", window.scrollY);
}, true);

document.addEventListener("change", () => {
  sessionStorage.setItem("st_scroll_y", window.scrollY);
}, true);
</script>
""", height=0)

# =========================
# FIXOS (Kobo + Password)
# =========================
KOBO_BASE_URL = "https://kf.kobotoolbox.org"
ASSET_UID = "aheNScdS989eMBppfHUHeW"
API_TOKEN = "8cd83af6fdc8e74c1dfa40430e7a88b5acc04b01"
APP_PASSWORD = "teste"
DEFAULT_LIMIT = 5000

# =========================
# UI / Config (sem CSS)
# =========================
st.set_page_config(page_title="Kobo Data Hub", layout="wide")

# Note: sem CSS, usamos os componentes do Streamlit (títulos, subheaders, markdown bold)
# e definimos fontes/tamanhos nos gráficos Plotly quando aplicável.

# =========================
# Login gate
# =========================
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("Avifauna 2026 🐦 Kobo Data Hub")
    st.caption("Acesso restrito")

    pwd = st.text_input("Palavra-passe", type="password")
    colL, colR = st.columns([1, 3])
    with colL:
        if st.button("Entrar"):
            if pwd == APP_PASSWORD:
                st.session_state.authenticated = True
            else:
                st.error("Palavra-passe incorreta.")
    st.stop()

# =========================
# Helpers
# =========================
def _headers(token: str) -> dict:
    return {"Authorization": f"Token {token}"}

def to_hashable_text(x):
    if isinstance(x, list):
        return " | ".join([str(i) for i in x])
    if isinstance(x, dict):
        return " | ".join([f"{k}:{v}" for k, v in x.items()])
    return x

def normalize_complex_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        try:
            if df2[c].apply(lambda v: isinstance(v, (list, dict))).any():
                df2[c] = df2[c].apply(to_hashable_text)
        except Exception:
            df2[c] = df2[c].astype(str)
    return df2

def try_parse_dates(series: pd.Series) -> pd.Series:
    formats = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"]
    for fmt in formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce", dayfirst=True)
        if parsed.notna().any():
            return parsed

    # Fallback: suprimir o UserWarning específico do pandas sobre inferência de formato
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not infer format, so each element will be parsed individually, falling back to `dateutil`."
        )
        parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
    return parsed

def parse_amostragem_cell(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    if isinstance(x, list):
        return [d for d in x if isinstance(d, dict)]
    if isinstance(x, dict):
        return [x]

    if isinstance(x, str):
        s = x.strip()
        if not s:
            return []

        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, list):
                    return [d for d in parsed if isinstance(d, dict)]
            except Exception:
                return []

        parts = [p.strip() for p in s.split(" | ") if p.strip()]
        out = []
        for p in parts:
            try:
                d = ast.literal_eval(p)
                if isinstance(d, dict):
                    out.append(d)
            except Exception:
                continue
        return out

    return []

def explode_amostragem(df_raw: pd.DataFrame, amostragem_col="Amostragem") -> pd.DataFrame:
    if amostragem_col not in df_raw.columns:
        return pd.DataFrame()

    base = df_raw.copy()
    base["_amostragem_list"] = base[amostragem_col].apply(parse_amostragem_cell)

    exploded = base.explode("_amostragem_list", ignore_index=True)
    exploded = exploded[exploded["_amostragem_list"].notna()].copy()

    am_df = pd.json_normalize(exploded["_amostragem_list"])
    am_df.columns = [str(c) for c in am_df.columns]

    exploded = exploded.drop(columns=[amostragem_col, "_amostragem_list"], errors="ignore").reset_index(drop=True)
    out = pd.concat([exploded, am_df], axis=1)

    if "Amostragem/Espécie" in out.columns:
        out["Amostragem/Espécie_final"] = out["Amostragem/Espécie"].astype(str)
        if "Amostragem/Outra_Esp_cie" in out.columns:
            mask_outra = out["Amostragem/Espécie_final"].str.strip().str.lower().eq("outra")
            out.loc[mask_outra, "Amostragem/Espécie_final"] = out.loc[mask_outra, "Amostragem/Outra_Esp_cie"].astype(str)

    return out

@st.cache_data(ttl=300, show_spinner=False)
def fetch_kobo_raw(limit: int = 5000) -> pd.DataFrame:
    url = f"{KOBO_BASE_URL.rstrip('/')}/api/v2/assets/{ASSET_UID}/data/"
    params = {"format": "json", "limit": limit}

    r = requests.get(url, headers=_headers(API_TOKEN), params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    results = data.get("results", data)
    df_raw = pd.json_normalize(results)

    if "_id" in df_raw.columns:
        df_raw["_id"] = pd.to_numeric(df_raw["_id"], errors="coerce")

    return df_raw

def prepare_submissions_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    return normalize_complex_columns(df_raw)

def apply_filters(df: pd.DataFrame, ui_state: dict) -> pd.DataFrame:
    out = df.copy()

    for col, meta in ui_state.items():
        ftype = meta.get("type")
        val = meta.get("value")

        if ftype is None or val is None:
            continue
        if col not in out.columns:
            continue

        if ftype == "categorical":
            if len(val) > 0:
                series = out[col].apply(to_hashable_text).astype(str)
                out = out[series.isin([str(v) for v in val])]

        elif ftype == "numeric":
            minv, maxv = val
            series = pd.to_numeric(out[col], errors="coerce")
            out = out[series.between(minv, maxv)]

        elif ftype == "text":
            text = str(val).strip()
            if text:
                series = out[col].apply(to_hashable_text).astype(str)
                out = out[series.str.contains(text, case=False, na=False)]

        elif ftype == "date":
            start_date, end_date = val
            s = try_parse_dates(out[col])
            mask = s.dt.date.between(start_date, end_date)
            out = out[mask]

    return out

# =========================
# Header + Controls
# =========================
st.title("Avifauna 2026 🐦 Kobo Data Hub")

b1, b2 = st.columns(2)

with b1:
    if st.button("Recarregar"):
        st.cache_data.clear()

with b2:
    if st.button("Sair"):
        st.session_state.authenticated = False

limit = DEFAULT_LIMIT

# =========================
# Load Data
# =========================
with st.spinner("A carregar dados do Kobo..."): 
    try:
        df_raw = fetch_kobo_raw(limit=int(limit))
    except requests.HTTPError as e:
        st.error("Erro a ligar ao Kobo.")
        st.exception(e)
        st.stop()
    except Exception as e:
        st.error("Erro inesperado ao carregar dados.")
        st.exception(e)
        st.stop()


df = prepare_submissions_df(df_raw)
df_amostras_raw = explode_amostragem(df_raw, amostragem_col="Amostragem")
df_amostras = normalize_complex_columns(df_amostras_raw) if not df_amostras_raw.empty else pd.DataFrame()

# =========================
# Tabs
# =========================
tab_tabela, tab_outputs = st.tabs(["📋 Tabela", "📊 Outputs"])

def build_species_list_pdf(local: str, species_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    today_str = date.today().strftime("%d-%m-%Y")

    # margens
    left = 2.0 * cm
    right = width - 2.0 * cm
    y = height - 2.0 * cm

    # Cabeçalho
    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, y, str(local))

    c.setFont("Helvetica", 11)
    c.drawRightString(right, y + 2, today_str)

    # Total espécies (centro)
    total_especies = int(len(species_df))
    y -= 2.0 * cm
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, y, f"Nº Total de Espécies: {total_especies}")

    # Lista
    y -= 1.5 * cm
    c.setFont("Helvetica", 11)

    line_h = 14  # pts
    bottom = 2.0 * cm

    for _, row in species_df.iterrows():
        especie = str(row["Espécie"])
        n = row["Nº indivíduos"]
        try:
            n_int = int(float(n))
        except Exception:
            n_int = 0

        text = f"{especie} ({n_int})"

        # nova página se necessário
        if y <= bottom:
            c.showPage()
            y = height - 2.0 * cm
            c.setFont("Helvetica", 11)

        c.drawString(left, y, text)
        y -= line_h

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

# =========================
# OUTPUTS TAB
# =========================
with tab_outputs:
    LOCAL_COL = "dados/Local"
    WEEK_COL = "dados/N_Semana"
    SPEC_COL = "Amostragem/Espécie_final"
    INDIV_COL = "Amostragem/N_Indiv_duos"

    k1, k2 = st.columns(2)

    total_registos = len(df_amostras) if not df_amostras.empty else len(df)
    k1.metric("Total de registos", f"{total_registos:,}".replace(",", " "))

    semana_mais_recente = None
    base_semana = df_amostras if (not df_amostras.empty and WEEK_COL in df_amostras.columns) else df
    if WEEK_COL in base_semana.columns:
        semanas = pd.to_numeric(base_semana[WEEK_COL], errors="coerce").dropna()
        if not semanas.empty:
            semana_mais_recente = int(semanas.max())
    k2.metric("Semana mais recente", "-" if semana_mais_recente is None else str(semana_mais_recente))

    st.divider()

    c1, c2 = st.columns([1, 1])

    with c1:
        st.subheader("📍 Registos por local (amostras)")
        if df_amostras.empty or LOCAL_COL not in df_amostras.columns:
            st.info("Não encontrei a coluna 'Amostragem' ou 'dados/Local'.")
        else:
            registos_por_local = (
                df_amostras[LOCAL_COL]
                .fillna("Sem local")
                .astype(str)
                .value_counts()
                .reset_index()
            )
            registos_por_local.columns = ["Local", "Nº de registos"]
            st.dataframe(registos_por_local, width='stretch', height=260)

    with c2:
        st.subheader("🦉 Espécie mais observada por local (por Nº indivíduos)")
        if df_amostras.empty or any(c not in df_amostras.columns for c in [LOCAL_COL, SPEC_COL, INDIV_COL]):
            st.info("Ainda não consegui gerar as colunas necessárias.")
        else:
            tmp2 = df_amostras[[LOCAL_COL, SPEC_COL, INDIV_COL]].copy()
            tmp2[LOCAL_COL] = tmp2[LOCAL_COL].fillna("Sem local").astype(str)
            tmp2[SPEC_COL] = tmp2[SPEC_COL].fillna("Sem espécie").astype(str)
            tmp2[INDIV_COL] = pd.to_numeric(tmp2[INDIV_COL], errors="coerce").fillna(0)

            counts = tmp2.groupby([LOCAL_COL, SPEC_COL])[INDIV_COL].sum().reset_index(name="Nº indivíduos")
            idx = counts.groupby(LOCAL_COL)["Nº indivíduos"].idxmax()
            top = counts.loc[idx].sort_values(LOCAL_COL)

            top = top.rename(columns={
                LOCAL_COL: "Local",
                SPEC_COL: "Espécie mais observada"
            })

            st.dataframe(top, width='stretch', height=260)

    st.divider()

    st.subheader("📍 Espécies por local")

    FIXED_LOCAIS = [
        "Ponte de Lima",
        "Ericeira",
        "Vila Franca de Xira",
        "Lisboa - Estefânia",
    ]

    if df_amostras.empty or any(c not in df_amostras.columns for c in [LOCAL_COL, SPEC_COL, INDIV_COL]):
        st.info("Faltam colunas necessárias para gerar a tabela por local.")
    else:
        base = df_amostras[[LOCAL_COL, SPEC_COL, INDIV_COL]].copy()
        base[LOCAL_COL] = base[LOCAL_COL].fillna(""").astype(str).str.strip()
        base[SPEC_COL]  = base[SPEC_COL].fillna(""").astype(str).str.strip()
        base[INDIV_COL] = pd.to_numeric(base[INDIV_COL], errors="coerce").fillna(0)

        local_sel = st.selectbox("Local", FIXED_LOCAIS, index=0)

        df_loc = base[base[LOCAL_COL] == local_sel]

        if df_loc.empty:
            st.caption("Sem registos.")
            tabela = pd.DataFrame(columns=["Espécie", "Nº indivíduos"])
        else:
            tabela = (
                df_loc.groupby(SPEC_COL, dropna=False)[INDIV_COL]
                .sum()
                .reset_index()
                .rename(columns={SPEC_COL: "Espécie", INDIV_COL: "Nº indivíduos"})
                .sort_values("Nº indivíduos", ascending=False)
                .reset_index(drop=True)
            )

        st.dataframe(tabela, width='stretch', height=420)

    # ===== Download Excel (para o local selecionado) =====
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        tabela.to_excel(writer, index=False, sheet_name="Especies")
    buffer.seek(0)

    st.download_button(
        "⬇️ Download Excel",
        data=buffer,
        file_name=f"especies_{local_sel.replace(' ', '_')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=f"download_excel_local_{local_sel}",
    )

    st.divider()

    st.subheader("📊 Abundância média por espécie (Nº indivíduos / 52 semanas)")

    FIXED_LOCAIS = [
        "Ponte de Lima",
        "Ericeira",
        "Vila Franca de Xira",
        "Lisboa - Estefânia",
    ]
    locais_plot = ["Total"] + FIXED_LOCAIS

    local_plot = st.selectbox("Local (Abundância média)", locais_plot, index=0, key="abund_local_sel")

    if df_amostras.empty or any(c not in df_amostras.columns for c in [LOCAL_COL, SPEC_COL, INDIV_COL]):
        st.info("Faltam colunas necessárias para gerar o gráfico.")
    else:
        base = df_amostras[[LOCAL_COL, SPEC_COL, INDIV_COL]].copy()
        base[LOCAL_COL] = base[LOCAL_COL].fillna(""").astype(str).str.strip()
        base[SPEC_COL]  = base[SPEC_COL].fillna(""").astype(str).str.strip()
        base[INDIV_COL] = pd.to_numeric(base[INDIV_COL], errors="coerce").fillna(0)

        if local_plot != "Total":
            base = base[base[LOCAL_COL] == local_plot]

        if base.empty:
            st.warning("Sem registos para este local.")
        else:
            agg = (
                base.groupby(SPEC_COL, dropna=False)[INDIV_COL]
                .sum()
                .reset_index()
                .rename(columns={SPEC_COL: "Espécie", INDIV_COL: "Total indivíduos"})
            )

            agg["Abundância média (N/52)"] = agg["Total indivíduos"] / 52.0

            top_n = st.slider("Top N espécies", min_value=3, max_value=18, value=10, step=3, key="abund_topn")
            agg = agg.sort_values("Abundância média (N/52)", ascending=True).tail(top_n)

            # gráfico horizontal interativo
            fig = px.bar(
                agg,
                x="Abundância média (N/52)",
                y="Espécie",
                orientation="h",
                hover_data={"Total indivíduos": True, "Abundância média (N/52)": ":.2f"},
                title=f"Abundância média por espécie — {local_plot}",
                color_discrete_sequence=["#FF6600"],
            )
            fig.update_layout(
                height=700,
                margin=dict(l=20, r=20, t=60, b=20),
                font=dict(size=12, family="Helvetica", color="#111111"),
                title=dict(font=dict(size=16)),
            )

            st.plotly_chart(fig, width='stretch')
            st.divider()

    st.subheader("Lista de Espécies (PDF)")
    if "show_lista_especies" not in st.session_state:
        st.session_state.show_lista_especies = False

    if st.button("📄 Lista de Espécies"):
        st.session_state.show_lista_especies = not st.session_state.show_lista_especies

    if st.session_state.show_lista_especies:
        st.subheader("📄 Gerar PDF — Lista de Espécies")

        FIXED_LOCAIS = [
            "Ponte de Lima",
            "Ericeira",
            "Vila Franca de Xira",
            "Lisboa - Estefânia",
        ]

        locais_pdf = ["Total"] + FIXED_LOCAIS
        local_sel = st.selectbox("Local", locais_pdf, index=0, key="pdf_local_sel_total")

        if df_amostras.empty or any(c not in df_amostras.columns for c in [LOCAL_COL, SPEC_COL, INDIV_COL]):
            st.info("Faltam colunas necessárias para gerar a lista.")
        else:
            base = df_amostras[[LOCAL_COL, SPEC_COL, INDIV_COL]].copy()
            base[LOCAL_COL] = base[LOCAL_COL].fillna(""").astype(str).str.strip()
            base[SPEC_COL]  = base[SPEC_COL].fillna(""").astype(str).str.strip()
            base[INDIV_COL] = pd.to_numeric(base[INDIV_COL], errors="coerce").fillna(0)

            if local_sel == "Total":
                df_loc = base.copy()
            else:
                df_loc = base[base[LOCAL_COL] == local_sel]

            if df_loc.empty:
                st.warning("Sem registos para este local.")
            else:
                species_table = (
                    df_loc.groupby(SPEC_COL, dropna=False)[INDIV_COL]
                    .sum()
                    .reset_index()
                    .rename(columns={SPEC_COL: "Espécie", INDIV_COL: "Nº indivíduos"})
                    .sort_values(["Nº indivíduos", "Espécie"], ascending=[False, True])
                    .reset_index(drop=True)
                )

                pdf_bytes = build_species_list_pdf(local_sel, species_table)

                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name=f"lista_especies_{local_sel.replace(' ', '_')}_{date.today().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    key=f"download_pdf_{local_sel}",
                )

    st.divider()
    st.subheader("🟠 Presença / Ausência por mês e semana (circular)")

    WEEK_COL = "dados/N_Semana"
    LOCAL_COL = "dados/Local"
    SPEC_COL = "Amostragem/Espécie_final"

    FIXED_LOCAIS = [
        "Ponte de Lima",
        "Ericeira",
        "Vila Franca de Xira",
        "Lisboa - Estefânia",
    ]
    locais_opts = ["Total"] + FIXED_LOCAIS

    if df_amostras.empty or any(c not in df_amostras.columns for c in [WEEK_COL, LOCAL_COL, SPEC_COL]):
        st.info("Faltam colunas para gerar o gráfico (dados/N_Semana, dados/Local, Amostragem/Espécie_final).")
    else:
        base = df_amostras[[WEEK_COL, LOCAL_COL, SPEC_COL]].copy()
        base[LOCAL_COL] = base[LOCAL_COL].fillna(""").astype(str).str.strip()
        base[SPEC_COL] = base[SPEC_COL].fillna(""").astype(str).str.strip()
        base[WEEK_COL] = pd.to_numeric(base[WEEK_COL], errors="coerce")

        base = base.dropna(subset=[WEEK_COL])
        base = base[(base[WEEK_COL] >= 1) & (base[WEEK_COL] <= 52)]

        if base.empty:
            st.warning("Não há valores válidos em dados/N_Semana (1..52).")
        else:
            colA, colB = st.columns(2)
            with colA:
                local_sel = st.selectbox("Local", locais_opts, index=0, key="pa_local_sel")

            especies = sorted([s for s in base[SPEC_COL].dropna().astype(str).unique() if s.strip() != ""])
            with colB:
                especie_sel = st.selectbox("Espécie", especies, index=0, key="pa_especie_sel")

            work = base[base[SPEC_COL] == especie_sel].copy()
            if local_sel != "Total":
                work = work[work[LOCAL_COL] == local_sel]

            work["Mes"] = ((work[WEEK_COL] - 1) // 4 + 1).astype(int)
            work.loc[work["Mes"] > 12, "Mes"] = 12
            work["SemanaMes"] = ((work[WEEK_COL] - 1) % 4 + 1).astype(int)

            presentes = set(zip(work["Mes"].tolist(), work["SemanaMes"].tolist()))

            meses_nome = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
            labels = []
            colors = []
            thetas = []

            n_segments = 48
            step = 360 / n_segments
            orange = "#00F715"
            gray = "#E0E0E0"

            idx = 0
            for m in range(1, 13):
                for w in range(1, 5):
                    idx += 1
                    thetas.append((idx - 1) * step + step/2)
                    labels.append(f"{meses_nome[m-1]} — Semana {w}")
                    colors.append(orange if (m, w) in presentes else gray)

            fig = go.Figure(
                data=[
                    go.Barpolar(
                        r=[1] * n_segments,
                        theta=thetas,
                        width=[step] * n_segments,
                        marker_color=colors,
                        marker_line_color="white",
                        marker_line_width=1,
                        hovertext=labels,
                        hoverinfo="text",
                        opacity=0.98,
                    )
                ]
            )

            month_centers = [((m - 1) * 4 * step) + (2 * step) for m in range(1, 13)]
            month_width = 4 * step

            fig.add_trace(
                go.Barpolar(
                    r=[1.00] * 12,
                    theta=month_centers,
                    width=[month_width] * 12,
                    marker=dict(
                        color=["rgba(0,0,0,0)"] * 12,
                        line=dict(color="black", width=3),
                    ),
                    hoverinfo="skip",
                    opacity=1,
                )
            )

            month_tickvals = month_centers
            month_ticktext = meses_nome

            fig.update_layout(
                title=f"Presença/Ausência — {especie_sel} ({local_sel})",
                polar=dict(
                    radialaxis=dict(visible=False, range=[0, 1.00]),
                    angularaxis=dict(
                        visible=True,
                        tickmode="array",
                        tickvals=month_tickvals,
                        ticktext=month_ticktext,
                        tickfont=dict(size=14, color="rgba(255,255,255,0.45)"),
                        rotation=0,
                        direction="clockwise",
                        showline=False,
                        showgrid=False,
                    ),
                ),
                showlegend=False,
                height=650,
                margin=dict(l=20, r=20, t=70, b=20),
            )

            st.plotly_chart(fig, width='stretch')

# =========================
# TABLE TAB
# =========================
with tab_tabela:
    st.subheader("📋 Tabela")

    df_base = df_amostras

    with st.expander("🧩 Abrir filtros", expanded=True):
        cols_to_filter = [
            "dados/N_Semana",
            "dados/Data",
            "dados/Local",
            "Amostragem/Espécie_final",
        ]
        cols_to_filter = [c for c in cols_to_filter if c in df_base.columns]

        ui_state = {}
        ncols = 3
        rows = [cols_to_filter[i:i+ncols] for i in range(0, len(cols_to_filter), ncols)]

        for row in rows:
            cols = st.columns(ncols)
            for i in range(ncols):
                if i >= len(row):
                    continue

                colname = row[i]
                s = df_base[colname]

                if pd.api.types.is_numeric_dtype(s):
                    numeric = pd.to_numeric(s, errors="coerce").dropna()
                    if numeric.empty:
                        ui_state[colname] = {"type": None, "value": None}
                        continue

                    minv = float(numeric.min())
                    maxv = float(numeric.max())

                    with cols[i]:
                        if minv >= maxv:
                            st.write(f"**{colname}**")
                            st.caption(f"Valor único: {minv}")
                            ui_state[colname] = {"type": None, "value": None}
                        else:
                            rng = st.slider(colname, min_value=minv, max_value=maxv, value=(minv, maxv))
                            ui_state[colname] = {"type": "numeric", "value": rng}
                else:
                    dt = try_parse_dates(s)
                    valid = dt.dropna()

                    if len(valid) >= max(10, int(0.2 * len(s.dropna()))) and not valid.empty:
                        mind = valid.min().date()
                        maxd = valid.max().date()

                        with cols[i]:
                            dr = st.date_input(colname, value=(mind, maxd))
                            if isinstance(dr, tuple) and len(dr) == 2:
                                ui_state[colname] = {"type": "date", "value": dr}
                            else:
                                ui_state[colname] = {"type": None, "value": None}
                    else:
                        s_safe = s.apply(to_hashable_text)
                        try:
                            nunique = s_safe.nunique(dropna=True)
                        except Exception:
                            nunique = 999999

                        with cols[i]:
                            if nunique <= 50:
                                options = sorted([x for x in s_safe.dropna().astype(str).unique()])
                                sel = st.multiselect(colname, options=options, default=[])
                                ui_state[colname] = {"type": "categorical", "value": sel}
                            else:
                                txt = st.text_input(f"{colname} (contains)", value="")
                                ui_state[colname] = {"type": "text", "value": txt}

    filtered = apply_filters(df_base, ui_state)

    all_cols = list(filtered.columns)
    prefer = [
        "_id", "dados/Data", "dados/Hora", "dados/N_Semana", "dados/Local",
        "Amostragem/Espécie", "Amostragem/Outra_Esp_cie", "Amostragem/Espécie_final",
        "Amostragem/N_Indiv_duos", "Amostragem/Notas"
    ]
    default_show = [c for c in prefer if c in all_cols]
    if len(default_show) < 6:
        default_show = all_cols[:20] if len(all_cols) >= 20 else all_cols

    show_cols = st.multiselect("Colunas visíveis", options=all_cols, default=default_show)

    st.dataframe(filtered[show_cols], width='stretch', height=520)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        filtered[show_cols].to_excel(writer, index=False, sheet_name="Dados Filtrados")
    buffer.seek(0)

    st.download_button(
        "⬇️ Exportar Excel (dados filtrados)",
        data=buffer,
        file_name="kobo_dados_filtrados.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
