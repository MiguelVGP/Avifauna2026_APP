import os
import ast
import io
from datetime import date

import requests
import pandas as pd
import streamlit as st

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

import plotly.express as px
import plotly.graph_objects as go


# =========================
# FIXOS (Kobo + Password)
# =========================
KOBO_BASE_URL = "https://kf.kobotoolbox.org"
ASSET_UID = "ajHjKHzV37ik8jJwLBwBrj"
API_TOKEN = "8cd83af6fdc8e74c1dfa40430e7a88b5acc04b01"
APP_PASSWORD = "BiotaAvifauna"
DEFAULT_LIMIT = 5000

# =========================
# UI (sem CSS)
# =========================
st.set_page_config(page_title="Kobo Data Hub", layout="wide")


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
                st.rerun()
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
            s = pd.to_datetime(out[col], errors="coerce", dayfirst=True, format="mixed")
            mask = s.dt.date.between(start_date, end_date)
            out = out[mask]

    return out


def build_species_list_pdf(local: str, species_df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    today_str = date.today().strftime("%d-%m-%Y")

    left = 2.0 * cm
    right = width - 2.0 * cm
    y = height - 2.0 * cm

    c.setFont("Helvetica-Bold", 14)
    c.drawString(left, y, str(local))

    c.setFont("Helvetica", 11)
    c.drawRightString(right, y + 2, today_str)

    total_especies = int(len(species_df))
    y -= 2.0 * cm
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(width / 2, y, f"Nº Total de Espécies: {total_especies}")

    y -= 1.5 * cm
    c.setFont("Helvetica", 11)

    line_h = 14
    bottom = 2.0 * cm

    for _, row in species_df.iterrows():
        especie = str(row["Espécie"])
        n = row["Nº indivíduos"]
        try:
            n_int = int(float(n))
        except Exception:
            n_int = 0

        text = f"{especie} ({n_int})"

        if y <= bottom:
            c.showPage()
            y = height - 2.0 * cm
            c.setFont("Helvetica", 11)

        c.drawString(left, y, text)
        y -= line_h

    c.save()
    buffer.seek(0)
    return buffer.getvalue()

def build_matrix_pdf(title: str, matrix_df: pd.DataFrame) -> bytes:
    """
    Gera PDF simples da matriz (index = espécies, colunas = locais) com paginação,
    com linhas verticais a separar colunas (locais) e linhas horizontais,
    sem prolongar linhas para além do último local.
    """
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    left = 1.3 * cm
    right = width - 1.3 * cm
    top = height - 1.3 * cm
    bottom = 1.3 * cm

    header_h = 1.2 * cm
    row_h = 0.60 * cm
    font_body = 8
    font_head = 9

    first_col_w = 7.0 * cm
    usable_w = (right - left) - first_col_w

    cols = list(matrix_df.columns)
    rows = list(matrix_df.index)

    col_w = 2.8 * cm
    col_gap = 0.25 * cm
    effective_col_w = col_w + col_gap

    cols_per_page = max(1, int(usable_w // effective_col_w))

    usable_h = (top - bottom) - header_h - 0.6 * cm
    rows_per_page = max(1, int(usable_h // row_h))

    def draw_header(page_title: str, col_chunk: list):
        y = top
        c.setFont("Helvetica-Bold", 12)
        c.drawString(left, y, page_title)

        c.setFont("Helvetica", 9)
        c.drawRightString(right, y, date.today().strftime("%d-%m-%Y"))

        y -= header_h

        c.setFont("Helvetica-Bold", font_head)
        c.drawString(left, y, "Espécie")

        x0 = left + first_col_w
        x = x0
        for col in col_chunk:
            cx = x + (col_w / 2)
            c.drawCentredString(cx, y, str(col)[:22])
            x += effective_col_w

        # Fim da grelha (última coluna de locais) — aqui está o ponto-chave
        grid_right = x0 + (len(col_chunk) * effective_col_w) - (col_gap / 2)

        # Linha separadora do header: só até ao fim da grelha
        y -= 0.25 * cm
        c.setLineWidth(0.6)
        c.line(left, y, grid_right, y)
        y -= 0.25 * cm

        return y, x0, grid_right

    for col_start in range(0, len(cols), cols_per_page):
        col_chunk = cols[col_start: col_start + cols_per_page]

        for row_start in range(0, len(rows), rows_per_page):
            y, x0, grid_right = draw_header(title, col_chunk)

            table_top_y = y + 0.25 * cm
            row_chunk = rows[row_start: row_start + rows_per_page]
            table_bottom_y = table_top_y - (len(row_chunk) * row_h)

            # Linhas verticais a separar LOCAIS (colunas)
            c.setLineWidth(0.3)
            x = x0

            # linha antes da 1ª coluna de locais
            c.line(x - (col_gap / 2), table_top_y, x - (col_gap / 2), table_bottom_y)

            for _ in col_chunk:
                x_end = x + col_w + (col_gap / 2)
                c.line(x_end, table_top_y, x_end, table_bottom_y)
                x += effective_col_w

            # Conteúdo
            c.setFont("Helvetica", font_body)
            y_row = table_top_y - row_h * 0.75

            for r in row_chunk:
                c.drawString(left, y_row, str(r)[:70])

                x = x0
                for col in col_chunk:
                    val = matrix_df.loc[r, col]
                    cx = x + (col_w / 2)
                    c.drawCentredString(cx, y_row, str(val))
                    x += effective_col_w

                # Linhas horizontais: só até ao fim da grelha (não até à margem da página)
                c.setLineWidth(0.2)
                y_line = y_row - (row_h * 0.35)
                c.line(left, y_line, grid_right, y_line)

                y_row -= row_h
                if y_row <= bottom + row_h:
                    break

            c.showPage()

    c.save()
    buffer.seek(0)
    return buffer.getvalue()



# =========================
# Header + Controls
# =========================
st.title("Avifauna 2026 🐦 Kobo Data Hub")

if st.button("Sair"):
    st.session_state.authenticated = False
    st.rerun()


# =========================
# Load Data
# =========================
with st.spinner("A carregar dados do Kobo..."):
    df_raw = fetch_kobo_raw(limit=DEFAULT_LIMIT)

df = normalize_complex_columns(df_raw)
df_amostras_raw = explode_amostragem(df_raw, amostragem_col="Amostragem")
df_amostras = normalize_complex_columns(df_amostras_raw) if not df_amostras_raw.empty else pd.DataFrame()

LOCAL_COL = "dados/Local"
WEEK_COL = "dados/N_Semana"
SPEC_COL = "Amostragem/Espécie_final"
INDIV_COL = "Amostragem/N_Indiv_duos"

FIXED_LOCAIS = ["Ponte de Lima", "Ericeira", "Vila Franca de Xira", "Lisboa - Estefânia", "Loures - Flamenga"]


# =========================
# Sidebar: navegação por secções (com emojis)
# =========================
OPTIONS = [
    "📋 Tabela",
    "🐦‍⬛ Visão geral",
    "📍 Espécies por local",
    "📊 Abundância média",
    "🫧 Bubble — Top espécies",
    "📄 PDF Lista de espécies",
    "✅ Presença / Ausência",
    "🧩 Matriz Presença",   # <-- NOVA
]

#if "section" not in st.session_state:
    #st.session_state.section = OPTIONS[0]

# valor inicial (só 1x) + proteção se houver valor antigo guardado
if "section" not in st.session_state or st.session_state.section not in OPTIONS:
    st.session_state.section = OPTIONS[0]

section = st.sidebar.radio(
    "Secção",
    OPTIONS,
    key="section",
)


# =========================
# RENDER: cada secção começa no TOPO
# =========================
if section == "🐦‍⬛ Visão geral":
    st.subheader("🐦‍⬛ Visão geral")

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

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📍 Registos por local")
        if df_amostras.empty or LOCAL_COL not in df_amostras.columns:
            st.info("Não encontrei a coluna 'Amostragem' ou 'dados/Local'.")
        else:
            registos_por_local = (
                df_amostras[LOCAL_COL].fillna("Sem local").astype(str).value_counts().reset_index()
            )
            registos_por_local.columns = ["Local", "Nº de registos"]
            st.dataframe(registos_por_local, width="stretch", height=420, hide_index=True)

    with c2:
        st.subheader("🦉 Espécie mais observada por local")
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
            top = top.rename(columns={LOCAL_COL: "Local", SPEC_COL: "Espécie mais observada"})
            st.dataframe(top, width="stretch", height=420, hide_index=True)


elif section == "📍 Espécies por local":
    st.subheader("📍 Espécies por local")

    if "out_local_sel" not in st.session_state:
        st.session_state.out_local_sel = FIXED_LOCAIS[0]

    with st.form("form_species_by_local", clear_on_submit=False):
        local_sel = st.selectbox(
            "Local",
            FIXED_LOCAIS,
            index=FIXED_LOCAIS.index(st.session_state.out_local_sel),
        )
        submitted = st.form_submit_button("Aplicar")
        if submitted:
            st.session_state.out_local_sel = local_sel

    local_sel = st.session_state.out_local_sel

    if df_amostras.empty or any(c not in df_amostras.columns for c in [LOCAL_COL, SPEC_COL, INDIV_COL]):
        st.info("Faltam colunas necessárias para gerar a tabela por local.")
        tabela = pd.DataFrame(columns=["Espécie", "Nº indivíduos"])
    else:
        base = df_amostras[[LOCAL_COL, SPEC_COL, INDIV_COL]].copy()
        base[LOCAL_COL] = base[LOCAL_COL].fillna("").astype(str).str.strip()
        base[SPEC_COL] = base[SPEC_COL].fillna("").astype(str).str.strip()
        base[INDIV_COL] = pd.to_numeric(base[INDIV_COL], errors="coerce").fillna(0)

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

    st.dataframe(tabela, width="stretch", height=520)

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


elif section == "📊 Abundância média":
    st.subheader("📊 Abundância média")

    if "out_abund_local" not in st.session_state:
        st.session_state.out_abund_local = "Total"
    if "out_abund_topn" not in st.session_state:
        st.session_state.out_abund_topn = 10

    locais_plot = ["Total"] + FIXED_LOCAIS

    with st.form("form_abund", clear_on_submit=False):
        local_plot = st.selectbox(
            "Local (Abundância média)",
            locais_plot,
            index=locais_plot.index(st.session_state.out_abund_local),
        )
        top_n = st.slider(
            "Top N espécies",
            min_value=3,
            max_value=18,
            value=int(st.session_state.out_abund_topn),
            step=3,
        )
        submitted_abund = st.form_submit_button("Aplicar")
        if submitted_abund:
            st.session_state.out_abund_local = local_plot
            st.session_state.out_abund_topn = top_n

    local_plot = st.session_state.out_abund_local
    top_n = int(st.session_state.out_abund_topn)

    if df_amostras.empty or any(c not in df_amostras.columns for c in [LOCAL_COL, SPEC_COL, INDIV_COL]):
        st.info("Faltam colunas necessárias para gerar o gráfico.")
    else:
        base = df_amostras[[LOCAL_COL, SPEC_COL, INDIV_COL]].copy()
        base[LOCAL_COL] = base[LOCAL_COL].fillna("").astype(str).str.strip()
        base[SPEC_COL] = base[SPEC_COL].fillna("").astype(str).str.strip()
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
            agg = agg.sort_values("Abundância média (N/52)", ascending=True).tail(top_n)

            fig = px.bar(
                agg,
                x="Abundância média (N/52)",
                y="Espécie",
                orientation="h",
                hover_data={"Total indivíduos": True, "Abundância média (N/52)": ":.2f"},
                title=f"Abundância média por espécie — {local_plot}",
            )
            fig.update_layout(height=720, margin=dict(l=20, r=20, t=60, b=20))
            st.plotly_chart(fig, width="stretch")

elif section == "🫧 Bubble — Top espécies":
    st.subheader("🫧 Bubble — Top espécies")

    import base64
    from PIL import Image, ImageDraw
    import numpy as np
    import plotly.graph_objects as go

    # =========================
    # Helper: imagem circular
    # =========================
    def image_to_circular_data_uri(img_path: str, out_px: int = 360) -> str:
        im = Image.open(img_path).convert("RGBA")
        im = im.resize((out_px, out_px), Image.LANCZOS)

        mask = Image.new("L", (out_px, out_px), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, out_px - 1, out_px - 1), fill=255)

        im.putalpha(mask)

        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    # =========================
    # UI state
    # =========================
    if "bb_local" not in st.session_state:
        st.session_state.bb_local = "Total"
    if "bb_topn" not in st.session_state:
        st.session_state.bb_topn = 10

    locais_plot = ["Total"] + FIXED_LOCAIS

    with st.form("form_bubble_abund", clear_on_submit=False):
        c1, c2 = st.columns([2, 1])
        with c1:
            local_plot = st.selectbox(
                "Local",
                locais_plot,
                index=locais_plot.index(st.session_state.bb_local),
            )
        with c2:
            top_n = st.slider(
                "Top N espécies",
                min_value=3,
                max_value=18,
                value=int(st.session_state.bb_topn),
                step=1,
            )

        if st.form_submit_button("Aplicar"):
            st.session_state.bb_local = local_plot
            st.session_state.bb_topn = top_n

    local_plot = st.session_state.bb_local
    top_n = int(st.session_state.bb_topn)

    # =========================
    # Dados
    # =========================
    if df_amostras.empty or any(c not in df_amostras.columns for c in [LOCAL_COL, SPEC_COL, INDIV_COL]):
        st.info("Faltam colunas necessárias para gerar o bubble output.")
    else:
        base = df_amostras[[LOCAL_COL, SPEC_COL, INDIV_COL]].copy()
        base[LOCAL_COL] = base[LOCAL_COL].fillna("").astype(str).str.strip()
        base[SPEC_COL] = base[SPEC_COL].fillna("").astype(str).str.strip()
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
            agg = (
                agg.sort_values("Abundância média (N/52)", ascending=False)
                .head(top_n)
                .reset_index(drop=True)
            )

            # =========================
            # Layout + sizing consistentes (DIÂMETRO proporcional ao valor)
            # =========================
            MAX_DIAM_PX = 220   # bolha maior (aumenta para 260/300 se quiseres)
            PX_TO_X = 0.03      # px -> unidades do eixo
            MIN_GAP_PX = 57    # espaço entre bolhas (aumenta para mais espaço)
            
            sizes = agg["Abundância média (N/52)"].astype(float).values
            max_size = float(np.max(sizes)) if len(sizes) else 1.0
            
            # sizeref para sizemode="diameter"
            sizeref = (max_size / MAX_DIAM_PX) if max_size > 0 else 1.0
            
            size_vals = np.maximum(sizes, 0)
            
            # >>> DIÂMETRO em px (não sqrt) porque vamos usar sizemode="diameter"
            diam_px = (size_vals / sizeref) if sizeref > 0 else np.zeros_like(size_vals)
            rad_px = diam_px / 2.0
            
            r_units = rad_px * PX_TO_X
            pad_units = MIN_GAP_PX * PX_TO_X


            def pack_circles_spiral(radii, pad=0.0, angle_step=0.35, r_step=0.6, max_iter=25000):
                radii = list(radii)
                n = len(radii)
                if n == 0:
                    return []
                placed = [(0.0, 0.0)]
                for i in range(1, n):
                    ri = radii[i]
                    t = 0.0
                    rr = 0.0
                    found = False
                    for _ in range(max_iter):
                        x = rr * np.cos(t)
                        y = rr * np.sin(t)
                        ok = True
                        for (xj, yj), rj in zip(placed, radii[:i]):
                            dx = x - xj
                            dy = y - yj
                            if (dx * dx + dy * dy) < (ri + rj + pad) ** 2:
                                ok = False
                                break
                        if ok:
                            placed.append((x, y))
                            found = True
                            break
                        t += angle_step
                        rr += r_step * angle_step
                    if not found:
                        placed.append((rr + ri + pad, 0.0))
                return placed

            n = len(agg)

            if n <= 5:
                xs = [0.0]
                for i in range(1, n):
                    xs.append(xs[-1] + (r_units[i - 1] + r_units[i] + pad_units))
                ys = [0.0] * n
            else:
                order = np.argsort(-r_units)  # maior->menor
                r_sorted = r_units[order]
                pts_sorted = pack_circles_spiral(r_sorted, pad=pad_units)

                pts = [None] * n
                for k, idx_orig in enumerate(order):
                    pts[idx_orig] = pts_sorted[k]

                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]

            agg["x"] = xs
            agg["y"] = ys
            agg["r_units"] = r_units  # <<< IMPORTANTE: guardar raio em unidades do eixo

            # =========================
            # Imagens por espécie
            # =========================
            PASSER_IMG_PATH = "assets/images/passer_domesticus.jpg"
            species_images = {}

            try:
                species_images["passer domesticus"] = image_to_circular_data_uri(
                    PASSER_IMG_PATH, out_px=400
                )
            except Exception:
                pass

            # =========================
            # Cores por bolha
            # =========================
            marker_colors = []
            for _, r in agg.iterrows():
                if str(r["Espécie"]).lower().strip() in species_images:
                    marker_colors.append("rgba(240,240,240,0.85)")  # cinzento claro
                else:
                    marker_colors.append("#BFF7C9")  # verde claro

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=agg["x"],
                    y=agg["y"],
                    mode="markers",
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Abundância média (N/52): %{customdata[1]:.2f}<br>"
                        "Total indivíduos: %{customdata[2]:.0f}<extra></extra>"
                    ),
                    customdata=agg[["Espécie", "Abundância média (N/52)", "Total indivíduos"]].values,
                    marker=dict(
                        size=agg["Abundância média (N/52)"],
                        sizemode="diameter",
                        sizeref=sizeref,
                        sizemin=22,
                        color=marker_colors,
                        line=dict(color="black", width=2.5),
                        opacity=1.0,
                    ),
                )
            )

            # =========================
            # Imagens + texto (imagem enche sempre a bolha)
            # =========================
            for _, r in agg.iterrows():
                especie = str(r["Espécie"])
                especie_key = especie.lower().strip()
                x = float(r["x"])
                y = float(r["y"])
                abund = float(r["Abundância média (N/52)"])

                has_image = especie_key in species_images

                # <<< tamanho da imagem baseado no diâmetro REAL da bolha (unidades do eixo)
                r_u = float(r["r_units"])
                img_size = max(2.0 * r_u * 0.98, 0.55)  # 0.98 "enche"; 0.55 evita desaparecer

                if has_image:
                    fig.add_layout_image(
                        dict(
                            source=species_images[especie_key],
                            xref="x",
                            yref="y",
                            x=x,
                            y=y,
                            xanchor="center",
                            yanchor="middle",
                            sizex=img_size,
                            sizey=img_size,
                            layer="above",
                            opacity=1.0,
                        )
                    )

                fig.add_annotation(
                    x=x,
                    y=y,
                    xref="x",
                    yref="y",
                    text=f"<b>{especie}</b><br><b>{abund:.2f}</b>",
                    showarrow=False,
                    align="center",
                    font=dict(color="black", size=16),
                    bgcolor="rgba(255,255,255,0.55)" if has_image else "rgba(0,0,0,0)",
                    bordercolor="rgba(0,0,0,0.35)" if has_image else "rgba(0,0,0,0)",
                    borderwidth=1 if has_image else 0,
                    borderpad=5 if has_image else 0,
                )

            # limites + escala 1:1 (para círculos “perfeitos”)
            x_min = float(agg["x"].min()) - 2.0
            x_max = float(agg["x"].max()) + 2.0
            y_min = float(agg["y"].min()) - 2.0
            y_max = float(agg["y"].max()) + 2.0

            fig.update_layout(
                title=f"Top {top_n} — Abundância média (N/52) — {local_plot}",
                height=560,
                margin=dict(l=10, r=10, t=70, b=10),
                showlegend=False,
                xaxis=dict(visible=False, range=[x_min, x_max], fixedrange=True),
                yaxis=dict(
                    visible=False,
                    range=[y_min, y_max],
                    fixedrange=True,
                    scaleanchor="x",  # <<< mantém proporções
                    scaleratio=1,
                ),
            )

            st.plotly_chart(fig, width="stretch")


elif section == "📄 PDF Lista de espécies":
    st.subheader("📄 PDF Lista de espécies")

    if "out_pdf_local" not in st.session_state:
        st.session_state.out_pdf_local = "Total"

    locais_pdf = ["Total"] + FIXED_LOCAIS

    with st.form("form_pdf", clear_on_submit=False):
        local_sel_pdf = st.selectbox(
            "Local",
            locais_pdf,
            index=locais_pdf.index(st.session_state.out_pdf_local),
            key="pdf_local_sel_total_form",
        )
        submitted_pdf = st.form_submit_button("Gerar / Atualizar")
        if submitted_pdf:
            st.session_state.out_pdf_local = local_sel_pdf

    local_sel_pdf = st.session_state.out_pdf_local

    if df_amostras.empty or any(c not in df_amostras.columns for c in [LOCAL_COL, SPEC_COL, INDIV_COL]):
        st.info("Faltam colunas necessárias para gerar a lista.")
    else:
        base = df_amostras[[LOCAL_COL, SPEC_COL, INDIV_COL]].copy()
        base[LOCAL_COL] = base[LOCAL_COL].fillna("").astype(str).str.strip()
        base[SPEC_COL] = base[SPEC_COL].fillna("").astype(str).str.strip()
        base[INDIV_COL] = pd.to_numeric(base[INDIV_COL], errors="coerce").fillna(0)

        df_loc = base.copy() if local_sel_pdf == "Total" else base[base[LOCAL_COL] == local_sel_pdf]

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

            pdf_bytes = build_species_list_pdf(local_sel_pdf, species_table)

            st.download_button(
                "⬇️ Download PDF",
                data=pdf_bytes,
                file_name=f"lista_especies_{local_sel_pdf.replace(' ', '_')}_{date.today().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                key=f"download_pdf_{local_sel_pdf}",
            )


elif section == "✅ Presença / Ausência":
    st.subheader("✅ Presença / Ausência")

    locais_opts = ["Total"] + FIXED_LOCAIS

    if df_amostras.empty or any(c not in df_amostras.columns for c in [WEEK_COL, LOCAL_COL, SPEC_COL]):
        st.info("Faltam colunas para gerar o gráfico (dados/N_Semana, dados/Local, Amostragem/Espécie_final).")
    else:
        base = df_amostras[[WEEK_COL, LOCAL_COL, SPEC_COL]].copy()
        base[LOCAL_COL] = base[LOCAL_COL].fillna("").astype(str).str.strip()
        base[SPEC_COL] = base[SPEC_COL].fillna("").astype(str).str.strip()
        base[WEEK_COL] = pd.to_numeric(base[WEEK_COL], errors="coerce")

        base = base.dropna(subset=[WEEK_COL])
        base = base[(base[WEEK_COL] >= 1) & (base[WEEK_COL] <= 52)]

        if base.empty:
            st.warning("Não há valores válidos em dados/N_Semana (1..52).")
        else:
            especies = sorted([s for s in base[SPEC_COL].dropna().astype(str).unique() if s.strip() != ""])
            if not especies:
                st.warning("Não há espécies válidas para selecionar.")
            else:
                if "out_pa_local" not in st.session_state:
                    st.session_state.out_pa_local = "Total"
                if "out_pa_especie" not in st.session_state:
                    st.session_state.out_pa_especie = especies[0]

                if st.session_state.out_pa_especie not in especies:
                    st.session_state.out_pa_especie = especies[0]

                with st.form("form_pa", clear_on_submit=False):
                    colA, colB = st.columns(2)
                    with colA:
                        local_sel_pa = st.selectbox(
                            "Local",
                            locais_opts,
                            index=locais_opts.index(st.session_state.out_pa_local),
                            key="pa_local_form",
                        )
                    with colB:
                        especie_sel_pa = st.selectbox(
                            "Espécie",
                            especies,
                            index=especies.index(st.session_state.out_pa_especie),
                            key="pa_especie_form",
                        )
                    submitted_pa = st.form_submit_button("Aplicar")
                    if submitted_pa:
                        st.session_state.out_pa_local = local_sel_pa
                        st.session_state.out_pa_especie = especie_sel_pa

                local_sel = st.session_state.out_pa_local
                especie_sel = st.session_state.out_pa_especie

                work = base[base[SPEC_COL] == especie_sel].copy()
                if local_sel != "Total":
                    work = work[work[LOCAL_COL] == local_sel]

                work["Mes"] = ((work[WEEK_COL] - 1) // 4 + 1).astype(int)
                work.loc[work["Mes"] > 12, "Mes"] = 12
                work["SemanaMes"] = ((work[WEEK_COL] - 1) % 4 + 1).astype(int)

                presentes = set(zip(work["Mes"].tolist(), work["SemanaMes"].tolist()))

                meses_nome = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
                labels, colors, thetas = [], [], []

                n_segments = 48
                step = 360 / n_segments
                orange = "#00F715"
                gray = "#E0E0E0"

                idx = 0
                for m in range(1, 13):
                    for w in range(1, 5):
                        idx += 1
                        thetas.append((idx - 1) * step + step / 2)
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

                fig.update_layout(
                    title=f"Presença/Ausência — {especie_sel} ({local_sel})",
                    polar=dict(
                        radialaxis=dict(visible=False, range=[0, 1.00]),
                        angularaxis=dict(
                            visible=True,
                            tickmode="array",
                            tickvals=month_centers,
                            ticktext=meses_nome,
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

                st.plotly_chart(fig, width="stretch")
                #st.caption("Laranja = há registo nessa semana (dados/N_Semana) • Cinzento = sem registos")

elif section == "🧩 Matriz Presença":
    st.subheader("🧩 Matriz Presença")

    if df_amostras.empty or any(c not in df_amostras.columns for c in [LOCAL_COL, SPEC_COL]):
        st.info("Faltam colunas para construir a matriz (dados/Local, Amostragem/Espécie_final).")
    else:
        base = df_amostras.copy()

        # Normalizações de segurança
        base[LOCAL_COL] = base[LOCAL_COL].fillna("").astype(str).str.strip()
        base[SPEC_COL] = base[SPEC_COL].fillna("").astype(str).str.strip()

        # --------
        # MÊS (preferir dados/Data; fallback via dados/N_Semana)
        # --------
        month_col = "__Mes"

        if "dados/Data" in base.columns:
            dt = pd.to_datetime(base["dados/Data"], errors="coerce", dayfirst=True, format="mixed")
            base[month_col] = dt.dt.month
        elif WEEK_COL in base.columns:
            w = pd.to_numeric(base[WEEK_COL], errors="coerce")
            # mesma lógica que já usas: 1..52 -> mês aproximado (4 semanas)
            m = ((w - 1) // 4 + 1)
            m = m.where(m <= 12, 12)
            base[month_col] = m
        else:
            base[month_col] = pd.NA

        # manter só registos válidos (local + espécie)
        base = base[(base[LOCAL_COL] != "") & (base[SPEC_COL] != "")].copy()

        if base.empty:
            st.warning("Sem registos válidos (local e/ou espécie vazios).")
        else:
            # Listas para filtros (com "Todos")
            locais_all = sorted([x for x in base[LOCAL_COL].unique() if x.strip() != ""])
            especies_all = sorted([x for x in base[SPEC_COL].unique() if x.strip() != ""])

            meses_nome = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
            meses_all = [m for m in sorted(base[month_col].dropna().unique()) if 1 <= int(m) <= 12]
            meses_labels = [meses_nome[int(m) - 1] for m in meses_all]

            # estado no session_state
            if "mx_locais" not in st.session_state:
                st.session_state.mx_locais = ["Todos"]
            if "mx_especies" not in st.session_state:
                st.session_state.mx_especies = ["Todos"]
            if "mx_meses" not in st.session_state:
                st.session_state.mx_meses = ["Todos"]

            with st.form("form_matriz_presenca", clear_on_submit=False):
                c1, c2, c3 = st.columns(3)

                with c1:
                    locais_opts = ["Todos"] + locais_all
                    sel_locais = st.multiselect(
                        "Locais",
                        options=locais_opts,
                        default=st.session_state.mx_locais if isinstance(st.session_state.mx_locais, list) else ["Todos"],
                    )

                with c2:
                    especies_opts = ["Todos"] + especies_all
                    sel_especies = st.multiselect(
                        "Espécies",
                        options=especies_opts,
                        default=st.session_state.mx_especies if isinstance(st.session_state.mx_especies, list) else ["Todos"],
                    )

                with c3:
                    meses_opts = ["Todos"] + meses_labels  # mostramos nomes
                    sel_meses = st.multiselect(
                        "Meses",
                        options=meses_opts,
                        default=st.session_state.mx_meses if isinstance(st.session_state.mx_meses, list) else ["Todos"],
                    )

                apply_mx = st.form_submit_button("Aplicar filtros")
                if apply_mx:
                    st.session_state.mx_locais = sel_locais if sel_locais else ["Todos"]
                    st.session_state.mx_especies = sel_especies if sel_especies else ["Todos"]
                    st.session_state.mx_meses = sel_meses if sel_meses else ["Todos"]

            # aplicar filtros
            work = base.copy()

            # Locais
            sel_locais = st.session_state.mx_locais
            if sel_locais and "Todos" not in sel_locais:
                work = work[work[LOCAL_COL].isin(sel_locais)]

            # Espécies
            sel_especies = st.session_state.mx_especies
            if sel_especies and "Todos" not in sel_especies:
                work = work[work[SPEC_COL].isin(sel_especies)]

            # Meses (convertendo labels -> número)
            sel_meses = st.session_state.mx_meses
            if sel_meses and "Todos" not in sel_meses:
                label_to_num = {meses_nome[i]: i + 1 for i in range(12)}
                meses_num = [label_to_num[m] for m in sel_meses if m in label_to_num]
                work = work[work[month_col].isin(meses_num)]

            if work.empty:
                st.warning("Sem registos após aplicar filtros.")
            else:
                # --------
                # MATRIZ PRESENÇA (Espécie x Local)
                # --------
                pres = (
                    work.groupby([SPEC_COL, LOCAL_COL])
                    .size()
                    .reset_index(name="n")
                )

                matrix_bool = (
                    pres.assign(presente=True)
                        .pivot_table(index=SPEC_COL, columns=LOCAL_COL, values="presente", aggfunc="max", fill_value=False)
                        .sort_index(axis=0)
                        .sort_index(axis=1)
                )

                # converter True/False em check verde (✖)
                matrix_display = matrix_bool.applymap(lambda v: "✖" if bool(v) else "")


                #st.caption("✅ = espécie registada nesse local (com os filtros atuais).")
                st.dataframe(matrix_display, width="stretch", height=650)


                # Export Excel da matriz
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                    matrix_display.reset_index().rename(columns={SPEC_COL: "Espécie"}).to_excel(
                        writer, index=False, sheet_name="Matriz_Presenca"
                    )
                buffer.seek(0)

                st.download_button(
                    "⬇️ Exportar Excel (matriz)",
                    data=buffer,
                    file_name="matriz_presenca_especie_local.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                pdf_bytes = build_matrix_pdf(
                    title="Matriz Presença — Espécie x Local",
                    matrix_df=matrix_display
                )
                
                st.download_button(
                    "⬇️ Download PDF (matriz)",
                    data=pdf_bytes,
                    file_name=f"matriz_presenca_especie_local_{date.today().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    key="download_pdf_matriz_presenca",
                )
                

elif section == "📋 Tabela":
    st.subheader("📋 Tabela")

    df_base = df_amostras if not df_amostras.empty else df

    if "table_ui_state" not in st.session_state:
        st.session_state.table_ui_state = {}
    if "table_filters_applied" not in st.session_state:
        st.session_state.table_filters_applied = False

    with st.expander("🧩 Abrir filtros", expanded=True):
        with st.form("table_filters_form", clear_on_submit=False):
            cols_to_filter = [
                "dados/N_Semana",
                "dados/Data",
                "dados/Local",
                "Amostragem/Espécie_final",
            ]
            cols_to_filter = [c for c in cols_to_filter if c in df_base.columns]

            ui_state = {}

            ncols = 3
            rows = [cols_to_filter[i:i + ncols] for i in range(0, len(cols_to_filter), ncols)]

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
                                prev = st.session_state.table_ui_state.get(colname, {}).get("value")
                                default_rng = prev if (isinstance(prev, (tuple, list)) and len(prev) == 2) else (minv, maxv)
                                rng = st.slider(
                                    colname,
                                    min_value=minv,
                                    max_value=maxv,
                                    value=(float(default_rng[0]), float(default_rng[1])),
                                )
                                ui_state[colname] = {"type": "numeric", "value": rng}
                    else:
                        dt = pd.to_datetime(s, errors="coerce", dayfirst=True, format="mixed")
                        valid = dt.dropna()

                        if len(valid) >= max(10, int(0.2 * len(s.dropna()))) and not valid.empty:
                            mind = valid.min().date()
                            maxd = valid.max().date()

                            with cols[i]:
                                prev = st.session_state.table_ui_state.get(colname, {}).get("value")
                                default_dr = prev if (isinstance(prev, tuple) and len(prev) == 2) else (mind, maxd)
                                dr = st.date_input(colname, value=default_dr)
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
                                    prev = st.session_state.table_ui_state.get(colname, {}).get("value")
                                    default_sel = prev if isinstance(prev, list) else []
                                    sel = st.multiselect(colname, options=options, default=default_sel)
                                    ui_state[colname] = {"type": "categorical", "value": sel}
                                else:
                                    prev = st.session_state.table_ui_state.get(colname, {}).get("value")
                                    default_txt = prev if isinstance(prev, str) else ""
                                    txt = st.text_input(f"{colname} (contains)", value=default_txt)
                                    ui_state[colname] = {"type": "text", "value": txt}

            cA, cB = st.columns([1, 1])
            with cA:
                apply_btn = st.form_submit_button("Aplicar filtros")
            with cB:
                clear_btn = st.form_submit_button("Limpar filtros")

            if clear_btn:
                st.session_state.table_ui_state = {}
                st.session_state.table_filters_applied = False
            elif apply_btn:
                st.session_state.table_ui_state = ui_state
                st.session_state.table_filters_applied = True

    if st.session_state.table_filters_applied and st.session_state.table_ui_state:
        filtered = apply_filters(df_base, st.session_state.table_ui_state)
    else:
        filtered = df_base.copy()

    all_cols = list(filtered.columns)
    prefer = [
        "_id", "dados/Data", "dados/Hora", "dados/N_Semana", "dados/Local",
        "Amostragem/Espécie", "Amostragem/Outra_Esp_cie", "Amostragem/Espécie_final",
        "Amostragem/N_Indiv_duos", "Amostragem/Notas"
    ]
    default_show = [c for c in prefer if c in all_cols]
    if len(default_show) < 6:
        default_show = all_cols[:20] if len(all_cols) >= 20 else all_cols

    show_cols = st.multiselect("Colunas visíveis", options=all_cols, default=default_show, key="table_show_cols")

    st.dataframe(filtered[show_cols], width="stretch", height=560)

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
