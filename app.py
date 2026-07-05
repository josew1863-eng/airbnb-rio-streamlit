"""
Análise Espacial — Airbnb Rio de Janeiro
App Streamlit com 3 abas: Mapa Dinâmico, Sazonalidade e Correlações.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px
import folium
from folium import FeatureGroup, LayerControl
from branca.colormap import linear
from streamlit_folium import st_folium
from scipy.spatial import cKDTree

st.set_page_config(page_title="Airbnb Rio de Janeiro  Análise Espacial", layout="wide")

PASTA_DADOS = os.path.join(os.path.dirname(__file__), "dados")
NOMES_MES = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
             7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}


# ---------------------------------------------------------------------------
# Carregamento e preparação dos dados (cacheado — só roda 1x por sessão)
# ---------------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    df = pd.read_parquet(os.path.join(PASTA_DADOS, "dataset_features_slim.parquet"))
    gdf = gpd.read_file(os.path.join(PASTA_DADOS, "neighbourhoods.geojson"))
    calendario = pd.read_parquet(os.path.join(PASTA_DADOS, "calendar_agregado.parquet"))
    temporal = pd.read_parquet(os.path.join(PASTA_DADOS, "listings_temporal.parquet"))
    return df, gdf, calendario, temporal


@st.cache_data
def agregar_por_bairro(df):
    agg = df.groupby("bairro_padronizado").agg(
        qtd_anuncios=("id_anuncio", "count"),
        score_luxo_medio=("score_luxo", "mean"),
        faixa_preco_moda=("faixa_preco", lambda s: s.mode().iat[0] if not s.mode().empty else None),
        preco_medio=("preco", "mean"),
        rentabilidade_media=("rentabilidade_diaria", "mean"),
        taxa_ocupacao_media=("taxa_ocupacao_estimada", "mean"),
        demanda_media=("demanda_bairro", "mean"),
        regiao_popular=("regiao_popular", lambda s: s.mode().iat[0] if not s.mode().empty else 0),
        lat=("latitude", "mean"),
        lon=("longitude", "mean"),
    ).reset_index()
    return agg


@st.cache_data(show_spinner="Buscando pontos turísticos no OpenStreetMap...")
def buscar_pontos_turisticos():
    try:
        import osmnx as ox
        tags = {
            "tourism": ["attraction", "viewpoint", "museum"],
            "natural": "beach",
            "historic": ["monument", "memorial"],
        }
        pontos = ox.features_from_place("Rio de Janeiro, Brazil", tags)
        pontos = pontos.assign(geometry=pontos.geometry.centroid)
        pontos = pontos.assign(lat=pontos.geometry.y, lon=pontos.geometry.x, nome=pontos.get("name"))
        pontos = pontos[["nome", "lat", "lon"]].dropna().drop_duplicates("nome").reset_index(drop=True)
        return pontos
    except Exception:
        return pd.DataFrame(columns=["nome", "lat", "lon"])


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_data
def adicionar_distancia_turistica(agg, pontos):
    agg = agg.copy()
    if pontos.empty:
        agg["distancia_ponto_turistico_km"] = np.nan
        agg["ponto_turistico_proximo"] = None
        return agg
    arvore = cKDTree(pontos[["lat", "lon"]].values)
    _, idx = arvore.query(agg[["lat", "lon"]].values, k=1)
    agg["ponto_turistico_proximo"] = pontos.loc[idx, "nome"].values
    agg["distancia_ponto_turistico_km"] = [
        haversine_km(r.lat, r.lon, pontos.loc[i, "lat"], pontos.loc[i, "lon"])
        for r, i in zip(agg.itertuples(), idx)
    ]
    return agg


# ---------------------------------------------------------------------------
# Carrega tudo
# ---------------------------------------------------------------------------
df, gdf, calendario, temporal = carregar_dados()
agg = agregar_por_bairro(df)
pontos_turisticos = buscar_pontos_turisticos()
agg = adicionar_distancia_turistica(agg, pontos_turisticos)
mapa_gdf = gdf.merge(agg, left_on="neighbourhood", right_on="bairro_padronizado", how="left")

st.title("🏖️ Análise Espacial — Airbnb Rio de Janeiro")
st.caption("Preço, ocupação, luxo, turismo e sazonalidade dos anúncios por bairro.")

aba_mapa, aba_sazonalidade = st.tabs(
    ["🗺️ Mapa Dinâmico", "📅 Sazonalidade"]
)

# ---------------------------------------------------------------------------
# ABA 1 — MAPA DINÂMICO
# ---------------------------------------------------------------------------
with aba_mapa:
    st.subheader("Mapa por bairro: luxo, rentabilidade, ocupação e turismo")

    col_filtros, col_mapa = st.columns([1, 3])

    with col_filtros:
        preco_min, preco_max = float(agg["preco_medio"].min()), float(agg["preco_medio"].max())
        faixa_preco = st.slider(
            "Faixa de preço médio do bairro (R$)",
            min_value=float(np.floor(preco_min)),
            max_value=float(np.ceil(preco_max)),
            value=(float(np.floor(preco_min)), float(np.ceil(preco_max))),
        )
        mostrar_populares = st.checkbox("Mostrar bairros de região popular", value=True)
        mostrar_nao_populares = st.checkbox("Mostrar bairros de região não popular", value=True)
        mostrar_turismo = st.checkbox("Mostrar pontos turísticos", value=True)

        st.metric("Bairros no filtro atual",
                   int(agg[(agg["preco_medio"] >= faixa_preco[0]) & (agg["preco_medio"] <= faixa_preco[1])].shape[0]))

    agg_filtrado = agg[(agg["preco_medio"] >= faixa_preco[0]) & (agg["preco_medio"] <= faixa_preco[1])]
    if not mostrar_populares:
        agg_filtrado = agg_filtrado[agg_filtrado["regiao_popular"] != 1]
    if not mostrar_nao_populares:
        agg_filtrado = agg_filtrado[agg_filtrado["regiao_popular"] != 0]

    mapa_gdf_filtrado = mapa_gdf[mapa_gdf["bairro_padronizado"].isin(agg_filtrado["bairro_padronizado"])]

    with col_mapa:
        m = folium.Map(location=[-22.925, -43.30], zoom_start=11, tiles="CartoDB positron")

        if len(mapa_gdf_filtrado) > 0:
            colormap_luxo = linear.YlOrRd_09.scale(
                agg["score_luxo_medio"].min(), agg["score_luxo_medio"].max()
            )
            colormap_luxo.caption = "Score de Luxo médio por bairro"

            def estilo_bairro(feature):
                valor = feature["properties"].get("score_luxo_medio")
                return {
                    "fillColor": colormap_luxo(valor) if valor is not None else "#cccccc",
                    "color": "#555555",
                    "weight": 0.8,
                    "fillOpacity": 0.65,
                }

            tooltip_fields = ["neighbourhood", "qtd_anuncios", "score_luxo_medio",
                               "faixa_preco_moda", "preco_medio", "rentabilidade_media",
                               "taxa_ocupacao_media", "distancia_ponto_turistico_km"]
            tooltip_aliases = ["Bairro:", "Nº anúncios:", "Score luxo:",
                                "Faixa de preço:", "Preço médio (R$):", "Rentabilidade média:",
                                "Taxa ocupação média:", "Dist. ponto turístico (km):"]

            folium.GeoJson(
                mapa_gdf_filtrado,
                name="Score de Luxo por bairro",
                style_function=estilo_bairro,
                tooltip=folium.GeoJsonTooltip(fields=tooltip_fields, aliases=tooltip_aliases,
                                                localize=True, sticky=True),
            ).add_to(m)
            colormap_luxo.add_to(m)

            colormap_ocup = linear.PuBuGn_09.scale(
                agg["taxa_ocupacao_media"].min(), agg["taxa_ocupacao_media"].max()
            )
            grupo_popular = FeatureGroup(name="Bairros — região popular", show=True)
            grupo_nao_popular = FeatureGroup(name="Bairros — região não popular", show=True)
            max_rent = agg["rentabilidade_media"].max()

            for _, row in agg_filtrado.dropna(subset=["lat", "lon"]).iterrows():
                raio = 4 + 20 * (row["rentabilidade_media"] / max_rent if max_rent else 0)
                popup_html = (
                    f"<b>{row['bairro_padronizado']}</b><br>"
                    f"Anúncios: {int(row['qtd_anuncios'])}<br>"
                    f"Score luxo: {row['score_luxo_medio']:.2f}<br>"
                    f"Faixa preço: {row['faixa_preco_moda']}<br>"
                    f"Preço médio: R$ {row['preco_medio']:.0f}<br>"
                    f"Rentabilidade média: {row['rentabilidade_media']:.2f}<br>"
                    f"Taxa ocupação média: {row['taxa_ocupacao_media']:.2%}"
                )
                marker = folium.CircleMarker(
                    location=[row["lat"], row["lon"]],
                    radius=raio,
                    color=colormap_ocup(row["taxa_ocupacao_media"]),
                    fill=True,
                    fill_color=colormap_ocup(row["taxa_ocupacao_media"]),
                    fill_opacity=0.75,
                    weight=1,
                    popup=folium.Popup(popup_html, max_width=250),
                )
                destino = grupo_popular if row["regiao_popular"] == 1 else grupo_nao_popular
                marker.add_to(destino)

            grupo_popular.add_to(m)
            grupo_nao_popular.add_to(m)

            if mostrar_turismo and not pontos_turisticos.empty:
                grupo_turismo = FeatureGroup(name="Pontos turísticos", show=True)
                for _, pt in pontos_turisticos.iterrows():
                    folium.Marker(
                        location=[pt["lat"], pt["lon"]],
                        popup=pt["nome"],
                        icon=folium.Icon(color="cadetblue", icon="star", prefix="fa"),
                    ).add_to(grupo_turismo)
                grupo_turismo.add_to(m)

            LayerControl(collapsed=False).add_to(m)
        else:
            st.warning("Nenhum bairro dentro do filtro selecionado.")

        st_folium(m, width=None, height=650, returned_objects=[])

# ---------------------------------------------------------------------------
# ABA 2 — SAZONALIDADE
# ---------------------------------------------------------------------------
with aba_sazonalidade:
    st.subheader("Ocupação e preço ao longo do ano")

    mensal = calendario[calendario["granularidade"] == "mensal"].copy()
    media_mes = mensal.groupby("mes")["taxa_ocupacao"].mean().reset_index()
    media_mes["mes_nome"] = media_mes["mes"].map(NOMES_MES)
    media_mes = media_mes.sort_values("mes")

    col1, col2 = st.columns(2)

    with col1:
        fig_ocup = px.line(media_mes, x="mes_nome", y="taxa_ocupacao", markers=True,
                            title="Taxa de ocupação média por mês (todas as coletas)")
        fig_ocup.update_layout(yaxis_tickformat=".0%", xaxis_title="Mês", yaxis_title="Taxa de ocupação")
        mes_pico = media_mes.loc[media_mes["taxa_ocupacao"].idxmax()]
        mes_baixa = media_mes.loc[media_mes["taxa_ocupacao"].idxmin()]
        fig_ocup.add_annotation(x=mes_pico["mes_nome"], y=mes_pico["taxa_ocupacao"],
                                 text="Alta temporada", showarrow=True, arrowhead=2, yshift=15)
        fig_ocup.add_annotation(x=mes_baixa["mes_nome"], y=mes_baixa["taxa_ocupacao"],
                                 text="Baixa temporada", showarrow=True, arrowhead=2, yshift=-15)
        st.plotly_chart(fig_ocup, width='stretch')

    with col2:
        preco_mes = temporal.groupby("mes_coleta")["price"].mean().reset_index()
        preco_mes["mes_nome"] = preco_mes["mes_coleta"].map(NOMES_MES)
        preco_mes = preco_mes.sort_values("mes_coleta")

        fig_preco = px.bar(preco_mes, x="mes_nome", y="price",
                            title="Preço médio por mês (coletas disponíveis)")
        fig_preco.update_layout(xaxis_title="Mês", yaxis_title="Preço médio (R$)")
        st.plotly_chart(fig_preco, width='stretch')

    st.info(
        f"📈 **Melhor mês pra alugar/investir (maior ocupação): {mes_pico['mes_nome']}** "
        f"({mes_pico['taxa_ocupacao']:.1%} de ocupação média)\n\n"
        f"📉 **Melhor mês pra economizar viajando (menor ocupação): {mes_baixa['mes_nome']}** "
        f"({mes_baixa['taxa_ocupacao']:.1%} de ocupação média)"
    )

    st.caption(
        "Nota: o preço por mês só está disponível para os meses efetivamente coletados "
        f"({', '.join(preco_mes['mes_nome'])}). "
        "A ocupação mensal cobre o ano cheio, pois vem do calendário futuro de disponibilidade."
    )