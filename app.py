"""
Análise Estatística — Airbnb Rio de Janeiro
App Streamlit com abas: Mapa Dinâmico, Sazonalidade, Evolução Histórica de Preços,
Análise Descritiva, Análise Fatorial e Modelagem Preditiva.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import FeatureGroup, LayerControl
from branca.colormap import linear
from streamlit_folium import st_folium
from scipy.spatial import cKDTree

# --- Bibliotecas de análise fatorial e modelagem (opcionais, com fallback) -------
try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    FACTOR_ANALYZER_DISPONIVEL = True
except ImportError:
    FACTOR_ANALYZER_DISPONIVEL = False

try:
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import RidgeCV, LassoCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    MODELAGEM_DISPONIVEL = True
except ImportError:
    MODELAGEM_DISPONIVEL = False

try:
    import joblib
    JOBLIB_DISPONIVEL = True
except ImportError:
    JOBLIB_DISPONIVEL = False

st.set_page_config(page_title="Airbnb Rio de Janeiro  Análise Espacial", layout="wide")

PASTA_DADOS = os.path.join(os.path.dirname(__file__), "dados")
NOMES_MES = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
             7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}


# ---------------------------------------------------------------------------
# Carregamento e preparação dos dados (cacheado — só roda 1x por sessão)
# ---------------------------------------------------------------------------
@st.cache_data
def carregar_dados():
    df = pd.read_parquet(os.path.join(PASTA_DADOS, "dataset_features.parquet"))
    gdf = gpd.read_file(os.path.join(PASTA_DADOS, "bairros_estatistica_espacial.geojson"))
    # o geojson já traz um preco_medio (calculado na análise LISA); descartamos aqui
    # para que o preco_medio recalculado em agregar_por_bairro() prevaleça sem sufixo _x/_y
    gdf = gdf.drop(columns=["preco_medio"], errors="ignore")
    try:
        calendario = pd.read_parquet(os.path.join(PASTA_DADOS, "calendar_agregado.parquet"))
    except FileNotFoundError:
        calendario = None
    try:
        temporal = pd.read_parquet(os.path.join(PASTA_DADOS, "listings_temporal.parquet"))
    except FileNotFoundError:
        temporal = None
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
# Modelos preditivos (Simulador de Investimento)
# Gerados pelo notebook 05-1_modelagem_preditiva.ipynb (pasta outputs/modelos/)
# ---------------------------------------------------------------------------
PASTA_MODELOS = os.path.join(os.path.dirname(__file__), "modelos")


@st.cache_resource
def carregar_modelos():
    if not JOBLIB_DISPONIVEL:
        return None
    caminho_preco = os.path.join(PASTA_MODELOS, "modelo_preco.joblib")
    caminho_ocup = os.path.join(PASTA_MODELOS, "modelo_ocupacao.joblib")
    caminho_config = os.path.join(PASTA_MODELOS, "config_preditores.joblib")
    if not (os.path.exists(caminho_preco) and os.path.exists(caminho_ocup) and os.path.exists(caminho_config)):
        return None
    config = joblib.load(caminho_config)
    return {
        "modelo_preco": joblib.load(caminho_preco),
        "modelo_ocupacao": joblib.load(caminho_ocup),
        **config,  # preditores_num_preco, preditores_cat_preco, preditores_num_ocup,
                   # preditores_cat_ocup, multiplicador_sazonal
    }


def prever_investimento(modelos: dict, dados_imovel: dict, mes_referencia: str = None) -> dict:
    """Réplica da função definida em 05-1_modelagem_preditiva.ipynb (célula 'Composição
    da rentabilidade estimada'). Recebe as características do imóvel e devolve preço
    de diária previsto, taxa de ocupação prevista e rentabilidade anual estimada, além
    da margem de erro dos modelos (se o `config_preditores.joblib` já tiver essas métricas —
    versões antigas do arquivo não têm, e o app cai de volta para "sem margem" nesse caso).
    """
    preditores_preco = modelos["preditores_num_preco"] + modelos["preditores_cat_preco"]
    preditores_ocup = modelos["preditores_num_ocup"] + modelos["preditores_cat_ocup"]

    linha_preco = pd.DataFrame([{k: dados_imovel.get(k) for k in preditores_preco}])
    linha_ocup = pd.DataFrame([{k: dados_imovel.get(k) for k in preditores_ocup}])

    preco_previsto = float(np.expm1(modelos["modelo_preco"].predict(linha_preco)[0]))
    ocupacao_prevista = float(modelos["modelo_ocupacao"].predict(linha_ocup)[0])
    ocupacao_prevista = min(max(ocupacao_prevista, 0.0), 1.0)  # limita entre 0% e 100%

    rentabilidade_anual = preco_previsto * ocupacao_prevista * 365
    fator_sazonal = 1.0
    multiplicador_sazonal = modelos.get("multiplicador_sazonal", {})
    if mes_referencia and mes_referencia in multiplicador_sazonal:
        fator_sazonal = multiplicador_sazonal[mes_referencia]
        rentabilidade_anual *= fator_sazonal

    resultado = {
        "preco_diaria_previsto": round(preco_previsto, 2),
        "taxa_ocupacao_prevista": round(ocupacao_prevista, 4),
        "noites_ocupadas_ano_estimadas": round(ocupacao_prevista * 365, 1),
        "rentabilidade_anual_estimada": round(rentabilidade_anual, 2),
        "fator_sazonal_aplicado": round(fator_sazonal, 3),
        "tem_margem_erro": False,
    }

    erro_preco = modelos.get("erro_preco_reais")
    erro_ocup = modelos.get("erro_ocupacao")
    if erro_preco is not None and erro_ocup is not None:
        preco_min = max(preco_previsto - erro_preco, 0.0)
        preco_max = preco_previsto + erro_preco
        ocup_min = min(max(ocupacao_prevista - erro_ocup, 0.0), 1.0)
        ocup_max = min(max(ocupacao_prevista + erro_ocup, 0.0), 1.0)
        rent_min = preco_min * ocup_min * 365 * fator_sazonal
        rent_max = preco_max * ocup_max * 365 * fator_sazonal
        resultado.update({
            "tem_margem_erro": True,
            "preco_erro_abs": round(erro_preco, 2),
            "ocupacao_erro_abs": round(erro_ocup, 4),
            "preco_intervalo": (round(preco_min, 2), round(preco_max, 2)),
            "ocupacao_intervalo": (round(ocup_min, 4), round(ocup_max, 4)),
            "rentabilidade_intervalo": (round(rent_min, 2), round(rent_max, 2)),
            "r2_preco": modelos.get("r2_preco"),
            "r2_ocupacao": modelos.get("r2_ocupacao"),
        })
    return resultado





# ---------------------------------------------------------------------------
# Perfis predefinidos do simulador (Econômico / Padrão / Luxo)
# Calculados a partir dos dados reais (tercis de `score_luxo`), não são valores
# "chutados" — são a mediana/moda dos imóveis que caem em cada faixa de luxo.
# ---------------------------------------------------------------------------
CAMPOS_NUM_PERFIL = [
    "capacidade_hospedes", "banheiros", "quartos", "camas", "qtd_comodidades",
    "nota_composta", "noites_minimas", "noites_maximas", "total_anuncios_anfitriao",
    "numero_avaliacoes", "avaliacoes_por_mes",
]
CAMPOS_CAT_PERFIL = [
    "tipo_quarto", "tipo_propriedade", "tipo_hospedagem", "e_superanfitriao", "reserva_instantanea",
]
CAMPOS_BINARIOS_PERFIL = ["tem_banheiro_privativo", "flexibilidade_estadia"]
# Campos cujo widget no formulário é int (min_value/max_value inteiros) — a mediana
# precisa ser arredondada e convertida, senão o Streamlit reclama de tipo (int vs float).
# `quartos`/`camas` entraram aqui porque não faz sentido ter "2,5 quartos" — só
# `banheiros` fica de fora, pois meio-banheiro (lavabo, sem chuveiro) é uma categoria
# real nos dados do Airbnb.
CAMPOS_INT_PERFIL = {
    "capacidade_hospedes", "quartos", "camas", "qtd_comodidades", "noites_minimas",
    "noites_maximas", "total_anuncios_anfitriao", "numero_avaliacoes",
}


@st.cache_data
def calcular_perfis_predefinidos(df: pd.DataFrame) -> dict:
    if "score_luxo" not in df.columns:
        return {}

    tercil_1, tercil_2 = df["score_luxo"].quantile([0.33, 0.66]).values
    faixas = {
        "Econômico": df["score_luxo"] <= tercil_1,
        "Padrão": (df["score_luxo"] > tercil_1) & (df["score_luxo"] <= tercil_2),
        "Luxo": df["score_luxo"] > tercil_2,
    }

    perfis = {}
    for nome_perfil, mascara in faixas.items():
        sub = df[mascara]
        if sub.empty:
            continue
        perfil = {}
        for c in CAMPOS_NUM_PERFIL:
            if c in sub.columns:
                mediana = sub[c].median()
                perfil[c] = int(round(mediana)) if c in CAMPOS_INT_PERFIL else float(mediana)
        for c in CAMPOS_CAT_PERFIL:
            if c in sub.columns and not sub[c].mode().empty:
                perfil[c] = sub[c].mode().iat[0]
        for c in CAMPOS_BINARIOS_PERFIL:
            if c in sub.columns:
                perfil[c] = int(round(sub[c].median()))
        perfis[nome_perfil] = perfil
    return perfis


@st.cache_data(show_spinner=False)
def calcular_previsoes_dataset(df: pd.DataFrame, _modelos: dict) -> pd.DataFrame:
    """Roda os modelos de preço e ocupação em LOTE sobre todos os imóveis reais do
    dataset (não linha a linha), para alimentar a busca do 'imóvel ideal'. O underscore
    em `_modelos` evita que o Streamlit tente fazer hash dos objetos do modelo (não
    hasheáveis) — só o `df` é usado como chave do cache."""
    preditores_preco = _modelos["preditores_num_preco"] + _modelos["preditores_cat_preco"]
    preditores_ocup = _modelos["preditores_num_ocup"] + _modelos["preditores_cat_ocup"]
    colunas_necessarias = [c for c in set(preditores_preco + preditores_ocup) if c in df.columns]

    base = df.dropna(subset=colunas_necessarias).copy()

    preco_previsto = np.expm1(_modelos["modelo_preco"].predict(base[preditores_preco]))
    ocupacao_prevista = _modelos["modelo_ocupacao"].predict(base[preditores_ocup])
    ocupacao_prevista = np.clip(ocupacao_prevista, 0.0, 1.0)

    base["preco_previsto"] = preco_previsto
    base["ocupacao_prevista"] = ocupacao_prevista
    base["rentabilidade_anual_prevista"] = preco_previsto * ocupacao_prevista * 365
    return base


def aplicar_perfil(perfil: dict):
    """Callback: copia os valores do perfil escolhido para o session_state,
    fazendo os widgets do formulário assumirem esses valores no próximo rerun."""
    mapa_chaves = {
        "capacidade_hospedes": "sim_capacidade_hospedes", "banheiros": "sim_banheiros",
        "quartos": "sim_quartos", "camas": "sim_camas", "qtd_comodidades": "sim_qtd_comodidades",
        "nota_composta": "sim_nota_composta", "noites_minimas": "sim_noites_minimas",
        "noites_maximas": "sim_noites_maximas", "total_anuncios_anfitriao": "sim_total_anuncios",
        "numero_avaliacoes": "sim_numero_avaliacoes", "avaliacoes_por_mes": "sim_avaliacoes_por_mes",
        "tipo_quarto": "sim_tipo_quarto", "tipo_propriedade": "sim_tipo_propriedade",
        "tipo_hospedagem": "sim_tipo_hospedagem", "e_superanfitriao": "sim_e_superanfitriao",
        "reserva_instantanea": "sim_reserva_instantanea",
        "tem_banheiro_privativo": "sim_banheiro_privativo", "flexibilidade_estadia": "sim_flexibilidade",
    }
    for campo, chave_widget in mapa_chaves.items():
        if campo in perfil:
            valor = perfil[campo]
            if campo in ("tem_banheiro_privativo", "flexibilidade_estadia"):
                valor = "Sim" if valor else "Não"
            st.session_state[chave_widget] = valor


# ---------------------------------------------------------------------------
# Carrega tudo
# ---------------------------------------------------------------------------
df, gdf, calendario, temporal = carregar_dados()
agg = agregar_por_bairro(df)
pontos_turisticos = buscar_pontos_turisticos()
agg = adicionar_distancia_turistica(agg, pontos_turisticos)
mapa_gdf = gdf.merge(agg, left_on="neighbourhood", right_on="bairro_padronizado", how="left")

st.title(" Análise Espacial  Airbnb Rio de Janeiro")
st.caption("Preço, ocupação, luxo, turismo e sazonalidade dos anúncios por bairro.")

aba_mapa, aba_sazonalidade, aba_evolucao, aba_simulador = st.tabs(
    ["🗺️ Mapa Dinâmico", "📅 Sazonalidade", "📈 Evolução Histórica de Preços", "🧮 Simulador de Investimento"]
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
            grupo_popular = FeatureGroup(name="Bairros de região popular", show=True)
            grupo_nao_popular = FeatureGroup(name="Bairros de região não popular", show=True)
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
    if calendario is None or temporal is None:
        st.warning(
            "Esta aba precisa dos arquivos `calendar_agregado.parquet` e `listings_temporal.parquet` "
            "(calendário de disponibilidade mês a mês e múltiplas coletas), que não foram encontrados "
            "na pasta de dados. Coloque esses arquivos em `dados/` para habilitar esta análise."
        )
    else:
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

# ---------------------------------------------------------------------------
# ABA 3 — EVOLUÇÃO HISTÓRICA DE PREÇOS
# ---------------------------------------------------------------------------
with aba_evolucao:
    if temporal is None:
        st.warning(
            "Esta aba precisa do arquivo `listings_temporal.parquet` (múltiplas coletas do "
            "Airbnb ao longo do tempo), que não foi encontrado na pasta de dados. "
            "Coloque esse arquivo em `dados/` para habilitar esta análise."
        )
    else:
        st.subheader("Evolução do preço da diária entre coletas")
        st.caption(
            "Cada coleta é uma foto do mercado num momento diferente — não é o preço de um "
            "mesmo anúncio ao longo do tempo, mas sim como o conjunto de anúncios ativos se "
            "comportava em cada data de coleta."
        )

        coluna_tempo = "mes_coleta" if "mes_coleta" in temporal.columns else None
        coluna_preco_temp = "price" if "price" in temporal.columns else (
            "preco" if "preco" in temporal.columns else None
        )

        if coluna_tempo is None or coluna_preco_temp is None:
            st.warning("Não encontrei as colunas necessárias em `listings_temporal.parquet` para montar esta análise.")
        else:
            resumo_temporal = (
                temporal.groupby(coluna_tempo)[coluna_preco_temp]
                .agg(preco_medio="mean", preco_mediano="median", n_anuncios="count")
                .reset_index()
            )
            resumo_temporal["mes_nome"] = resumo_temporal[coluna_tempo].map(NOMES_MES).fillna(
                resumo_temporal[coluna_tempo].astype(str)
            )
            resumo_temporal = resumo_temporal.sort_values(coluna_tempo)
            resumo_temporal["variacao_pct_nominal"] = (
                resumo_temporal["preco_medio"] / resumo_temporal["preco_medio"].iloc[0] - 1
            ) * 100

            col_evo1, col_evo2 = st.columns(2)

            with col_evo1:
                fig_evo = make_subplots(specs=[[{"secondary_y": True}]])
                fig_evo.add_trace(
                    go.Scatter(x=resumo_temporal["mes_nome"], y=resumo_temporal["preco_medio"],
                               mode="lines+markers", name="Preço médio", line=dict(color="#1e3d59", width=3)),
                    secondary_y=False,
                )
                fig_evo.add_trace(
                    go.Scatter(x=resumo_temporal["mes_nome"], y=resumo_temporal["preco_mediano"],
                               mode="lines+markers", name="Preço mediano",
                               line=dict(color="#ff6e40", width=3, dash="dash")),
                    secondary_y=False,
                )
                fig_evo.add_trace(
                    go.Bar(x=resumo_temporal["mes_nome"], y=resumo_temporal["n_anuncios"],
                           name="Nº de anúncios", marker_color="#17b978", opacity=0.35),
                    secondary_y=True,
                )
                fig_evo.update_layout(title="Preço médio/mediano e volume de anúncios por coleta",
                                       xaxis_title="Coleta", legend=dict(orientation="h", y=-0.2))
                fig_evo.update_yaxes(title_text="Preço (R$)", secondary_y=False)
                fig_evo.update_yaxes(title_text="Nº de anúncios", secondary_y=True)
                st.plotly_chart(fig_evo, width="stretch")

            with col_evo2:
                fig_var = px.line(
                    resumo_temporal, x="mes_nome", y="variacao_pct_nominal", markers=True,
                    title="Variação percentual acumulada do preço médio (nominal)",
                )
                fig_var.add_hline(y=0, line_dash="dot", line_color="gray")
                fig_var.update_layout(xaxis_title="Coleta", yaxis_title="Variação acumulada (%)")
                fig_var.update_traces(line_color="#9B59B6")
                st.plotly_chart(fig_var, width="stretch")
                st.caption("Valores nominais, sem correção pela inflação (IPCA).")

            st.dataframe(
                resumo_temporal[["mes_nome", "preco_medio", "preco_mediano", "n_anuncios", "variacao_pct_nominal"]]
                .rename(columns={
                    "mes_nome": "Coleta", "preco_medio": "Preço médio (R$)",
                    "preco_mediano": "Preço mediano (R$)", "n_anuncios": "Nº anúncios",
                    "variacao_pct_nominal": "Variação acumulada (%)",
                }).round(2),
                width="stretch", hide_index=True,
            )

            st.divider()
            st.subheader("Evolução do preço médio por bairro")

            coluna_bairro_temp = next(
                (c for c in ["bairro_padronizado", "neighbourhood_cleansed", "neighbourhood"] if c in temporal.columns),
                None,
            )
            if coluna_bairro_temp is None:
                st.info("A base `listings_temporal.parquet` não possui coluna de bairro — não é possível detalhar a evolução por bairro.")
            else:
                ultima_coleta = temporal[coluna_tempo].max()
                top_bairros_default = (
                    temporal[temporal[coluna_tempo] == ultima_coleta]
                    .groupby(coluna_bairro_temp).size().sort_values(ascending=False).head(10).index.tolist()
                )
                todos_bairros = sorted(temporal[coluna_bairro_temp].dropna().unique().tolist())
                bairros_selecionados = st.multiselect(
                    "Bairros para comparar (padrão: top 10 com mais anúncios na coleta mais recente)",
                    options=todos_bairros, default=top_bairros_default,
                )

                if bairros_selecionados:
                    evolucao_bairro = (
                        temporal[temporal[coluna_bairro_temp].isin(bairros_selecionados)]
                        .groupby([coluna_tempo, coluna_bairro_temp])[coluna_preco_temp]
                        .mean().reset_index()
                    )
                    evolucao_bairro["mes_nome"] = evolucao_bairro[coluna_tempo].map(NOMES_MES).fillna(
                        evolucao_bairro[coluna_tempo].astype(str)
                    )
                    evolucao_bairro = evolucao_bairro.sort_values(coluna_tempo)

                    fig_bairro = px.line(
                        evolucao_bairro, x="mes_nome", y=coluna_preco_temp, color=coluna_bairro_temp,
                        markers=True, title="Evolução do preço médio por bairro selecionado",
                    )
                    fig_bairro.update_layout(xaxis_title="Coleta", yaxis_title="Preço médio (R$)",
                                              legend_title="Bairro")
                    st.plotly_chart(fig_bairro, width="stretch")
                else:
                    st.info("Selecione pelo menos um bairro para ver o gráfico.")

# ---------------------------------------------------------------------------
# ABA 4 — SIMULADOR DE INVESTIMENTO
# ---------------------------------------------------------------------------
with aba_simulador:
    st.subheader("Simulador de Preço, Ocupação e Rentabilidade")
    st.caption(
        "Informe as características de um imóvel e receba uma estimativa de preço de diária, "
        "taxa de ocupação e rentabilidade anual."
    )

    modelos = carregar_modelos()

    if not JOBLIB_DISPONIVEL:
        st.warning("A biblioteca `joblib` não está instalada. Rode `pip install joblib` para habilitar esta aba.")
    elif modelos is None:
        st.warning(
            "Não encontrei os arquivos do modelo. Rode o notebook `05-1_modelagem_preditiva.ipynb` "
            "(ele já gera tudo sozinho) e copie os 3 arquivos que ele salva em `outputs/modelos/`:\n\n"
            "- `modelo_preco.joblib`\n"
            "- `modelo_ocupacao.joblib`\n"
            "- `config_preditores.joblib`\n\n"
            f"para a pasta `{PASTA_MODELOS}` (crie a pasta `modelos` ao lado de `app2.py` se ela não existir)."
        )
    else:
        perfis_predefinidos = calcular_perfis_predefinidos(df)

        st.markdown("##### 🏷️ Perfil rápido")
        st.caption(
            "Escolha um perfil de Airbnb."
        )
        opcoes_perfil = ["Personalizado"] + list(perfis_predefinidos.keys())
        st.radio(
            "Perfil do imóvel", opcoes_perfil, horizontal=True, key="sim_perfil_escolhido",
            on_change=lambda: (aplicar_perfil(perfis_predefinidos[st.session_state["sim_perfil_escolhido"]])
                                if st.session_state["sim_perfil_escolhido"] != "Personalizado" else None),
        )

        st.markdown("##### 📍 Localização")
        opcoes_bairro = ["bairro"] + sorted(
            agg["bairro_padronizado"].dropna().unique().tolist()
        )
        bairro_escolhido = st.selectbox(
            "Bairro de referência (preenche automaticamente a latitude/longitude média do bairro)",
            opcoes_bairro,
        )
        if bairro_escolhido != "bairro":
            linha_bairro = agg[agg["bairro_padronizado"] == bairro_escolhido].iloc[0]
            lat_padrao, lon_padrao = float(linha_bairro["lat"]), float(linha_bairro["lon"])
            bairro_referencia = {
                "nome": bairro_escolhido,
                "preco_medio": float(linha_bairro["preco_medio"]),
                "taxa_ocupacao_media": float(linha_bairro["taxa_ocupacao_media"]),
                "rentabilidade_anual_media": float(linha_bairro["rentabilidade_media"]) * 365,
            }
        else:
            lat_padrao, lon_padrao = float(df["latitude"].mean()), float(df["longitude"].mean())
            bairro_referencia = None

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            max_hospedes = int(df["capacidade_hospedes"].max())
            capacidade_hospedes = st.number_input(f"Capacidade de hóspedes (máx.: {max_hospedes})",
                                                     min_value=1, max_value=max_hospedes,
                                                     value=4, key="sim_capacidade_hospedes")
            max_banheiros = float(df["banheiros"].max())
            banheiros = st.number_input(f"Banheiros (máx.: {max_banheiros:.1f})",
                                          min_value=0.0, max_value=max_banheiros,
                                          value=1.0, step=0.5, key="sim_banheiros",
                                          help="Meio-banheiro (ex.: 1.5) representa um lavabo/banheiro social sem chuveiro.")
            max_quartos = int(df["quartos"].max())
            quartos = st.number_input(f"Quartos (máx.: {max_quartos})",
                                        min_value=0, max_value=max_quartos,
                                        value=2, step=1, key="sim_quartos")
            max_camas = int(df["camas"].max())
            camas = st.number_input(f"Camas (máx.: {max_camas})",
                                      min_value=0, max_value=max_camas,
                                      value=2, step=1, key="sim_camas")
        with col_b:
            max_comodidades = int(df["qtd_comodidades"].max())
            qtd_comodidades = st.number_input(f"Nº de comodidades (máx.: {max_comodidades})",
                                                min_value=0, max_value=max_comodidades,
                                                value=15, key="sim_qtd_comodidades")
            nota_composta = st.slider("Nota composta (0 a 5)", 0.0, 5.0, 4.8, 0.1, key="sim_nota_composta")
            max_noites_min = int(df["noites_minimas"].max())
            noites_minimas = st.number_input(f"Noites mínimas (máx.: {max_noites_min})",
                                               min_value=1, max_value=max_noites_min,
                                               value=2, key="sim_noites_minimas")
            max_noites_max = int(df["noites_maximas"].max())
            noites_maximas = st.number_input(f"Noites máximas (máx.: {max_noites_max})",
                                               min_value=1, max_value=max_noites_max,
                                               value=30, key="sim_noites_maximas")
        with col_c:
            max_anuncios = int(df["total_anuncios_anfitriao"].max())
            total_anuncios_anfitriao = st.number_input(f"Total de anúncios do anfitrião (máx.: {max_anuncios})",
                                                          min_value=1, max_value=max_anuncios,
                                                          value=1, key="sim_total_anuncios")
            tem_banheiro_privativo = st.selectbox("Banheiro privativo?", ["Sim", "Não"],
                                                    key="sim_banheiro_privativo") == "Sim"
            flexibilidade_estadia = st.selectbox("Cancelamento/estadia flexível?", ["Sim", "Não"],
                                                   key="sim_flexibilidade") == "Sim"
            latitude = st.number_input("Latitude", value=lat_padrao, format="%.5f")
            longitude = st.number_input("Longitude", value=lon_padrao, format="%.5f")

        st.markdown("##### 🏷️ Características categóricas")
        col_d, col_e, col_f = st.columns(3)
        with col_d:
            tipo_quarto = st.selectbox("Tipo de quarto", sorted(df["tipo_quarto"].dropna().unique().tolist()),
                                         key="sim_tipo_quarto")
            tipo_propriedade = st.selectbox("Tipo de propriedade",
                                              sorted(df["tipo_propriedade"].dropna().unique().tolist()),
                                              key="sim_tipo_propriedade")
        with col_e:
            tipo_hospedagem = st.selectbox("Tipo de hospedagem",
                                             sorted(df["tipo_hospedagem"].dropna().unique().tolist()),
                                             key="sim_tipo_hospedagem")
            e_superanfitriao = st.selectbox("É superanfitrião?", ["f", "t"],
                                              format_func=lambda x: "Sim" if x == "t" else "Não",
                                              key="sim_e_superanfitriao")
        with col_f:
            reserva_instantanea = st.selectbox("Reserva instantânea?", ["f", "t"],
                                                 format_func=lambda x: "Sim" if x == "t" else "Não",
                                                 key="sim_reserva_instantanea")
            meses_disponiveis = list(modelos.get("multiplicador_sazonal", {}).keys())
            mes_referencia = st.selectbox("Mês de referência (ajuste sazonal, opcional)",
                                            ["(sem ajuste sazonal)"] + meses_disponiveis)

        st.markdown("##### ⭐ Histórico de avaliações (usado apenas no modelo de preço)")
        col_g, col_h = st.columns(2)
        with col_g:
            max_avaliacoes = int(df["numero_avaliacoes"].max())
            numero_avaliacoes = st.number_input(f"Número de avaliações (máx.: {max_avaliacoes})",
                                                  min_value=0, max_value=max_avaliacoes,
                                                  value=20, key="sim_numero_avaliacoes")
        with col_h:
            max_avaliacoes_mes = float(df["avaliacoes_por_mes"].max())
            avaliacoes_por_mes = st.number_input(f"Avaliações por mês (máx.: {max_avaliacoes_mes:.1f})",
                                                   min_value=0.0, max_value=max_avaliacoes_mes,
                                                   value=1.5, step=0.1, key="sim_avaliacoes_por_mes")

        if st.button("🔍 Calcular estimativa", type="primary"):
            dados_imovel = {
                "capacidade_hospedes": capacidade_hospedes, "banheiros": banheiros, "quartos": quartos,
                "camas": camas, "qtd_comodidades": qtd_comodidades, "nota_composta": nota_composta,
                "latitude": latitude, "longitude": longitude, "noites_minimas": noites_minimas,
                "noites_maximas": noites_maximas, "total_anuncios_anfitriao": total_anuncios_anfitriao,
                "tem_banheiro_privativo": int(tem_banheiro_privativo),
                "flexibilidade_estadia": int(flexibilidade_estadia),
                "tipo_quarto": tipo_quarto, "tipo_propriedade": tipo_propriedade,
                "tipo_hospedagem": tipo_hospedagem, "e_superanfitriao": e_superanfitriao,
                "reserva_instantanea": reserva_instantanea,
                "numero_avaliacoes": numero_avaliacoes, "avaliacoes_por_mes": avaliacoes_por_mes,
            }
            mes_para_calculo = None if mes_referencia == "(sem ajuste sazonal)" else mes_referencia

            try:
                resultado = prever_investimento(modelos, dados_imovel, mes_referencia=mes_para_calculo)
            except Exception as e:
                st.error(f"Não foi possível calcular a estimativa: {e}")
            else:
                st.divider()
                st.subheader("📊 Resultado da simulação")

                if resultado["tem_margem_erro"]:
                    p_min, p_max = resultado["preco_intervalo"]
                    o_min, o_max = resultado["ocupacao_intervalo"]
                    r_min, r_max = resultado["rentabilidade_intervalo"]
                    valor_preco = f"R$ {resultado['preco_diaria_previsto']:.2f} (± R$ {resultado['preco_erro_abs']:.2f})"
                    valor_ocup = f"{resultado['taxa_ocupacao_prevista']:.1%} (± {resultado['ocupacao_erro_abs']:.1%})"
                    valor_rent = f"R$ {resultado['rentabilidade_anual_estimada']:,.2f}"
                else:
                    valor_preco = f"R$ {resultado['preco_diaria_previsto']:.2f}"
                    valor_ocup = f"{resultado['taxa_ocupacao_prevista']:.1%}"
                    valor_rent = f"R$ {resultado['rentabilidade_anual_estimada']:,.2f}"

                if bairro_referencia:
                    delta_preco = resultado["preco_diaria_previsto"] - bairro_referencia["preco_medio"]
                    delta_ocup = resultado["taxa_ocupacao_prevista"] - bairro_referencia["taxa_ocupacao_media"]
                    delta_rent = (resultado["rentabilidade_anual_estimada"]
                                  - bairro_referencia["rentabilidade_anual_media"])
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Preço sugerido / diária", valor_preco,
                              delta=f"R$ {delta_preco:+.2f} vs. média do bairro")
                    m2.metric("Taxa de ocupação estimada", valor_ocup,
                              delta=f"{delta_ocup:+.1%} vs. média do bairro")
                    m3.metric("Noites ocupadas/ano (estim.)", f"{resultado['noites_ocupadas_ano_estimadas']:.0f}")
                    m4.metric("Rentabilidade anual estimada", valor_rent,
                              delta=f"R$ {delta_rent:+,.2f} vs. média do bairro")
                    st.caption(
                        f"Comparação com a média observada em **{bairro_referencia['nome']}**: "
                        f"preço R$ {bairro_referencia['preco_medio']:.2f}, ocupação "
                        f"{bairro_referencia['taxa_ocupacao_media']:.1%}, rentabilidade anual "
                        f"R$ {bairro_referencia['rentabilidade_anual_media']:,.2f} (estimativa própria do app, "
                        "não vem do modelo)."
                    )
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Preço sugerido / diária", valor_preco)
                    m2.metric("Taxa de ocupação estimada", valor_ocup)
                    m3.metric("Noites ocupadas/ano (estim.)", f"{resultado['noites_ocupadas_ano_estimadas']:.0f}")
                    m4.metric("Rentabilidade anual estimada", valor_rent)
                    st.caption("💡 Escolha um bairro de referência acima para comparar com a média da região.")

                if resultado["tem_margem_erro"]:
                    st.caption(
                        f"📐 Faixa realista considerando o erro do modelo: rentabilidade anual entre "
                        f"R$ {r_min:,.2f} e R$ {r_max:,.2f}. "
                        f"Confiabilidade dos modelos: preço R² = {resultado['r2_preco']:.2f} · "
                        f"ocupação R² = {resultado['r2_ocupacao']:.2f} "
                        "(quanto mais próximo de 1, melhor o modelo explica os dados)."
                    )


                if mes_para_calculo:
                    st.caption(
                        f"Ajuste sazonal aplicado para {mes_para_calculo}: "
                        f"fator {resultado['fator_sazonal_aplicado']:.3f} "
                        "(1.0 = ocupação média das coletas; acima de 1.0 indica alta temporada)."
                    )
                st.caption(
                    "⚠️ Estimativa gerada por modelos de Machine Learning treinados com dados históricos "
                    "do Airbnb Rio de Janeiro. Não constitui recomendação de investimento."
                )

        # -----------------------------------------------------------------
        # Buscar o imóvel ideal (otimização inversa, entre imóveis reais)
        # -----------------------------------------------------------------
        st.divider()
        st.markdown("##### 🎯 Buscar o imóvel ideal")
        st.caption(
            "Aqui você informa o **objetivo** e o app busca, entre os imóveis "
            "**reais** do dataset, o top 5 de configurações que o modelo prevê como mais próximas dele. "
        )

        objetivo = st.selectbox(
            "Objetivo",
            ["Maior rentabilidade possível", "Maior preço de diária possível", "Ocupação-alvo (mín. dias ocupados)"],
            key="busca_objetivo",
        )
        ocupacao_alvo_pct = None
        if objetivo == "Ocupação-alvo (mín. dias ocupados)":
            ocupacao_alvo_pct = st.selectbox(
                "Taxa de ocupação-alvo", [50, 75, 90, 100],
                format_func=lambda x: f"{x}%", key="busca_ocupacao_alvo",
            )

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            filtro_tipo_propriedade = st.multiselect(
                "Filtrar por tipo de propriedade (opcional)",
                sorted(df["tipo_propriedade"].dropna().unique().tolist()),
                key="busca_filtro_tipo_prop",
            )
        with col_f2:
            filtro_bairro_busca = st.multiselect(
                "Filtrar por bairro (opcional)",
                sorted(agg["bairro_padronizado"].dropna().unique().tolist()),
                key="busca_filtro_bairro",
            )

        if st.button("🔎 Buscar top 5 imóveis ideais", type="primary", key="busca_btn"):
            with st.spinner("Calculando previsões do modelo para os imóveis reais do dataset..."):
                base_pred = calcular_previsoes_dataset(df, modelos)

            filtrado = base_pred
            if filtro_tipo_propriedade:
                filtrado = filtrado[filtrado["tipo_propriedade"].isin(filtro_tipo_propriedade)]
            if filtro_bairro_busca:
                filtrado = filtrado[filtrado["bairro_padronizado"].isin(filtro_bairro_busca)]

            if filtrado.empty:
                st.warning("Nenhum imóvel real corresponde a esses filtros. Tente remover algum filtro.")
            else:
                if objetivo == "Maior rentabilidade possível":
                    top5 = filtrado.sort_values("rentabilidade_anual_prevista", ascending=False).head(5)
                elif objetivo == "Maior preço de diária possível":
                    top5 = filtrado.sort_values("preco_previsto", ascending=False).head(5)
                else:
                    alvo_frac = ocupacao_alvo_pct / 100
                    filtrado = filtrado.copy()
                    filtrado["dist_ocupacao_alvo"] = (filtrado["ocupacao_prevista"] - alvo_frac).abs()
                    top5 = filtrado.sort_values(
                        ["dist_ocupacao_alvo", "rentabilidade_anual_prevista"], ascending=[True, False]
                    ).head(5)

                erro_preco_busca = modelos.get("erro_preco_reais")
                erro_ocup_busca = modelos.get("erro_ocupacao")
                tem_margem_busca = erro_preco_busca is not None and erro_ocup_busca is not None

                st.success(f"Top {len(top5)} imóveis reais mais próximos do objetivo escolhido:")
                for posicao, (_, linha) in enumerate(top5.iterrows(), start=1):
                    with st.container(border=True):
                        st.markdown(
                            f"**#{posicao} — {linha.get('bairro_padronizado', 'bairro desconhecido')}** · "
                            f"{linha.get('tipo_propriedade', '-')} · {linha.get('tipo_quarto', '-')}"
                        )
                        cc1, cc2, cc3, cc4 = st.columns(4)
                        if tem_margem_busca:
                            cc1.metric("Preço/diária previsto",
                                       f"R$ {linha['preco_previsto']:.2f} (± R$ {erro_preco_busca:.2f})")
                            cc2.metric("Ocupação prevista",
                                       f"{linha['ocupacao_prevista']:.1%} (± {erro_ocup_busca:.1%})")
                        else:
                            cc1.metric("Preço/diária previsto", f"R$ {linha['preco_previsto']:.2f}")
                            cc2.metric("Ocupação prevista", f"{linha['ocupacao_prevista']:.1%}")
                        cc3.metric("Rentabilidade anual prevista", f"R$ {linha['rentabilidade_anual_prevista']:,.2f}")
                        cc4.metric(
                            "Capacidade / quartos",
                            f"{int(linha.get('capacidade_hospedes', 0))} hóspedes · "
                            f"{int(linha.get('quartos', 0))} qts"
                        )
                if not tem_margem_busca:
                    st.caption(
                        "ℹ️ Margem de erro não disponível neste `config_preditores.joblib` "
                        "(gere de novo com a versão atualizada do notebook `05-1` para vê-la aqui)."
                    )

