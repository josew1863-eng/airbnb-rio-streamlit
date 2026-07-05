# Airbnb Rio de Janeiro — Análise Espacial (App Streamlit)

App com 3 abas:
- 🗺️ **Mapa Dinâmico** — bairros coloridos por score de luxo, bolhas de rentabilidade/ocupação, pontos turísticos, com filtro por faixa de preço e por região popular/não popular.
- 📅 **Sazonalidade** — ocupação média por mês (ano cheio) e preço médio por mês (meses coletados), destacando alta e baixa temporada.
- 📊 **Correlações** — matriz de correlação (preço, ocupação, luxo, rentabilidade, distância a ponto turístico) e Índice de Moran Global (autocorrelação espacial do preço entre bairros vizinhos).

## Estrutura de pastas

```
.
├── app.py
├── requirements.txt
└── dados/
    ├── dataset_features_slim.parquet
    ├── neighbourhoods.geojson
    ├── calendar_agregado.parquet
    └── listings_temporal.parquet
```

Suba essa estrutura inteira (pasta `dados/` incluída) pro repositório do GitHub — o `app.py` espera achar os arquivos dentro de `dados/`, no mesmo nível dele.

## Como colocar no ar (Streamlit Community Cloud — gratuito)

1. Crie um repositório novo no GitHub e suba todos esses arquivos (pode ser público ou privado).
2. Acesse [streamlit.io/cloud](https://streamlit.io/cloud) e faça login com sua conta do GitHub.
3. Clique em **"New app"**, escolha o repositório, a branch (`main`) e o arquivo principal (`app.py`).
4. Clique em **Deploy**. Em 1-2 minutos o app sobe e gera um link tipo `seunome-airbnb-rio.streamlit.app`.

## Rodando localmente (pra testar antes de subir)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre automaticamente em `http://localhost:8501`.

## Observações importantes

- **Pontos turísticos (osmnx):** a primeira vez que o app carrega, ele busca os pontos turísticos do Rio no OpenStreetMap (precisa de internet). O resultado fica em cache (`@st.cache_data`) enquanto o app estiver no ar — não busca de novo a cada clique do usuário, só quando o servidor reinicia.
- **Se o deploy no Streamlit Cloud falhar por causa do `geopandas`/`osmnx`** (erro relacionado a GDAL): crie um arquivo `packages.txt` na raiz do repositório com o conteúdo abaixo — ele instala as dependências de sistema que faltam:
  ```
  gdal-bin
  libgdal-dev
  ```
- **Se quiser trocar os dados no futuro** (nova coleta, mais bairros etc.): é só substituir os arquivos dentro de `dados/` mantendo os mesmos nomes de coluna usados no `app.py`.
