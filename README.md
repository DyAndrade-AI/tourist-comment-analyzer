# Analizador de Comentarios Turisticos

Aplicacion para analizar comentarios turisticos desde archivos CSV. El proyecto clasifica sentimiento, detecta comentarios atipicos, agrupa temas, calcula menciones relacionadas con precio/valor/costo y genera reportes descargables.

Incluye una interfaz de consola, una API FastAPI y un frontend React.

## Caracteristicas

- Analisis local sin APIs externas.
- Carga de comentarios desde CSV.
- Deteccion de outliers con `IsolationForest`.
- Clasificacion de sentimiento para `es`, `en` y `fr`.
- Modelado de topicos con NMF, LSA-KMeans y BERTopic opcional.
- Reportes HTML, graficas interactivas y CSV final con resultados.

## Estructura

```text
.
├── tourist_comment_analyzer.py  # Motor de analisis y CLI
├── api/main.py                  # API FastAPI
├── web/                         # Frontend React
├── data/sample_reviews.csv      # Dataset de ejemplo
├── requirements.txt             # Dependencias Python
├── Dockerfile                   # Imagen del backend
├── web/Dockerfile               # Imagen del frontend
└── docker-compose.yml           # Entorno completo
```

## Requisitos

- Python 3.11+
- Node.js 22+
- Docker Desktop, opcional para ejecucion con contenedores

## Instalacion local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps bertopic==0.17.4
```

## Uso por consola

```bash
python tourist_comment_analyzer.py data/sample_reviews.csv comentario es "Reporte turistico" cividis
```

Argumentos:

- `csv_path`: ruta del archivo CSV.
- `text_column`: columna que contiene los comentarios.
- `language`: idioma de analisis (`es`, `en`, `fr`).
- `report_title`: titulo del reporte.
- `palette`: paleta de color (`viridis`, `cividis`, `plasma`, `inferno`, `magma`, `plotly`, `safe`).

Opciones utiles:

```bash
python tourist_comment_analyzer.py data/sample_reviews.csv comentario es "Reporte turistico" cividis \
  --output-dir reports \
  --topics 4 \
  --min-topic-docs 8 \
  --bertopic
```

## Uso web local

Backend:

```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd web
npm install
npm run dev
```

Abrir:

```text
http://localhost:5173
```

## Docker

```bash
docker compose up --build
```

Servicios:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

## Salidas

Los resultados se guardan en `reports/`:

- `*_report.html`: reporte principal.
- `*_topics.html`: visualizacion de topicos.
- `*_price.html`: analisis de precio, valor y costo.
- `*_analysis.csv`: comentarios procesados con sentimiento, topico, outlier y similitud de precio.
