# Analizador de Comentarios Turisticos

Proyecto final con dos versiones:

- Consola: script CLI offline para analizar comentarios desde CSV.
- Web: frontend React con backend FastAPI que reutiliza el mismo motor de analisis.

El sistema limpia texto, detecta outliers, calcula n-gramas, clasifica sentimiento, genera topicos con un motor hibrido que incluye BERTopic, analiza el concepto "precio / valor / costo" y produce visualizaciones interactivas.

## Version de consola

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps bertopic==0.17.4
```

El proyecto no usa APIs externas. BERTopic se instala con `--no-deps` porque el sistema le pasa embeddings locales TF-IDF + SVD y no necesita descargar modelos de `sentence-transformers` ni instalar PyTorch.

## Uso

```bash
python tourist_comment_analyzer.py <archivo_csv> <columna_comentarios> <idioma> <titulo_reporte> <paleta>
```

Parametros obligatorios:

1. `archivo_csv`: ruta del archivo CSV.
2. `columna_comentarios`: nombre exacto de la columna con comentarios.
3. `idioma`: codigo del idioma, por ejemplo `es`, `en` o `fr`.
4. `titulo_reporte`: titulo usado en archivos y graficas.
5. `paleta`: paleta para visualizaciones. Opciones: `viridis`, `cividis`, `plasma`, `inferno`, `magma`, `plotly`, `safe`.

Ejemplo:

```bash
python tourist_comment_analyzer.py data/sample_reviews.csv comentario es "Reporte turistico" cividis
```

Opciones adicionales:

```bash
python tourist_comment_analyzer.py data/sample_reviews.csv comentario es "Reporte turistico" cividis \
  --output-dir reports \
  --topics 4 \
  --min-topic-docs 8
```

BERTopic/UMAP puede ser lento en equipos locales. El analisis rapido usa NMF y LSA-KMeans por defecto; para incluir BERTopic en la comparacion:

```bash
python tourist_comment_analyzer.py data/sample_reviews.csv comentario es "Reporte turistico" cividis --bertopic
```

## Version web local

Terminal 1:

```bash
python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```bash
cd web
npm install
npm run dev
```

Abrir:

```text
http://localhost:5173
```

La pagina permite subir un CSV, configurar columna, idioma, titulo, paleta, numero de topicos y umbral minimo. Tambien incluye un boton para cargar el CSV de ejemplo y un selector para activar BERTopic solo cuando se necesite el analisis mas pesado.

## Docker

Con Docker Desktop abierto:

```bash
docker compose up --build
```

Servicios:

- Frontend React: `http://localhost:5173`
- Backend FastAPI: `http://localhost:8000`
- Documentacion API: `http://localhost:8000/docs`

## Salidas

El script genera una carpeta `reports/` con:

- `*_report.html`: reporte principal con resumen, topicos, n-gramas, comentarios representativos y visualizaciones embebidas.
- `*_topics.html`: scatter plot interactivo por topico y sentimiento.
- `*_price.html`: scatter plot interactivo de similitud con "precio / valor / costo".
- `*_analysis.csv`: tabla detallada con texto limpio, outlier, sentimiento, topico y similitud con precio.
- Nube de palabras por grupo: global, positivo, negativo y outlier.

La version web tambien guarda reportes en `reports/<id_ejecucion>/` y permite descargar el HTML principal y el CSV analizado.

## Estructura

```text
.
├── api/main.py                  # Backend FastAPI
├── data/sample_reviews.csv      # CSV de ejemplo
├── web/                         # Frontend React
├── tourist_comment_analyzer.py  # Motor de analisis y CLI
├── requirements.txt             # Dependencias Python
├── Dockerfile                   # Imagen backend
└── docker-compose.yml           # API + frontend
```

## Metodologia

- Limpieza: minusculas, eliminacion de URLs, correos, numeros, caracteres especiales, stopwords y stemming con SnowballStemmer.
- Outliers: `IsolationForest` sobre matriz TF-IDF.
- N-gramas: unigramas, bigramas y trigramas calculados sobre comentarios atipicos.
- Nube de palabras: frecuencias normalizadas de terminos procesados, separadas por grupo.
- Sentimiento: clasificador lexicon-based offline para `es`, `en` y `fr`.
- Topicos: motor hibrido offline que compara NMF y LSA-KMeans en modo rapido. Opcionalmente puede incluir BERTopic con embeddings locales TF-IDF + SVD precomputados, UMAP y clustering local; el sistema selecciona el mejor por silueta coseno y balance de clusters. Si una particion es pequena, usa frecuencia de palabras.
- Scatter plots: reduccion a 2D con `TruncatedSVD`; color y simbolo se usan como codificacion redundante.
- Precio / valor / costo: similitud coseno entre comentarios y un concepto sintetico multilingue.

## Roles sugeridos para presentacion

- Preprocesamiento: limpieza, tokenizacion, stopwords y stemming.
- Outliers: IsolationForest y analisis de n-gramas.
- Sentimiento: separacion positivo/negativo.
- Topicos: BERTopic, NMF, LSA-KMeans y alternativa de frecuencia por particiones pequenas.
- Visualizacion: reportes HTML y scatter plots interactivos.
- Documentacion: README, requirements y demostracion de uso.
