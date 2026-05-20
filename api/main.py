from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from tourist_comment_analyzer import SUPPORTED_PALETTES, analyze_csv, write_analysis_outputs


BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "uploads"
REPORT_DIR = BASE_DIR / "reports"
MAX_RESPONSE_RECORDS = 2000
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Tourist Comment Analyzer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/reports", StaticFiles(directory=REPORT_DIR), name="reports")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/palettes")
def palettes() -> dict[str, list[str]]:
    return {"palettes": sorted(SUPPORTED_PALETTES)}


@app.get("/sample")
def sample_csv() -> FileResponse:
    sample_path = BASE_DIR / "data" / "sample_reviews.csv"
    return FileResponse(sample_path, media_type="text/csv", filename="sample_reviews.csv")


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    text_column: str = Form(...),
    language: str = Form("es"),
    report_title: str = Form("Reporte turistico"),
    palette: str = Form("cividis"),
    topics: int = Form(4),
    min_topic_docs: int = Form(8),
    use_bertopic: bool = Form(False),
) -> dict[str, object]:
    palette = palette.lower().strip()
    if palette not in SUPPORTED_PALETTES:
        raise HTTPException(status_code=400, detail="Paleta no soportada.")

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Sube un archivo CSV.")

    run_id = uuid.uuid4().hex[:10]
    upload_path = UPLOAD_DIR / f"{run_id}_{Path(file.filename).name}"
    output_dir = REPORT_DIR / run_id
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    with upload_path.open("wb") as target:
        shutil.copyfileobj(file.file, target)

    try:
        result = analyze_csv(upload_path, text_column, language, topics, min_topic_docs, use_bertopic)
        outputs = write_analysis_outputs(result, report_title, palette, output_dir)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    frame = result.frame.drop(columns=["tokens"], errors="ignore").copy()
    frame = frame.replace({float("inf"): None, float("-inf"): None})
    frame = frame.where(frame.notna(), None)
    records_frame = frame.head(MAX_RESPONSE_RECORDS)
    records = records_frame.to_dict(orient="records")
    top_price = (
        frame.sort_values("price_similarity", ascending=False)
        .head(5)[["comment", "price_similarity", "sentiment"]]
        .to_dict(orient="records")
    )

    return {
        "run_id": run_id,
        "metrics": {
            "total": len(frame),
            "outliers": int(frame["is_outlier"].sum()),
            "normal": int((~frame["is_outlier"]).sum()),
            "positive": int((frame["sentiment"] == "positivo").sum()),
            "negative": int((frame["sentiment"] == "negativo").sum()),
        },
        "records_limited": len(frame) > len(records_frame),
        "records_returned": len(records),
        "topic_summaries": result.topic_summaries,
        "topic_diagnostics": result.topic_diagnostics,
        "outlier_ngrams": result.outlier_ngrams,
        "word_clouds": result.word_clouds,
        "top_price": top_price,
        "records": records,
        "files": {
            "report": f"/reports/{run_id}/{outputs['report_html'].name}",
            "topics": f"/reports/{run_id}/{outputs['topics_html'].name}",
            "price": f"/reports/{run_id}/{outputs['price_html'].name}",
            "csv": f"/reports/{run_id}/{outputs['analysis_csv'].name}",
        },
    }
