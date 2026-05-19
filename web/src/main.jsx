import React, { useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import ReactECharts from "echarts-for-react";
import {
  BarChart3,
  Brain,
  Download,
  FileText,
  Gauge,
  Loader2,
  Network,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import "./styles.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";
const palettes = ["cividis", "viridis", "plasma", "inferno", "magma", "plotly", "safe"];
const topicColors = ["#0f766e", "#2563eb", "#7c3aed", "#db2777", "#ea580c", "#16a34a", "#0891b2", "#475569"];

function App() {
  const [file, setFile] = useState(null);
  const [form, setForm] = useState({
    textColumn: "comentario",
    language: "es",
    title: "Reporte turistico",
    palette: "cividis",
    topics: 4,
    minTopicDocs: 8,
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const charts = useMemo(() => {
    const records = result?.records || [];
    return {
      topics: topicMapOption(records),
      price: priceMapOption(records),
      sentiment: sentimentOption(result?.metrics),
      quality: qualityOption(result?.topic_summaries || []),
      model: modelOption(result?.topic_diagnostics || []),
    };
  }, [result]);

  async function loadSample() {
    const response = await fetch(`${API_URL}/sample`);
    const blob = await response.blob();
    setFile(new File([blob], "sample_reviews.csv", { type: "text/csv" }));
  }

  async function submit(event) {
    event.preventDefault();
    if (!file) {
      setError("Selecciona un archivo CSV o carga el ejemplo.");
      return;
    }

    const body = new FormData();
    body.append("file", file);
    body.append("text_column", form.textColumn);
    body.append("language", form.language);
    body.append("report_title", form.title);
    body.append("palette", form.palette);
    body.append("topics", String(form.topics));
    body.append("min_topic_docs", String(form.minTopicDocs));

    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_URL}/analyze`, { method: "POST", body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "No se pudo analizar el CSV.");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="app-frame">
        <header className="top-bar">
          <div>
            <p className="eyebrow">Tourism Intelligence Suite</p>
            <h1>Analizador avanzado de experiencia turistica</h1>
          </div>
          <div className="engine-badge">
            <Brain size={17} />
            Hybrid Topic Engine
          </div>
        </header>

        <form onSubmit={submit} className="command-bar">
          <label className="drop-zone">
            <UploadCloud size={19} />
            <span>{file ? file.name : "CSV"}</span>
            <input
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => setFile(event.target.files?.[0] || null)}
            />
          </label>
          <button type="button" className="secondary-action" onClick={loadSample}>
            <FileText size={15} />
            Ejemplo
          </button>
          <label>
            Columna
            <input value={form.textColumn} onChange={(event) => setForm({ ...form, textColumn: event.target.value })} />
          </label>
          <label>
            Reporte
            <input value={form.title} onChange={(event) => setForm({ ...form, title: event.target.value })} />
          </label>
          <label>
            Idioma
            <select value={form.language} onChange={(event) => setForm({ ...form, language: event.target.value })}>
              <option value="es">ES</option>
              <option value="en">EN</option>
              <option value="fr">FR</option>
            </select>
          </label>
          <label>
            Paleta
            <select value={form.palette} onChange={(event) => setForm({ ...form, palette: event.target.value })}>
              {palettes.map((palette) => (
                <option key={palette} value={palette}>
                  {palette}
                </option>
              ))}
            </select>
          </label>
          <label>
            Topicos
            <input
              type="number"
              min="2"
              max="8"
              value={form.topics}
              onChange={(event) => setForm({ ...form, topics: Number(event.target.value) })}
            />
          </label>
          <label>
            Min docs
            <input
              type="number"
              min="2"
              max="50"
              value={form.minTopicDocs}
              onChange={(event) => setForm({ ...form, minTopicDocs: Number(event.target.value) })}
            />
          </label>
          <button className="primary-action" disabled={loading}>
            {loading ? <Loader2 className="spin" size={17} /> : <Sparkles size={17} />}
            {loading ? "Analizando" : "Analizar"}
          </button>
        </form>

        {error && <p className="error-message">{error}</p>}

        {!result ? (
          <section className="empty-state">
            <div className="empty-mark">
              <BarChart3 size={42} />
            </div>
            <h2>Sube un CSV para generar el dashboard</h2>
          </section>
        ) : (
          <Dashboard result={result} charts={charts} />
        )}
      </section>
    </main>
  );
}

function Dashboard({ result, charts }) {
  const files = result.files;
  const avgQuality = average(result.topic_summaries.map((row) => Number(row.quality || 0)));
  const selectedDiagnostics = (result.topic_diagnostics || []).filter((row) => row.selected);
  const dominantModel = mostCommon(selectedDiagnostics.map((row) => row.method));

  return (
    <section className="dashboard">
      <div className="kpi-row">
        <Metric icon={<BarChart3 size={18} />} label="Comentarios" value={result.metrics.total} />
        <Metric icon={<Gauge size={18} />} label="Calidad media" value={avgQuality.toFixed(3)} />
        <Metric icon={<Brain size={18} />} label="Modelo" value={dominantModel || "n/a"} />
        <Metric icon={<Network size={18} />} label="Outliers" value={result.metrics.outliers} />
        <div className="download-panel">
          <a href={`${API_URL}${files.report}`} target="_blank" rel="noreferrer">
            <FileText size={15} />
            HTML
          </a>
          <a href={`${API_URL}${files.csv}`}>
            <Download size={15} />
            CSV
          </a>
        </div>
      </div>

      <div className="dashboard-grid">
        <ChartCard title="Mapa semantico" meta={`${result.records.length} comentarios`} className="map-card">
          <ReactECharts option={charts.topics} className="chart chart-tall" notMerge lazyUpdate />
        </ChartCard>
        <ChartCard title="Precio / valor / costo" meta="similitud">
          <ReactECharts option={charts.price} className="chart" notMerge lazyUpdate />
        </ChartCard>
        <ChartCard title="Sentimiento" meta="volumen">
          <ReactECharts option={charts.sentiment} className="chart chart-small" notMerge lazyUpdate />
        </ChartCard>
        <ChartCard title="Calidad por topico" meta="silueta coseno">
          <ReactECharts option={charts.quality} className="chart chart-small" notMerge lazyUpdate />
        </ChartCard>
        <ChartCard title="Modelos evaluados" meta="BERTopic / NMF / LSA">
          <ReactECharts option={charts.model} className="chart chart-small" notMerge lazyUpdate />
        </ChartCard>
        <WordCloudPanel clouds={result.word_clouds || {}} />
        <TopicPanel topics={result.topic_summaries} />
      </div>

      <div className="bottom-grid">
        <InsightList title="Comentarios mas cercanos a precio" rows={result.top_price} />
        <NgramPanel ngrams={result.outlier_ngrams} />
      </div>
    </section>
  );
}

function ChartCard({ title, meta, className = "", children }) {
  return (
    <article className={`card chart-card ${className}`}>
      <div className="section-title">
        <h2>{title}</h2>
        <span>{meta}</span>
      </div>
      {children}
    </article>
  );
}

function TopicPanel({ topics }) {
  return (
    <article className="card topic-panel">
      <div className="section-title">
        <h2>Topicos explicables</h2>
        <span>{topics.length} segmentos</span>
      </div>
      <div className="topic-list">
        {topics.map((topic, index) => (
          <div className="topic-row" key={`${topic.sentiment}-${topic.topic}`}>
            <span className="topic-dot" style={{ background: topicColors[index % topicColors.length] }} />
            <div>
              <strong>{topic.topic}</strong>
              <p>{topic.keywords}</p>
            </div>
            <em>{Number(topic.quality || 0).toFixed(2)}</em>
          </div>
        ))}
      </div>
    </article>
  );
}

function WordCloudPanel({ clouds }) {
  const groups = [
    ["global", "Global"],
    ["positivo", "Positivo"],
    ["negativo", "Negativo"],
    ["outlier", "Outlier"],
  ];

  return (
    <article className="card word-cloud-panel">
      <div className="section-title">
        <h2>Nube de palabras</h2>
        <span>frecuencia</span>
      </div>
      <div className="word-cloud-grid">
        {groups.map(([key, label]) => (
          <div className="word-cloud-group" key={key}>
            <h3>{label}</h3>
            <div className="word-cloud">
              {(clouds[key] || []).slice(0, 34).map((item) => (
                <span
                  key={`${key}-${item.term}`}
                  style={{ "--size": `${12 + Number(item.weight || 0) * 22}px` }}
                  title={`${item.term}: ${item.count}`}
                >
                  {item.term}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </article>
  );
}

function InsightList({ title, rows }) {
  return (
    <article className="card">
      <div className="section-title">
        <h2>{title}</h2>
        <span>top 5</span>
      </div>
      <ol className="comment-list">
        {rows.map((row, index) => (
          <li key={index}>
            <strong>{Number(row.price_similarity).toFixed(3)}</strong>
            <span>{row.comment}</span>
          </li>
        ))}
      </ol>
    </article>
  );
}

function NgramPanel({ ngrams }) {
  return (
    <article className="card">
      <div className="section-title">
        <h2>N-gramas outlier</h2>
        <span>anomalias</span>
      </div>
      <div className="ngram-grid">
        {Object.entries(ngrams).map(([label, values]) => (
          <div key={label}>
            <h3>{label}</h3>
            <p>{values.slice(0, 8).map(([term, count]) => `${term} (${count})`).join(", ") || "Sin datos"}</p>
          </div>
        ))}
      </div>
    </article>
  );
}

function Metric({ icon, label, value }) {
  return (
    <div className="metric-card">
      <div className="metric-icon">{icon}</div>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function topicMapOption(records) {
  const groups = groupBy(records, "topic_label");
  return {
    ...baseOption(),
    legend: { top: 0, type: "scroll", textStyle: { color: "#475569", fontSize: 11 } },
    grid: { left: 34, right: 18, top: 42, bottom: 28 },
    xAxis: axis(),
    yAxis: axis(),
    series: Object.entries(groups).map(([name, rows], index) => ({
      name,
      type: "scatter",
      symbolSize: (value) => 10 + Math.min(14, value[2] * 46),
      data: rows.map((row) => [
        Number(row.x || 0),
        Number(row.y || 0),
        Number(row.price_similarity || 0),
        row.comment,
        row.sentiment,
        row.topic_model,
        Number(row.topic_quality || 0),
      ]),
      itemStyle: {
        color: topicColors[index % topicColors.length],
        borderColor: "#fff",
        borderWidth: 1.5,
        shadowBlur: 8,
        shadowColor: "rgba(15,23,42,.18)",
      },
      emphasis: { scale: 1.35, focus: "series" },
    })),
    tooltip: {
      ...tooltip(),
      formatter: (params) =>
        `<b>${params.seriesName}</b><br/>${params.value[4]} · ${params.value[5]}<br/>Calidad: ${params.value[6].toFixed(3)}<br/><span>${params.value[3]}</span>`,
    },
  };
}

function priceMapOption(records) {
  return {
    ...baseOption(),
    grid: { left: 32, right: 16, top: 16, bottom: 26 },
    visualMap: {
      min: 0,
      max: Math.max(0.1, ...records.map((row) => Number(row.price_similarity || 0))),
      right: 4,
      top: 18,
      itemWidth: 10,
      itemHeight: 110,
      inRange: { color: ["#dbeafe", "#2dd4bf", "#f59e0b", "#dc2626"] },
      textStyle: { color: "#64748b", fontSize: 10 },
    },
    xAxis: axis(),
    yAxis: axis(),
    series: [
      {
        type: "scatter",
        symbolSize: (value) => 9 + value[2] * 42,
        data: records.map((row) => [
          Number(row.price_x || 0),
          Number(row.price_y || 0),
          Number(row.price_similarity || 0),
          row.comment,
          row.sentiment,
        ]),
        itemStyle: { borderColor: "#fff", borderWidth: 1.3, shadowBlur: 7, shadowColor: "rgba(15,23,42,.16)" },
      },
    ],
    tooltip: {
      ...tooltip(),
      formatter: (params) => `<b>${params.value[4]}</b><br/>Similitud: ${params.value[2].toFixed(3)}<br/>${params.value[3]}`,
    },
  };
}

function sentimentOption(metrics) {
  const data = [
    { name: "positivo", value: metrics?.positive || 0, itemStyle: { color: "#0f766e" } },
    { name: "negativo", value: metrics?.negative || 0, itemStyle: { color: "#dc2626" } },
    { name: "outlier", value: metrics?.outliers || 0, itemStyle: { color: "#64748b" } },
  ];
  return {
    ...baseOption(),
    grid: { left: 8, right: 8, top: 8, bottom: 8 },
    series: [
      {
        type: "pie",
        radius: ["52%", "78%"],
        center: ["50%", "52%"],
        data,
        label: { color: "#334155", fontWeight: 800, formatter: "{b}\n{c}" },
        itemStyle: { borderColor: "#fff", borderWidth: 3 },
      },
    ],
    tooltip: tooltip(),
  };
}

function qualityOption(topics) {
  return {
    ...baseOption(),
    grid: { left: 92, right: 20, top: 10, bottom: 22 },
    xAxis: { ...axis(), max: 1 },
    yAxis: { type: "category", data: topics.map((row) => row.topic).reverse(), axisLabel: { color: "#475569", fontSize: 10 } },
    series: [
      {
        type: "bar",
        data: topics.map((row) => Number(row.quality || 0)).reverse(),
        barWidth: 12,
        itemStyle: { color: "#0f766e", borderRadius: [0, 8, 8, 0] },
      },
    ],
    tooltip: tooltip(),
  };
}

function modelOption(diagnostics) {
  const rows = diagnostics.length ? diagnostics : [{ sentiment: "n/a", method: "sin datos", quality: 0, selected: false }];
  return {
    ...baseOption(),
    grid: { left: 112, right: 16, top: 8, bottom: 20 },
    xAxis: { ...axis(), max: 1 },
    yAxis: {
      type: "category",
      data: rows.map((row) => `${row.sentiment} · ${row.method}`).reverse(),
      axisLabel: { color: "#475569", fontSize: 10 },
    },
    series: [
      {
        type: "bar",
        data: rows.map((row) => ({
          value: Number(row.quality || 0),
          itemStyle: { color: row.selected ? "#0f766e" : "#94a3b8", borderRadius: [0, 8, 8, 0] },
        })).reverse(),
        barWidth: 11,
      },
    ],
    tooltip: tooltip(),
  };
}

function baseOption() {
  return {
    backgroundColor: "transparent",
    animationDuration: 650,
    textStyle: { fontFamily: "Inter, system-ui, sans-serif", color: "#17202a" },
  };
}

function axis() {
  return {
    type: "value",
    axisLine: { show: false },
    axisTick: { show: false },
    axisLabel: { color: "#94a3b8", fontSize: 10 },
    splitLine: { lineStyle: { color: "rgba(148,163,184,.18)" } },
  };
}

function tooltip() {
  return {
    trigger: "item",
    confine: true,
    backgroundColor: "rgba(15,23,42,.94)",
    borderColor: "rgba(15,23,42,.94)",
    textStyle: { color: "#fff", fontSize: 12 },
    extraCssText: "max-width:360px;white-space:normal;border-radius:8px;box-shadow:0 14px 32px rgba(15,23,42,.24);",
  };
}

function groupBy(records, key) {
  return records.reduce((acc, row) => {
    const value = row[key] || "sin_topico";
    acc[value] = acc[value] || [];
    acc[value].push(row);
    return acc;
  }, {});
}

function average(values) {
  const valid = values.filter((value) => Number.isFinite(value));
  if (!valid.length) return 0;
  return valid.reduce((sum, value) => sum + value, 0) / valid.length;
}

function mostCommon(values) {
  const counts = values.filter(Boolean).reduce((acc, value) => {
    acc[value] = (acc[value] || 0) + 1;
    return acc;
  }, {});
  return Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] || "";
}

createRoot(document.getElementById("root")).render(<App />);
