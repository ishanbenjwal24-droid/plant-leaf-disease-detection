import { ChangeEvent, DragEvent, useEffect, useMemo, useState } from "react";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BarChart3,
  Check,
  ChevronRight,
  CircleHelp,
  Clock3,
  Cpu,
  Database,
  FileImage,
  FlaskConical,
  Leaf,
  LoaderCircle,
  ShieldCheck,
  Trash2,
  Upload,
  Zap,
} from "lucide-react";
import {
  clearHistory,
  compareModels,
  getHistory,
  getMetrics,
  HistoryItem,
  MetricsResponse,
  Prediction,
} from "./api";

type Page = "overview" | "metrics" | "history";
type CompareResult = { ml: Prediction; cnn: Prediction; total_inference_ms: number };

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/webp", "image/bmp"];

function App() {
  const [page, setPage] = useState<Page>("overview");
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState<CompareResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [history, setHistory] = useState<HistoryItem[] | null>(null);
  const [dataError, setDataError] = useState("");

  useEffect(() => {
    if (!file) {
      setPreview("");
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (page === "metrics") {
      getMetrics().then(setMetrics).catch((reason: Error) => setDataError(reason.message));
    }
    if (page === "history") {
      getHistory().then(setHistory).catch((reason: Error) => setDataError(reason.message));
    }
  }, [page]);

  const navItems = useMemo(
    () => [
      { id: "overview" as const, label: "Workspace", icon: Leaf },
      { id: "metrics" as const, label: "Model metrics", icon: BarChart3 },
      { id: "history" as const, label: "Prediction history", icon: Clock3 },
    ],
    [],
  );

  function acceptFile(candidate: File | undefined) {
    setError("");
    setResult(null);
    if (!candidate) return;
    if (!ACCEPTED_TYPES.includes(candidate.type)) {
      setFile(null);
      setError("Choose a JPEG, PNG, WebP, or BMP image.");
      return;
    }
    if (candidate.size > MAX_FILE_BYTES) {
      setFile(null);
      setError("The image must be smaller than 10 MB.");
      return;
    }
    setFile(candidate);
  }

  function onFileChange(event: ChangeEvent<HTMLInputElement>) {
    acceptFile(event.target.files?.[0]);
    event.target.value = "";
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    acceptFile(event.dataTransfer.files?.[0]);
  }

  async function runComparison() {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      setResult(await compareModels(file));
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not compare the models.");
    } finally {
      setBusy(false);
    }
  }

  async function onClearHistory() {
    try {
      await clearHistory();
      setHistory([]);
      setDataError("");
    } catch (reason) {
      setDataError(reason instanceof Error ? reason.message : "Could not clear history.");
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><Leaf size={19} strokeWidth={2.5} /></div>
          <div><strong>LeafLab</strong><span>MODEL COMPARISON</span></div>
        </div>
        <div className="side-label">PROJECT</div>
        <nav aria-label="Main navigation">
          {navItems.map(({ id, label, icon: Icon }) => (
            <button
              className={`nav-item ${page === id ? "active" : ""}`}
              key={id}
              onClick={() => { setPage(id); setDataError(""); }}
              aria-current={page === id ? "page" : undefined}
            >
              <Icon size={17} /><span>{label}</span>
              {page === id && <ChevronRight className="nav-chevron" size={15} />}
            </button>
          ))}
        </nav>
        <div className="sidebar-spacer" />
        <div className="sidebar-note">
          <div className="note-icon"><ShieldCheck size={17} /></div>
          <strong>Built for learning</strong>
          <p>Predictions are educational and are not a plant health diagnosis.</p>
        </div>
        <div className="side-footer"><span className="status-dot" /> Local project environment</div>
      </aside>

      <main className="main-area">
        <header className="topbar">
          <div className="breadcrumb"><span>Plant disease project</span><span className="crumb-slash">/</span><strong>{navItems.find((item) => item.id === page)?.label}</strong></div>
          <div className="topbar-right"><span className="dataset-chip"><Database size={14} /> PlantVillage</span><span className="version-chip">SEMESTER PROJECT</span></div>
        </header>

        {page === "overview" && (
          <OverviewPage
            preview={preview}
            file={file}
            dragging={dragging}
            result={result}
            busy={busy}
            error={error}
            onFileChange={onFileChange}
            onDragState={setDragging}
            onDrop={onDrop}
            onRun={runComparison}
            onReset={() => { setFile(null); setResult(null); setError(""); }}
          />
        )}
        {page === "metrics" && <MetricsPage metrics={metrics} error={dataError} />}
        {page === "history" && <HistoryPage items={history} error={dataError} onClear={onClearHistory} />}
      </main>
    </div>
  );
}

type OverviewProps = {
  preview: string;
  file: File | null;
  dragging: boolean;
  result: CompareResult | null;
  busy: boolean;
  error: string;
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onDragState: (active: boolean) => void;
  onDrop: (event: DragEvent<HTMLDivElement>) => void;
  onRun: () => void;
  onReset: () => void;
};

function OverviewPage(props: OverviewProps) {
  const {
    preview, file, dragging, result, busy, error, onFileChange,
    onDragState, onDrop, onRun, onReset,
  } = props;
  return (
    <div className="page-content">
      <section className="hero-row">
        <div>
          <div className="eyebrow"><span className="eyebrow-dot" /> COMPUTER VISION · MODEL STUDY</div>
          <h1>See how two models<br /><em>read a leaf.</em></h1>
          <p className="hero-copy">Compare a feature-based ML baseline with a convolutional neural network on the same leaf image.</p>
          <div className="hero-tags"><span><Cpu size={14} /> HOG + Linear SVM</span><span><Zap size={14} /> Convolutional network</span></div>
        </div>
        <div className="hero-illustration" aria-hidden="true">
          <div className="orbit orbit-one" /><div className="orbit orbit-two" />
          <div className="leaf-disc"><Leaf size={82} strokeWidth={1.1} /></div>
          <span className="float-label label-ml"><span className="mini-dot amber" /> ML</span>
          <span className="float-label label-dl"><span className="mini-dot green" /> DL</span>
          <span className="float-line line-one" /><span className="float-line line-two" />
        </div>
      </section>

      <div className="workspace-grid">
        <section className="panel upload-panel">
          <div className="panel-heading">
            <div><span className="step-label">STEP 01</span><h2>Choose a leaf image</h2></div>
            <span className="quiet-pill">LOCAL PROCESSING</span>
          </div>
          <div
            className={`dropzone ${dragging ? "dragging" : ""} ${preview ? "has-image" : ""}`}
            onDragOver={(event) => { event.preventDefault(); onDragState(true); }}
            onDragLeave={() => onDragState(false)}
            onDrop={onDrop}
          >
            {preview ? (
              <>
                <img className="preview-image" src={preview} alt="Selected plant leaf preview" />
                <div className="image-overlay"><span><Check size={14} /> Image ready</span></div>
              </>
            ) : (
              <div className="drop-empty">
                <div className="upload-icon"><Upload size={21} /></div>
                <strong>Drop your image here</strong>
                <span>or choose a file from your device</span>
                <label className="browse-button" htmlFor="leaf-file"><FileImage size={15} /> Browse files</label>
              </div>
            )}
            <input id="leaf-file" type="file" accept="image/jpeg,image/png,image/webp,image/bmp" onChange={onFileChange} hidden />
          </div>
          <div className="file-meta-row">
            {file ? <><span className="file-name"><FileImage size={14} />{file.name}</span><button className="text-button" onClick={onReset}>Remove</button></> : <><span>JPEG, PNG, WebP or BMP</span><span>Max 10 MB</span></>}
          </div>
          {error && <div className="error-box" role="alert"><CircleHelp size={16} />{error}</div>}
          <button className="primary-button compare-button" disabled={!file || busy} onClick={onRun}>
            {busy ? <><LoaderCircle className="spin" size={17} /> Comparing models…</> : <><FlaskConical size={17} /> Compare both models <ChevronRight size={17} /></>}
          </button>
          <p className="privacy-note"><ShieldCheck size={13} /> Your image is processed in memory and is not saved.</p>
        </section>

        <section className="panel method-panel">
          <div className="panel-heading">
            <div><span className="step-label">THE COMPARISON</span><h2>Two ways to learn</h2></div>
            <span className="sparkle">✳</span>
          </div>
          <div className="method-card ml-card">
            <div className="method-icon ml-icon"><Activity size={19} /></div>
            <div className="method-details"><div className="method-title-row"><strong>Traditional ML</strong><span className="model-index">01</span></div><p>HOG shape + HSV color features</p><div className="method-model">Linear Support Vector Machine</div></div>
          </div>
          <div className="method-connector"><span /></div>
          <div className="method-card dl-card">
            <div className="method-icon dl-icon"><Zap size={18} /></div>
            <div className="method-details"><div className="method-title-row"><strong>Deep learning</strong><span className="model-index">02</span></div><p>Learns patterns directly from pixels</p><div className="method-model">Convolutional Neural Network</div></div>
          </div>
          <div className="method-footnote"><span>Same split</span><i /> <span>Same labels</span><i /> <span>Fair comparison</span></div>
        </section>
      </div>

      {result && <ComparisonResults result={result} />}

      <div className="limitation-strip"><ShieldCheck size={16} /><div><strong>Use this as a learning tool.</strong><span>Dataset images were collected in controlled conditions. Results may differ on field photos; predictions are not professional agricultural advice.</span></div></div>
      <footer className="page-footer"><span>LeafLab · Plant leaf disease model comparison</span><span>Built with scikit-learn + TensorFlow</span></footer>
    </div>
  );
}

function ComparisonResults({ result }: { result: CompareResult }) {
  return (
    <section className="results-section" aria-live="polite">
      <div className="results-title"><div><span className="step-label">COMPARISON COMPLETE</span><h2>What each model saw</h2></div><span className="duration-pill"><Clock3 size={14} /> Total {result.total_inference_ms.toFixed(1)} ms</span></div>
      <div className="result-grid"><ResultCard prediction={result.ml} /><ResultCard prediction={result.cnn} /></div>
      <p className="score-explainer">Scores are model outputs and are not calibrated probabilities or verified diagnoses.</p>
    </section>
  );
}

function ResultCard({ prediction }: { prediction: Prediction }) {
  const isCnn = prediction.model === "cnn";
  return (
    <article className={`result-card ${isCnn ? "result-cnn" : "result-ml"}`}>
      <div className="result-card-head"><span className={`result-type ${isCnn ? "cnn-type" : "ml-type"}`}><span className="type-dot" />{isCnn ? "DEEP LEARNING" : "TRADITIONAL ML"}</span><span className="latency"><Clock3 size={13} />{prediction.inference_ms.toFixed(1)} ms</span></div>
      <div className="predicted-label">PREDICTED DATASET CLASS</div>
      <h3>{prediction.display_name}</h3>
      <div className="top-score-line"><span>Top model score</span><strong>{prediction.class_score.toFixed(4)}</strong></div>
      <div className="score-list">
        {prediction.top_classes.map((item, index) => <div className="score-row" key={item.class_name}><span className="score-rank">0{index + 1}</span><span className="score-name">{item.display_name}</span><span className="score-value">{item.score.toFixed(4)}</span></div>)}
      </div>
      {prediction.model_version && <div className="model-version">Model version · {prediction.model_version}</div>}
    </article>
  );
}

function MetricsPage({ metrics, error }: { metrics: MetricsResponse | null; error: string }) {
  return (
    <div className="page-content inner-page">
      <div className="page-title-row"><div><div className="eyebrow"><span className="eyebrow-dot" /> EVALUATION</div><h1>Model <em>metrics.</em></h1><p className="page-subtitle">Both models are evaluated on the same held-out test split.</p></div><div className="title-icon"><BarChart3 size={24} /></div></div>
      {error && <div className="error-box" role="alert">{error}</div>}
      {!metrics ? <LoadingState label="Loading evaluation results…" /> : (
        <>
          {metrics.comparison && <div className="comparison-banner"><div className="banner-icon"><Activity size={18} /></div><div><strong>CNN minus ML</strong><span>Difference on this project’s test split</span></div><div className="banner-values"><span>{formatSigned(metrics.comparison.accuracy_difference_cnn_minus_ml, "accuracy")}</span><span>{formatSigned(metrics.comparison.macro_f1_difference_cnn_minus_ml, "macro F1")}</span></div></div>}
          <div className="metric-model-grid"><ModelMetricsCard name="Traditional ML" tag="HOG + LINEAR SVM" model={metrics.models.ml} accent="amber" /><ModelMetricsCard name="Deep learning" tag="CONVOLUTIONAL NEURAL NETWORK" model={metrics.models.cnn} accent="green" /></div>
          {(metrics.models.ml.available || metrics.models.cnn.available) && <PerClassMetrics models={metrics.models} />}
          {!metrics.models.ml.available && !metrics.models.cnn.available && <div className="empty-card"><div className="empty-icon"><BarChart3 size={22} /></div><h2>Metrics will appear after training</h2><p>No training metrics are present yet. Prepare the dataset and run the training commands in the README; this page never fills in placeholder results.</p></div>}
        </>
      )}
      <div className="limitation-strip"><ShieldCheck size={16} /><div><strong>Read metrics with context.</strong><span>Performance on curated PlantVillage images does not establish performance on real-world field photographs.</span></div></div>
    </div>
  );
}

function formatSigned(value: number, label: string) {
  const points = (value * 100).toFixed(2);
  const Icon = value >= 0 ? ArrowUpRight : ArrowDownRight;
  return <span className="delta"><Icon size={15} />{value > 0 ? "+" : ""}{points} pp <small>{label}</small></span>;
}

function ModelMetricsCard({ name, tag, model, accent }: { name: string; tag: string; model: MetricsResponse["models"]["ml"]; accent: string }) {
  const values = model.metrics;
  return (
    <section className={`panel metric-model-card ${accent}`}>
      <div className="metric-card-title"><div><span className="step-label">{tag}</span><h2>{name}</h2></div><span className={`availability ${model.available ? "available" : "unavailable"}`}><i />{model.available ? "TRAINED" : "NOT TRAINED"}</span></div>
      {!values ? <p className="not-available">Metrics not available yet.</p> : <>
        <div className="metric-kpis">
          <Kpi label="Accuracy" value={values.accuracy} primary />
          <Kpi label="Macro precision" value={values.macro_precision} />
          <Kpi label="Macro recall" value={values.macro_recall} />
          <Kpi label="Macro F1" value={values.macro_f1} />
        </div>
        <div className="metric-meta"><span><Clock3 size={13} /> {values.average_inference_ms_per_image.toFixed(2)} ms / image</span><span><Database size={13} /> {formatBytes(values.model_size_bytes)}</span></div>
        <ConfusionMatrix metrics={values} />
      </>}
    </section>
  );
}

function Kpi({ label, value, primary = false }: { label: string; value: number; primary?: boolean }) {
  return <div className={`kpi ${primary ? "primary-kpi" : ""}`}><span>{label}</span><strong>{(value * 100).toFixed(1)}<small>%</small></strong></div>;
}

function ConfusionMatrix({ metrics }: { metrics: NonNullable<MetricsResponse["models"]["ml"]["metrics"]> }) {
  const names = metrics.class_names;
  return (
    <details className="matrix-details">
      <summary>Confusion matrix <span>actual × predicted</span></summary>
      <div className="matrix-scroll"><table className="matrix-table"><thead><tr><th>Actual \ Predicted</th>{names.map((name) => <th key={name} title={name}>{shortName(name)}</th>)}</tr></thead><tbody>{metrics.confusion_matrix.map((row, rowIndex) => <tr key={names[rowIndex]}><th title={names[rowIndex]}>{shortName(names[rowIndex])}</th>{row.map((count, columnIndex) => <td className={rowIndex === columnIndex ? "matrix-diagonal" : ""} key={`${rowIndex}-${columnIndex}`}>{count}</td>)}</tr>)}</tbody></table></div>
    </details>
  );
}

function PerClassMetrics({ models }: { models: MetricsResponse["models"] }) {
  const selectedModels = [
    { key: "ml" as const, label: "Traditional ML" },
    { key: "cnn" as const, label: "Deep learning" },
  ].filter(({ key }) => models[key].available && models[key].metrics);
  if (!selectedModels.length) return null;
  return (
    <div className="per-class-grid">{selectedModels.map(({ key, label }) => {
      const model = models[key].metrics;
      if (!model) return null;
      const rows = model.class_names.map((name) => {
        const report = model.classification_report[name] ?? {};
        return {
          name,
          precision: report.precision ?? 0,
          recall: report.recall ?? 0,
          f1: report["f1-score"] ?? 0,
          support: report.support ?? 0,
        };
      });
      return <section className="panel per-class-panel" key={key}><div className="panel-heading"><div><span className="step-label">DETAILED RESULTS · {key.toUpperCase()}</span><h2>{label} per-class performance</h2></div></div><div className="table-scroll"><table className="data-table"><thead><tr><th>Dataset class</th><th>Precision</th><th>Recall</th><th>F1 score</th><th>Support</th></tr></thead><tbody>{rows.map((row) => <tr key={row.name}><td>{row.name}</td><td>{(row.precision * 100).toFixed(1)}%</td><td>{(row.recall * 100).toFixed(1)}%</td><td>{(row.f1 * 100).toFixed(1)}%</td><td>{row.support}</td></tr>)}</tbody></table></div></section>;
    })}</div>
  );
}

function HistoryPage({ items, error, onClear }: { items: HistoryItem[] | null; error: string; onClear: () => void }) {
  return (
    <div className="page-content inner-page">
      <div className="page-title-row"><div><div className="eyebrow"><span className="eyebrow-dot" /> ACTIVITY</div><h1>Prediction <em>history.</em></h1><p className="page-subtitle">Recent comparisons and individual model runs from this local app.</p></div><button className="secondary-button" onClick={onClear} disabled={!items?.length}><Trash2 size={15} /> Clear history</button></div>
      {error && <div className="error-box" role="alert">{error}</div>}
      {items === null ? <LoadingState label="Loading prediction history…" /> : items.length === 0 ? <div className="empty-card"><div className="empty-icon"><Clock3 size={22} /></div><h2>No predictions recorded</h2><p>Run a model comparison from the workspace. The app stores result metadata, never the image itself.</p></div> : (
        <div className="history-list">{items.map((item) => <article className="history-card" key={item.id}><div className="history-date"><span>{new Date(item.created_at).toLocaleDateString()}</span><small>{new Date(item.created_at).toLocaleTimeString()}</small></div><div className="history-mode"><span className="history-mode-icon"><FlaskConical size={16} /></span><span>{item.mode === "compare" ? "Model comparison" : `${item.mode.toUpperCase()} prediction`}</span></div><div className="history-predictions">{item.predictions.map((prediction) => <span className="history-prediction" key={`${item.id}-${prediction.model}`}><i className={prediction.model === "cnn" ? "green-dot" : "amber-dot"} />{prediction.model.toUpperCase()}: {prediction.display_name}</span>)}</div><div className="history-latency"><Clock3 size={13} />{item.latency_ms.toFixed(1)} ms</div></article>)}</div>
      )}
      <div className="privacy-card"><ShieldCheck size={18} /><div><strong>Only prediction metadata is stored</strong><p>History includes the model, predicted class, scores, timestamp, and processing time. Uploaded image files are not saved.</p></div></div>
    </div>
  );
}

function LoadingState({ label }: { label: string }) {
  return <div className="loading-state"><LoaderCircle className="spin" size={19} />{label}</div>;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function shortName(name: string) {
  return name.replace("___", " · ").replace(/_/g, " ");
}

export default App;
