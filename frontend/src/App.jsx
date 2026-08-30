import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  AlertTriangle, Activity, FileWarning, Info,
  RotateCcw, FolderOpen, Check, ArrowRight, Eraser, Undo2, Loader2,
  ChevronLeft, ChevronRight,
} from "lucide-react";

// ============================================================================
// DESIGN TOKENS
// ============================================================================
const COLORS = {
  paper: "#F7F8FA", panel: "#FFFFFF", ink: "#101B2E", inkSoft: "#4B5768",
  hairline: "#D8DCE2", teal: "#0E7C86", tealSoft: "#E4F2F2",
  violet: "#6B4C7A", violetSoft: "#F1EAF4",
  healthy: "#2E8B57", stageA: "#C9A227", stageB: "#D97B29", advanced: "#B23A2E",
};

const STAGES = [
  { key: "Healthy", label: "Healthy", color: COLORS.healthy },
  { key: "A", label: "Stage A", color: COLORS.stageA },
  { key: "B", label: "Stage B", color: COLORS.stageB },
  { key: "Advanced", label: "Advanced", color: COLORS.advanced },
];

const STEPS = [
  { n: 1, label: "Post-contrast" },
  { n: 2, label: "Pre-contrast" },
  { n: 3, label: "Subtract & Draw ROI" },
  { n: 4, label: "Extracted Features" },
  { n: 5, label: "Prediction" },
];

// ============================================================================
// FEATURE SCHEMA (same 25-feature/12-TOP schema used throughout the study)
// ============================================================================
const FEATURE_META = {
  Volume: { label: "Volume", top: false, group: "morph" },
  Area: { label: "Area", top: false, group: "morph" },
  MaxDiameter: { label: "Max Diameter", top: true, group: "morph" },
  SurfaceArea: { label: "Surface Area (perimeter)", top: true, group: "morph" },
  Sphericity: { label: "Sphericity", top: false, group: "morph" },
  Compactness: { label: "Compactness", top: false, group: "morph" },
  Elongation: { label: "Elongation", top: false, group: "morph" },
  Mean: { label: "Mean", top: true, group: "first" },
  Median: { label: "Median", top: true, group: "first" },
  Min: { label: "Min", top: true, group: "first" },
  Max: { label: "Max", top: true, group: "first" },
  Std: { label: "Std Dev", top: true, group: "first" },
  Skewness: { label: "Skewness", top: false, group: "first" },
  Kurtosis: { label: "Kurtosis", top: false, group: "first" },
  Entropy: { label: "Entropy", top: true, group: "first" },
  GLCM_Contrast: { label: "GLCM Contrast", top: true, group: "glcm" },
  GLCM_Correlation: { label: "GLCM Correlation", top: true, group: "glcm" },
  GLCM_Homogeneity: { label: "GLCM Homogeneity", top: false, group: "glcm" },
  GLCM_Energy: { label: "GLCM Energy", top: false, group: "glcm" },
  GLCM_Entropy: { label: "GLCM Entropy", top: false, group: "glcm" },
  SRE: { label: "Short Run Emphasis", top: false, group: "glcm" },
  LRE: { label: "Long Run Emphasis", top: false, group: "glcm" },
  GLN: { label: "Gray-Level Nonuniformity", top: false, group: "glcm" },
  LiverEntropy: { label: "Liver-context Entropy*", top: true, group: "glcm" },
  TumorLiverContrast: { label: "Tumor-Liver Contrast*", top: true, group: "glcm" },
};
const GROUP_LABELS = { morph: "Morphological (7)", first: "First-order Intensity (8)", glcm: "GLCM / Run-length (10)" };

// ============================================================================
// IMAGE HELPERS
// ============================================================================
function loadDataUrlToCanvas(dataUrl, canvasEl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      canvasEl.width = img.width;
      canvasEl.height = img.height;
      canvasEl.getContext("2d").drawImage(img, 0, 0);
      resolve({ width: img.width, height: img.height });
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

// ============================================================================
// BACKEND API — series upload/subtract/extract + model inference.
// A browser cannot parse raw DICOM or execute the trained .joblib pickles
// directly, so the local FastAPI server handles both. See backend/SETUP.md
// if these calls fail (most likely cause: API not running).
// ============================================================================
const STAGING_MODEL_OPTIONS = ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting"];

async function postJSON(apiBase, path, body) {
  const res = await fetch(`${apiBase}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} — ${text}`);
  }
  return res.json();
}

async function uploadDicomSeries(apiBase, phase, files) {
  const form = new FormData();
  form.append("phase", phase);
  files.forEach((f) => form.append("files", f, f.name));
  const res = await fetch(`${apiBase}/series/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} — ${text}`);
  }
  return res.json();
}

function subtractDicomSeries(apiBase, postSessionId, preSessionId) {
  return postJSON(apiBase, "/series/subtract", {
    post_session_id: postSessionId,
    pre_session_id: preSessionId,
  });
}

function extractFeaturesFromSlice(apiBase, postSessionId, preSessionId, sliceIndex, roi) {
  return postJSON(apiBase, "/series/extract", {
    post_session_id: postSessionId,
    pre_session_id: preSessionId,
    slice_index: sliceIndex,
    roi,
  });
}

function deleteSessionBestEffort(apiBase, sessionId) {
  if (!sessionId) return;
  fetch(`${apiBase}/series/session/${sessionId}`, { method: "DELETE" }).catch(() => {});
}

async function runRealModel(apiBase, stagingModel, features) {
  const [stageRes, necroticRes] = await Promise.all([
    postJSON(apiBase, "/predict/stage", { model: stagingModel, features }),
    postJSON(apiBase, "/predict/necrotic", { features }),
  ]);

  // Prefer the staging model's SHAP contributions for the evidence panel;
  // fall back to the necrotic model's if the staging ones are unavailable.
  const contributions =
    (stageRes.contributions && stageRes.contributions.length ? stageRes.contributions : necroticRes.contributions) || [];

  return {
    stageProbs: stageRes.stageProbs,
    necroticProb: necroticRes.necroticProb,
    contributions,
  };
}

const fmt = (v) => (v === undefined || v === null || Number.isNaN(v) ? "—" : Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(3));

// ============================================================================
// UI PIECES
// ============================================================================
function Stepper({ step }) {
  return (
    <div className="flex items-center w-full overflow-x-auto py-1">
      {STEPS.map((s, i) => (
        <React.Fragment key={s.n}>
          <div className="flex items-center gap-2 shrink-0">
            <div
              className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-semibold"
              style={{
                background: step === s.n ? COLORS.teal : step > s.n ? COLORS.tealSoft : "transparent",
                color: step === s.n ? "#fff" : step > s.n ? COLORS.teal : "#9AA7B8",
                border: `1.5px solid ${step >= s.n ? COLORS.teal : "#D8DCE2"}`,
              }}
            >
              {step > s.n ? <Check size={14} /> : s.n}
            </div>
            <span className="text-xs font-medium whitespace-nowrap" style={{ color: step === s.n ? COLORS.ink : "#9AA7B8" }}>
              {s.label}
            </span>
          </div>
          {i < STEPS.length - 1 && (
            <div className="h-px flex-1 mx-3 min-w-6" style={{ background: step > s.n ? COLORS.teal : COLORS.hairline }} />
          )}
        </React.Fragment>
      ))}
    </div>
  );
}

function SeriesUploadStep({ title, subtitle, files, onFilesChange, onContinue, uploading, error, canGoBack, onBack }) {
  const inputRef = useRef(null);
  return (
    <div className="rounded-lg p-8 border flex flex-col items-center text-center" style={{ background: COLORS.panel, borderColor: COLORS.hairline }}>
      <div className="w-12 h-12 rounded-full flex items-center justify-center mb-3" style={{ background: COLORS.tealSoft }}>
        <FolderOpen size={22} color={COLORS.teal} />
      </div>
      <div className="text-lg font-semibold mb-1" style={{ color: COLORS.ink }}>{title}</div>
      <div className="text-sm mb-5" style={{ color: COLORS.inkSoft }}>{subtitle}</div>

      <input
        ref={inputRef}
        type="file"
        accept=".dcm"
        multiple
        className="hidden"
        onChange={(e) => e.target.files.length && onFilesChange(Array.from(e.target.files))}
      />

      {files.length > 0 && (
        <div className="w-full max-w-sm rounded border mb-4 p-3 text-left text-xs" style={{ borderColor: COLORS.hairline }}>
          <div className="font-semibold mb-1" style={{ color: COLORS.ink }}>
            {files.length} DICOM file{files.length === 1 ? "" : "s"} selected
          </div>
          <div className="max-h-24 overflow-y-auto space-y-0.5" style={{ fontFamily: "'IBM Plex Mono', monospace", color: COLORS.inkSoft }}>
            {files.slice(0, 6).map((f) => <div key={f.name} className="truncate">{f.name}</div>)}
            {files.length > 6 && <div>&hellip; and {files.length - 6} more</div>}
          </div>
        </div>
      )}

      {error && (
        <div className="text-xs mb-3 px-3 py-2 rounded flex gap-2 items-start text-left" style={{ background: "#FBEAE8", color: "#7A2A21" }}>
          <AlertTriangle size={13} className="shrink-0 mt-0.5" /> {error}
        </div>
      )}

      {files.length === 0 ? (
        <button onClick={() => inputRef.current.click()} className="px-4 py-2 rounded-md text-sm font-medium text-white" style={{ background: COLORS.teal }}>
          Choose DICOM files&hellip;
        </button>
      ) : (
        <div className="w-full flex flex-col items-center">
          <div className="flex gap-2">
            <button onClick={() => inputRef.current.click()} className="px-3 py-1.5 rounded-md text-xs border" style={{ borderColor: COLORS.hairline, color: COLORS.inkSoft }}>
              Replace files
            </button>
            <button
              onClick={onContinue}
              disabled={uploading}
              className="px-4 py-1.5 rounded-md text-xs font-medium text-white flex items-center gap-1 disabled:opacity-50"
              style={{ background: COLORS.teal }}
            >
              {uploading ? (<><Loader2 size={13} className="animate-spin" /> Uploading&hellip;</>) : (<>Continue <ArrowRight size={13} /></>)}
            </button>
          </div>
        </div>
      )}

      {canGoBack && (
        <button onClick={onBack} className="text-xs mt-4" style={{ color: COLORS.inkSoft }}>
          &larr; Back
        </button>
      )}
    </div>
  );
}

function SliceCarousel({ slices, onSelect }) {
  const [idx, setIdx] = useState(0);

  const goPrev = useCallback(() => setIdx((i) => Math.max(0, i - 1)), []);
  const goNext = useCallback(() => setIdx((i) => Math.min(slices.length - 1, i + 1)), [slices.length]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "ArrowLeft") { e.preventDefault(); goPrev(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); goNext(); }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [goPrev, goNext]);

  const current = slices[idx];

  return (
    <div>
      <div className="text-xs mb-3 px-3 py-2 rounded flex gap-2 items-start" style={{ background: COLORS.tealSoft, color: COLORS.teal }}>
        <Info size={14} className="shrink-0 mt-0.5" />
        {slices.length} matching slices were aligned and subtracted (post &minus; pre).
        Use the &larr; &rarr; arrow keys to step through the stack and pick the slice with the clearest tumor enhancement.
      </div>

      <div
        className="relative flex items-center justify-center"
        style={{ background: COLORS.ink, borderRadius: 8, border: `1px solid ${COLORS.hairline}`, minHeight: 480 }}
      >
        <button
          onClick={goPrev}
          disabled={idx === 0}
          aria-label="Previous slice"
          className="absolute left-2 z-10 rounded-full p-1.5 disabled:opacity-25 hover:bg-white/10 transition"
        >
          <ChevronLeft size={32} color="white" />
        </button>

        <img
          key={current.index}
          src={current.image_data_b64}
          alt={`Slice ${current.index}`}
          draggable={false}
          className="block select-none"
          style={{ maxHeight: 600, maxWidth: "100%", objectFit: "contain" }}
        />

        <button
          onClick={goNext}
          disabled={idx === slices.length - 1}
          aria-label="Next slice"
          className="absolute right-2 z-10 rounded-full p-1.5 disabled:opacity-25 hover:bg-white/10 transition"
        >
          <ChevronRight size={32} color="white" />
        </button>
      </div>

      <div
        className="flex items-center justify-between mt-2 text-xs"
        style={{ fontFamily: "'IBM Plex Mono', monospace", color: COLORS.inkSoft }}
      >
        <span>Slice #{current.index}</span>
        <span>{idx + 1} / {slices.length}</span>
      </div>

      <div className="text-center mt-4">
        <button
          onClick={() => onSelect(current)}
          className="px-4 py-1.5 rounded-md text-xs font-medium text-white"
          style={{ background: COLORS.teal }}
        >
          Draw ROI on slice #{current.index}
        </button>
      </div>
    </div>
  );
}

function SectionHeader({ eyebrow, title }) {
  return (
    <div className="mb-4">
      <div className="text-xs font-semibold tracking-widest uppercase" style={{ color: COLORS.teal, fontFamily: "'IBM Plex Mono', monospace" }}>{eyebrow}</div>
      <div className="text-lg font-semibold" style={{ color: COLORS.ink }}>{title}</div>
    </div>
  );
}

function StageGauge({ stageProbs }) {
  const order = ["Healthy", "A", "B", "Advanced"];
  const centers = { Healthy: 12.5, A: 37.5, B: 62.5, Advanced: 87.5 };
  const markerPos = order.reduce((acc, k) => acc + stageProbs[k] * centers[k], 0);
  const topClass = order.reduce((a, b) => (stageProbs[a] > stageProbs[b] ? a : b));
  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-sm font-semibold" style={{ color: COLORS.ink }}>BCLC Stage</div>
        <div className="text-xs px-2 py-0.5 rounded-full font-semibold" style={{ background: STAGES.find((s) => s.key === topClass).color + "20", color: STAGES.find((s) => s.key === topClass).color }}>
          Most likely: {STAGES.find((s) => s.key === topClass).label}
        </div>
      </div>
      <div className="relative h-9 rounded-md overflow-hidden flex mb-1">
        {STAGES.map((s) => (<div key={s.key} style={{ background: s.color, width: "25%" }} />))}
        <div className="absolute top-0 bottom-0" style={{ left: `calc(${markerPos}% - 1.5px)`, width: "3px", background: COLORS.ink }} />
        <div className="absolute -top-1.5" style={{ left: `calc(${markerPos}% - 6px)`, width: 0, height: 0, borderLeft: "6px solid transparent", borderRight: "6px solid transparent", borderTop: `7px solid ${COLORS.ink}` }} />
      </div>
      <div className="flex text-[11px] mb-3" style={{ color: COLORS.inkSoft }}>
        {STAGES.map((s) => (<div key={s.key} style={{ width: "25%" }} className="text-center">{s.label}</div>))}
      </div>
      <div className="grid grid-cols-4 gap-2">
        {STAGES.map((s) => (
          <div key={s.key} className="text-center">
            <div className="text-base font-semibold tabular-nums" style={{ fontFamily: "'IBM Plex Mono', monospace", color: s.color }}>{(stageProbs[s.key] * 100).toFixed(0)}%</div>
            <div className="text-[11px]" style={{ color: COLORS.inkSoft }}>{s.label}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function NecroticReadout({ necroticProb }) {
  const pct = necroticProb * 100;
  const label = necroticProb >= 0.5 ? "Necrotic" : "Non-necrotic";
  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <div className="text-sm font-semibold" style={{ color: COLORS.ink }}>Tissue Composition</div>
        <div className="text-xs px-2 py-0.5 rounded-full font-semibold" style={{ background: COLORS.violetSoft, color: COLORS.violet }}>Most likely: {label}</div>
      </div>
      <div className="relative h-9 rounded-md overflow-hidden mb-1" style={{ background: "#EDEBF0" }}>
        <div className="absolute inset-y-0 left-0" style={{ width: `${pct}%`, background: COLORS.violet, opacity: 0.85 }} />
        <div className="absolute top-0 bottom-0" style={{ left: `calc(${pct}% - 1.5px)`, width: "3px", background: COLORS.ink }} />
      </div>
      <div className="flex justify-between text-[11px]" style={{ color: COLORS.inkSoft }}>
        <span>Non-necrotic</span>
        <span className="font-semibold tabular-nums" style={{ fontFamily: "'IBM Plex Mono', monospace", color: COLORS.violet }}>{pct.toFixed(0)}% necrotic</span>
        <span>Necrotic</span>
      </div>
    </div>
  );
}

function EvidenceStrip({ contributions }) {
  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.value)), 0.01);
  return (
    <div>
      <div className="text-sm font-semibold mb-3" style={{ color: COLORS.ink }}>Evidence for this prediction</div>
      <div className="space-y-2.5">
        {contributions.map((c) => {
          const widthPct = (Math.abs(c.value) / maxAbs) * 100;
          const positive = c.value >= 0;
          return (
            <div key={c.name}>
              <div className="flex justify-between text-xs mb-1">
                <span style={{ color: COLORS.ink }}>{c.name}</span>
                <span className="tabular-nums font-medium" style={{ fontFamily: "'IBM Plex Mono', monospace", color: positive ? COLORS.advanced : COLORS.teal }}>{positive ? "+" : ""}{c.value.toFixed(2)}</span>
              </div>
              <div className="h-2 rounded-full" style={{ background: "#EEF0F2" }}>
                <div className="h-2 rounded-full" style={{ width: `${widthPct}%`, background: positive ? COLORS.advanced : COLORS.teal }} />
              </div>
            </div>
          );
        })}
      </div>
      <div className="text-[11px] mt-3" style={{ color: COLORS.inkSoft }}>Positive values push toward higher stage / necrosis; negative values push toward healthy / viable tissue.</div>
    </div>
  );
}

// ============================================================================
// MAIN APP
// ============================================================================
export default function App() {
  const [step, setStep] = useState(1);
  const [patient, setPatient] = useState({ id: "HCC-0000", age: "" });
  const [apiBase, setApiBase] = useState(import.meta.env.VITE_API_BASE_URL || "http://localhost:8000");
  const [stagingModel, setStagingModel] = useState("XGBoost");
  const [apiStatus, setApiStatus] = useState("unknown"); // unknown | ok | error
  const [predictLoading, setPredictLoading] = useState(false);
  const [predictError, setPredictError] = useState(null);
  const [result, setResult] = useState(null);

  const subCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);

  // Step 1/2 — DICOM series upload
  const [postFiles, setPostFiles] = useState([]);
  const [preFiles, setPreFiles] = useState([]);
  const [postSessionId, setPostSessionId] = useState(null);
  const [preSessionId, setPreSessionId] = useState(null);
  const [postUploading, setPostUploading] = useState(false);
  const [preUploading, setPreUploading] = useState(false);
  const [postUploadError, setPostUploadError] = useState(null);
  const [preUploadError, setPreUploadError] = useState(null);

  // Step 3 — subtraction, slice selection, ROI
  const [subtractedSeries, setSubtractedSeries] = useState(null);
  const [subtractLoading, setSubtractLoading] = useState(false);
  const [subtractError, setSubtractError] = useState(null);
  const [selectedSlice, setSelectedSlice] = useState(null);
  const [polygon, setPolygon] = useState([]);
  const [drawing, setDrawing] = useState(false);
  const [extractLoading, setExtractLoading] = useState(false);
  const [extractError, setExtractError] = useState(null);

  const [features, setFeatures] = useState(null);

  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap";
    document.head.appendChild(link);
    return () => document.head.removeChild(link);
  }, []);

  const handlePostContinue = async () => {
    if (postFiles.length === 0) return;
    setPostUploading(true);
    setPostUploadError(null);
    try {
      const resp = await uploadDicomSeries(apiBase, "post-contrast", postFiles);
      setPostSessionId(resp.session_id);
      setStep(2);
    } catch (e) {
      setPostUploadError(e.message || "Upload failed. Check that the API is running.");
    } finally {
      setPostUploading(false);
    }
  };

  const handlePreContinue = async () => {
    if (preFiles.length === 0) return;
    setPreUploading(true);
    setPreUploadError(null);
    try {
      const resp = await uploadDicomSeries(apiBase, "pre-contrast", preFiles);
      setPreSessionId(resp.session_id);
      setStep(3);
    } catch (e) {
      setPreUploadError(e.message || "Upload failed. Check that the API is running.");
    } finally {
      setPreUploading(false);
    }
  };

  // Trigger subtraction once both sessions exist and we've reached step 3.
  useEffect(() => {
    if (step === 3 && postSessionId && preSessionId && !subtractedSeries && !subtractLoading && !subtractError) {
      setSubtractLoading(true);
      subtractDicomSeries(apiBase, postSessionId, preSessionId)
        .then((resp) => setSubtractedSeries(resp.subtracted_series))
        .catch((e) => setSubtractError(e.message || "Subtraction failed. Check that both series uploaded correctly."))
        .finally(() => setSubtractLoading(false));
    }
  }, [step, postSessionId, preSessionId, subtractedSeries, subtractLoading, subtractError, apiBase]);

  // Load the selected slice's PNG into the drawing canvas.
  useEffect(() => {
    if (!selectedSlice || !subCanvasRef.current) return;
    loadDataUrlToCanvas(selectedSlice.image_data_b64, subCanvasRef.current).then(() => {
      if (overlayCanvasRef.current) {
        overlayCanvasRef.current.width = subCanvasRef.current.width;
        overlayCanvasRef.current.height = subCanvasRef.current.height;
      }
    });
  }, [selectedSlice]);

  // Redraw the ROI overlay whenever the polygon changes.
  useEffect(() => {
    if (!selectedSlice || !overlayCanvasRef.current) return;
    const o = overlayCanvasRef.current;
    const octx = o.getContext("2d");
    octx.clearRect(0, 0, o.width, o.height);
    if (polygon.length > 0) {
      octx.strokeStyle = "#0E7C86";
      octx.fillStyle = "rgba(14,124,134,0.18)";
      octx.lineWidth = 2;
      octx.beginPath();
      octx.moveTo(polygon[0].x, polygon[0].y);
      polygon.slice(1).forEach((p) => octx.lineTo(p.x, p.y));
      if (!drawing) octx.closePath();
      octx.stroke();
      if (!drawing) octx.fill();
    }
  }, [selectedSlice, polygon, drawing]);

  const handleCanvasMouseDown = (e) => {
    const rect = overlayCanvasRef.current.getBoundingClientRect();
    const scaleX = overlayCanvasRef.current.width / rect.width;
    const scaleY = overlayCanvasRef.current.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    setDrawing(true);
    setPolygon([{ x, y }]);
  };
  const handleCanvasMouseMove = (e) => {
    if (!drawing) return;
    const rect = overlayCanvasRef.current.getBoundingClientRect();
    const scaleX = overlayCanvasRef.current.width / rect.width;
    const scaleY = overlayCanvasRef.current.height / rect.height;
    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;
    setPolygon((p) => [...p, { x, y }]);
  };
  const handleCanvasMouseUp = () => setDrawing(false);

  const handleSelectSlice = (slice) => {
    setSelectedSlice(slice);
    setPolygon([]);
    setExtractError(null);
  };
  const handleChangeSlice = () => {
    setSelectedSlice(null);
    setPolygon([]);
    setExtractError(null);
  };

  const clearROI = () => { setPolygon([]); setExtractError(null); };
  const undoROI = () => { setPolygon([]); };

  const finalizeROIAndExtract = async () => {
    if (!selectedSlice || polygon.length < 3) return;
    setExtractLoading(true);
    setExtractError(null);
    try {
      const resp = await extractFeaturesFromSlice(apiBase, postSessionId, preSessionId, selectedSlice.index, polygon);
      setFeatures(resp.features);
      setStep(4);
    } catch (e) {
      setExtractError(e.message || "Feature extraction failed. Check that the API is running and the ROI is valid.");
    } finally {
      setExtractLoading(false);
    }
  };

  const checkApiHealth = async () => {
    try {
      const res = await fetch(`${apiBase}/health`);
      setApiStatus(res.ok ? "ok" : "error");
    } catch {
      setApiStatus("error");
    }
  };

  const runPrediction = async () => {
    if (!features) return;
    setPredictLoading(true);
    setPredictError(null);
    try {
      const r = await runRealModel(apiBase, stagingModel, features);
      setResult(r);
      setStep(5);
    } catch (e) {
      setPredictError(
        e.message ||
          "Could not reach the inference API. Make sure the backend is running " +
            "(python run.py) and the API base URL above is correct."
      );
    } finally {
      setPredictLoading(false);
    }
  };

  const startNewCase = () => {
    deleteSessionBestEffort(apiBase, postSessionId);
    deleteSessionBestEffort(apiBase, preSessionId);
    setStep(1);
    setPostFiles([]); setPreFiles([]);
    setPostSessionId(null); setPreSessionId(null);
    setPostUploadError(null); setPreUploadError(null);
    setSubtractedSeries(null); setSubtractError(null);
    setSelectedSlice(null); setPolygon([]);
    setExtractError(null);
    setFeatures(null);
    setResult(null); setPredictError(null);
  };

  const grouped = { morph: [], first: [], glcm: [] };
  Object.entries(FEATURE_META).forEach(([k, m]) => grouped[m.group].push(k));

  return (
    <div className="min-h-screen w-full" style={{ background: COLORS.paper, fontFamily: "'IBM Plex Sans', ui-sans-serif, system-ui, sans-serif" }}>
      <div className="w-full border-b px-6 py-3" style={{ background: COLORS.ink, borderColor: COLORS.hairline }}>
        <div className="flex flex-wrap items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-md flex items-center justify-center" style={{ background: COLORS.teal }}>
              <Activity size={18} color="#fff" />
            </div>
            <div>
              <div className="text-white text-sm font-semibold tracking-wide">HCC RADIOMICS &middot; DECISION SUPPORT</div>
            </div>
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <label className="text-xs text-white/70 flex items-center gap-2">
              Patient ID
              <input value={patient.id} onChange={(e) => setPatient((p) => ({ ...p, id: e.target.value }))} className="px-2 py-1 rounded text-sm w-28" style={{ fontFamily: "'IBM Plex Mono', monospace" }} />
            </label>
            <label className="text-xs text-white/70 flex items-center gap-2">
              Age
              <input value={patient.age} onChange={(e) => setPatient((p) => ({ ...p, age: e.target.value }))} placeholder="—" className="px-2 py-1 rounded text-sm w-16" style={{ fontFamily: "'IBM Plex Mono', monospace" }} />
            </label>
          </div>
        </div>
        <Stepper step={step} />
      </div>

      <div className="max-w-6xl mx-auto p-6">
        {step === 1 && (
          <SeriesUploadStep
            title="Step 1 — Upload post-contrast DICOM series"
            files={postFiles}
            onFilesChange={setPostFiles}
            onContinue={handlePostContinue}
            uploading={postUploading}
            error={postUploadError}
            canGoBack={false}
          />
        )}

        {step === 2 && (
          <SeriesUploadStep
            title="Step 2 — Upload pre-contrast DICOM series"
            files={preFiles}
            onFilesChange={setPreFiles}
            onContinue={handlePreContinue}
            uploading={preUploading}
            error={preUploadError}
            canGoBack={true}
            onBack={() => setStep(1)}
          />
        )}

        {step === 3 && (
          <div className="rounded-lg p-6 border" style={{ background: COLORS.panel, borderColor: COLORS.hairline }}>
            <SectionHeader eyebrow="Step 3" title="Subtracted series — select slice & draw tumor ROI" />

            {subtractLoading && (
              <div className="text-sm flex items-center gap-2" style={{ color: COLORS.inkSoft }}>
                <Loader2 size={14} className="animate-spin" /> Aligning series and computing subtraction (post &minus; pre)&hellip;
              </div>
            )}

            {subtractError && !subtractLoading && (
              <div className="text-xs px-3 py-2 rounded flex gap-2 items-start" style={{ background: "#FBEAE8", color: "#7A2A21" }}>
                <AlertTriangle size={13} className="shrink-0 mt-0.5" />
                <div>
                  {subtractError}
                  <button onClick={() => setSubtractError(null)} className="ml-3 underline">Retry</button>
                </div>
              </div>
            )}

            {!subtractLoading && subtractedSeries && !selectedSlice && (
              <>
                <SliceCarousel slices={subtractedSeries} onSelect={handleSelectSlice} />
                <div className="text-center mt-4">
                  <button onClick={() => setStep(2)} className="text-xs" style={{ color: COLORS.inkSoft }}>&larr; Back to pre-contrast upload</button>
                </div>
              </>
            )}

            {!subtractLoading && subtractedSeries && selectedSlice && (
              <>
                <div className="text-xs mb-3 px-3 py-2 rounded flex gap-2 items-start" style={{ background: COLORS.tealSoft, color: COLORS.teal }}>
                  <Info size={14} className="shrink-0 mt-0.5" />
                  Click and drag to draw a smooth freehand outline around the
                  tumor on slice #{selectedSlice.index}. Release to close the outline.
                </div>
                <div style={{ display: "block", textAlign: "center" }}>
                  <div className="relative inline-block">
                    <canvas ref={subCanvasRef} className="rounded border" style={{ borderColor: COLORS.hairline, maxWidth: "100%", maxHeight: 420 }} />
                    <canvas
                      ref={overlayCanvasRef}
                      className="absolute top-0 left-0 rounded cursor-crosshair"
                      style={{ maxWidth: "100%", maxHeight: 420 }}
                      onMouseDown={handleCanvasMouseDown}
                      onMouseMove={handleCanvasMouseMove}
                      onMouseUp={handleCanvasMouseUp}
                      onMouseLeave={() => drawing && setDrawing(false)}
                    />
                  </div>
                </div>

                {extractError && (
                  <div className="text-xs mt-3 px-3 py-2 rounded flex gap-2 items-start" style={{ background: "#FBEAE8", color: "#7A2A21" }}>
                    <AlertTriangle size={13} className="shrink-0 mt-0.5" /> {extractError}
                  </div>
                )}

                <div className="flex items-center justify-center gap-3 mt-4">
                  <button onClick={undoROI} className="px-3 py-1.5 rounded-md text-xs border flex items-center gap-1" style={{ borderColor: COLORS.hairline, color: COLORS.inkSoft }}>
                    <Undo2 size={13} /> Redraw
                  </button>
                  <button onClick={clearROI} className="px-3 py-1.5 rounded-md text-xs border flex items-center gap-1" style={{ borderColor: COLORS.hairline, color: COLORS.inkSoft }}>
                    <Eraser size={13} /> Clear
                  </button>
                  <button
                    onClick={finalizeROIAndExtract}
                    disabled={polygon.length < 3 || extractLoading}
                    className="px-4 py-1.5 rounded-md text-xs font-medium text-white flex items-center gap-1 disabled:opacity-40"
                    style={{ background: COLORS.teal }}
                  >
                    {extractLoading ? (<><Loader2 size={13} className="animate-spin" /> Extracting&hellip;</>) : (<>Extract features from ROI <ArrowRight size={13} /></>)}
                  </button>
                </div>
                <div className="text-center mt-3 flex justify-center gap-4">
                  <button onClick={handleChangeSlice} className="text-xs" style={{ color: COLORS.inkSoft }}>&larr; Change slice</button>
                  <button onClick={() => setStep(2)} className="text-xs" style={{ color: COLORS.inkSoft }}>&larr; Back to pre-contrast upload</button>
                </div>
              </>
            )}
          </div>
        )}

        {step === 4 && features && (
          <div className="rounded-lg p-6 border" style={{ background: COLORS.panel, borderColor: COLORS.hairline }}>
            <SectionHeader eyebrow="Step 4" title="Features extracted from your drawn ROI" />
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {Object.entries(GROUP_LABELS).map(([gk, gl]) => (
                <div key={gk}>
                  <div className="text-sm font-semibold mb-2" style={{ color: COLORS.ink }}>{gl}</div>
                  {grouped[gk].map((k) => (
                    <div key={k} className="flex justify-between text-xs py-1.5 border-b" style={{ borderColor: COLORS.hairline }}>
                      <span className="flex items-center gap-1" style={{ color: COLORS.ink }}>
                        {FEATURE_META[k].top && <span className="text-[9px] font-bold px-1 rounded" style={{ background: COLORS.tealSoft, color: COLORS.teal }}>TOP</span>}
                        {FEATURE_META[k].label}
                      </span>
                      <span className="tabular-nums" style={{ fontFamily: "'IBM Plex Mono', monospace", color: COLORS.inkSoft }}>{fmt(features[k])}</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
            <div className="mt-6 pt-5 border-t" style={{ borderColor: COLORS.hairline }}>
              <div className="text-sm font-semibold mb-2" style={{ color: COLORS.ink }}>Inference settings</div>
              <div className="flex flex-wrap items-end gap-4 mb-3">
                <label className="text-xs" style={{ color: COLORS.inkSoft }}>
                  API base URL
                  <input
                    value={apiBase}
                    onChange={(e) => { setApiBase(e.target.value); setApiStatus("unknown"); }}
                    className="block mt-1 px-2 py-1 rounded border text-xs w-56"
                    style={{ borderColor: COLORS.hairline, fontFamily: "'IBM Plex Mono', monospace" }}
                  />
                </label>
                <label className="text-xs" style={{ color: COLORS.inkSoft }}>
                  Staging model
                  <select
                    value={stagingModel}
                    onChange={(e) => setStagingModel(e.target.value)}
                    className="block mt-1 px-2 py-1 rounded border text-xs"
                    style={{ borderColor: COLORS.hairline }}
                  >
                    {STAGING_MODEL_OPTIONS.map((m) => (<option key={m} value={m}>{m}</option>))}
                  </select>
                </label>
                <button onClick={checkApiHealth} className="px-3 py-1.5 rounded-md text-xs border" style={{ borderColor: COLORS.hairline, color: COLORS.inkSoft }}>
                  Test connection
                </button>
                {apiStatus === "ok" && <span className="text-xs font-medium" style={{ color: COLORS.healthy }}>&#9679; API reachable</span>}
                {apiStatus === "error" && <span className="text-xs font-medium" style={{ color: COLORS.advanced }}>&#9679; API unreachable</span>}
              </div>

              {predictError && (
                <div className="text-xs mb-3 px-3 py-2 rounded flex gap-2 items-start" style={{ background: "#FBEAE8", color: "#7A2A21" }}>
                  <AlertTriangle size={13} className="shrink-0 mt-0.5" /> {predictError}
                </div>
              )}

              <div className="flex justify-between items-center">
                <button onClick={() => setStep(3)} className="text-xs" style={{ color: COLORS.inkSoft }}>&larr; Back to ROI</button>
                <button
                  onClick={runPrediction}
                  disabled={predictLoading}
                  className="px-4 py-1.5 rounded-md text-xs font-medium text-white flex items-center gap-1 disabled:opacity-50"
                  style={{ background: COLORS.teal }}
                >
                  {predictLoading ? "Running your trained models…" : "Run prediction"} <ArrowRight size={13} />
                </button>
              </div>
            </div>
          </div>
        )}

        {step === 5 && result && (
          <div className="flex flex-col gap-6">
            <div className="rounded-lg p-5 border" style={{ background: COLORS.panel, borderColor: COLORS.hairline }}>
              <SectionHeader eyebrow="Step 5" title="Staging Readout" />
              <StageGauge stageProbs={result.stageProbs} />
            </div>
            <div className="rounded-lg p-5 border" style={{ background: COLORS.panel, borderColor: COLORS.hairline }}>
              <NecroticReadout necroticProb={result.necroticProb} />
            </div>
            <div className="rounded-lg p-5 border" style={{ background: COLORS.panel, borderColor: COLORS.hairline }}>
              <EvidenceStrip contributions={result.contributions} />
            </div>
            <div className="flex justify-between">
              <button onClick={() => setStep(4)} className="text-xs" style={{ color: COLORS.inkSoft }}>&larr; Back to features</button>
              <button
                onClick={startNewCase}
                className="flex items-center gap-1 text-xs px-3 py-1.5 rounded border"
                style={{ borderColor: COLORS.hairline, color: COLORS.inkSoft }}
              >
                <RotateCcw size={12} /> Start new case
              </button>
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
