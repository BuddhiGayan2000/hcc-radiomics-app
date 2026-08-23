import React, { useState, useRef, useEffect } from "react";
import {
  AlertTriangle, Activity, FileWarning, Info,
  RotateCcw, Upload, Check, ArrowRight, Eraser, Undo2,
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
function loadImageToCanvas(file, canvasEl, maxDim = 512) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const img = new Image();
      img.onload = () => {
        const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
        const w = Math.round(img.width * scale);
        const h = Math.round(img.height * scale);
        canvasEl.width = w;
        canvasEl.height = h;
        const ctx = canvasEl.getContext("2d");
        ctx.drawImage(img, 0, 0, w, h);
        resolve(ctx.getImageData(0, 0, w, h));
      };
      img.onerror = reject;
      img.src = e.target.result;
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function toGray(imageData) {
  const { data, width, height } = imageData;
  const out = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const r = data[i * 4], g = data[i * 4 + 1], b = data[i * 4 + 2];
    out[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return out;
}

function subtractImages(postData, preData) {
  const w = postData.width, h = postData.height;
  const post = toGray(postData);
  const pre = toGray(preData);
  const n = w * h;
  const raw = new Float32Array(n);
  for (let i = 0; i < n; i++) raw[i] = Math.max(0, post[i] - pre[i]);

  let max = 0;
  for (let i = 0; i < n; i++) if (raw[i] > max) max = raw[i];
  max = max || 1;

  const display = new ImageData(w, h);
  for (let i = 0; i < n; i++) {
    const v = Math.round((raw[i] / max) * 255);
    display.data[i * 4] = v;
    display.data[i * 4 + 1] = v;
    display.data[i * 4 + 2] = v;
    display.data[i * 4 + 3] = 255;
  }
  return { raw, display, width: w, height: h };
}

function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x, yi = poly[i].y, xj = poly[j].x, yj = poly[j].y;
    const intersect = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
    if (intersect) inside = !inside;
  }
  return inside;
}

function rasterizeMask(poly, width, height) {
  const mask = new Uint8Array(width * height);
  if (poly.length < 3) return mask;
  let minX = width, maxX = 0, minY = height, maxY = 0;
  poly.forEach((p) => {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  });
  minX = Math.max(0, Math.floor(minX)); maxX = Math.min(width - 1, Math.ceil(maxX));
  minY = Math.max(0, Math.floor(minY)); maxY = Math.min(height - 1, Math.ceil(maxY));
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      if (pointInPolygon(x + 0.5, y + 0.5, poly)) mask[y * width + x] = 1;
    }
  }
  return mask;
}

// ============================================================================
// FEATURE EXTRACTION (genuine computation from real pixel data + mask)
// ============================================================================
function shannonEntropy(vals, nBins = 32) {
  if (vals.length === 0) return NaN;
  let min = Infinity, max = -Infinity;
  for (const v of vals) { if (v < min) min = v; if (v > max) max = v; }
  if (max === min) return 0;
  const counts = new Array(nBins).fill(0);
  for (const v of vals) {
    let b = Math.floor(((v - min) / (max - min)) * nBins);
    if (b >= nBins) b = nBins - 1;
    counts[b]++;
  }
  const n = vals.length;
  let h = 0;
  for (const c of counts) if (c > 0) { const p = c / n; h -= p * Math.log2(p); }
  return h;
}

function firstOrderStats(vals) {
  const n = vals.length;
  if (n === 0) return { Mean: NaN, Median: NaN, Min: NaN, Max: NaN, Std: NaN, Skewness: NaN, Kurtosis: NaN, Entropy: NaN };
  const sorted = [...vals].sort((a, b) => a - b);
  const mean = vals.reduce((a, b) => a + b, 0) / n;
  const variance = vals.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  const std = Math.sqrt(variance);
  const skew = std > 0 ? vals.reduce((a, b) => a + ((b - mean) / std) ** 3, 0) / n : 0;
  const kurt = std > 0 ? vals.reduce((a, b) => a + ((b - mean) / std) ** 4, 0) / n : 0;
  return {
    Mean: mean, Median: sorted[Math.floor(n / 2)], Min: sorted[0], Max: sorted[n - 1],
    Std: std, Skewness: skew, Kurtosis: kurt, Entropy: shannonEntropy(vals),
  };
}

function shapeStats(mask, width, height) {
  let area = 0, sumX = 0, sumY = 0;
  const pts = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (mask[y * width + x]) { area++; sumX += x; sumY += y; pts.push([x, y]); }
    }
  }
  if (area === 0) {
    return { Volume: 0, Area: 0, MaxDiameter: 0, SurfaceArea: 0, Sphericity: 0, Compactness: 0, Elongation: 1 };
  }
  const cx = sumX / area, cy = sumY / area;

  let perimeter = 0;
  const boundary = [];
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (!mask[y * width + x]) continue;
      const up = y > 0 ? mask[(y - 1) * width + x] : 0;
      const down = y < height - 1 ? mask[(y + 1) * width + x] : 0;
      const left = x > 0 ? mask[y * width + x - 1] : 0;
      const right = x < width - 1 ? mask[y * width + x + 1] : 0;
      if (!up || !down || !left || !right) { perimeter++; boundary.push([x, y]); }
    }
  }

  const step = Math.max(1, Math.floor(boundary.length / 150));
  const sample = boundary.filter((_, i) => i % step === 0);
  let maxDiameter = 0;
  for (let i = 0; i < sample.length; i++) {
    for (let j = i + 1; j < sample.length; j++) {
      const dx = sample[i][0] - sample[j][0], dy = sample[i][1] - sample[j][1];
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d > maxDiameter) maxDiameter = d;
    }
  }

  let sxx = 0, syy = 0, sxy = 0;
  for (const [x, y] of pts) { sxx += (x - cx) ** 2; syy += (y - cy) ** 2; sxy += (x - cx) * (y - cy); }
  sxx /= area; syy /= area; sxy /= area;
  const tr = sxx + syy, det = sxx * syy - sxy * sxy;
  const disc = Math.sqrt(Math.max(0, (tr * tr) / 4 - det));
  const l1 = tr / 2 + disc, l2 = Math.max(1e-6, tr / 2 - disc);
  const elongation = Math.sqrt(l1 / l2);

  const sphericity = (4 * Math.PI * area) / (perimeter * perimeter + 1e-9);
  const compactness = area / (perimeter * perimeter + 1e-9);

  return {
    Volume: area, Area: area, MaxDiameter: maxDiameter, SurfaceArea: perimeter,
    Sphericity: sphericity, Compactness: compactness, Elongation: elongation,
  };
}

function bbox(mask, width, height) {
  let minX = width, maxX = -1, minY = height, maxY = -1;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    if (mask[y * width + x]) { if (x < minX) minX = x; if (x > maxX) maxX = x; if (y < minY) minY = y; if (y > maxY) maxY = y; }
  }
  return maxX < 0 ? null : { minX, maxX, minY, maxY };
}

function glcmFeatures(raw, mask, width, height, nLevels = 24) {
  const bb = bbox(mask, width, height);
  const empty = { GLCM_Contrast: NaN, GLCM_Correlation: NaN, GLCM_Homogeneity: NaN, GLCM_Energy: NaN, GLCM_Entropy: NaN };
  if (!bb) return empty;
  const roiVals = [];
  for (let y = bb.minY; y <= bb.maxY; y++) for (let x = bb.minX; x <= bb.maxX; x++)
    if (mask[y * width + x]) roiVals.push(raw[y * width + x]);
  if (roiVals.length === 0) return empty;
  const min = Math.min(...roiVals), max = Math.max(...roiVals);
  if (max === min) return empty;

  const q = new Int16Array(width * height).fill(-1);
  for (let y = bb.minY; y <= bb.maxY; y++) for (let x = bb.minX; x <= bb.maxX; x++) {
    if (!mask[y * width + x]) continue;
    let lvl = Math.floor(((raw[y * width + x] - min) / (max - min)) * (nLevels - 1));
    lvl = Math.max(0, Math.min(nLevels - 1, lvl));
    q[y * width + x] = lvl;
  }

  const glcm = Array.from({ length: nLevels }, () => new Float64Array(nLevels));
  const offsets = [[1, 0], [1, 1], [0, 1], [-1, 1]];
  let total = 0;
  for (const [dx, dy] of offsets) {
    for (let y = bb.minY; y <= bb.maxY; y++) for (let x = bb.minX; x <= bb.maxX; x++) {
      const a = q[y * width + x]; if (a < 0) continue;
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
      const b = q[ny * width + nx]; if (b < 0) continue;
      glcm[a][b]++; glcm[b][a]++; total += 2;
    }
  }
  if (total === 0) return empty;
  for (let i = 0; i < nLevels; i++) for (let j = 0; j < nLevels; j++) glcm[i][j] /= total;

  let contrast = 0, energy = 0, entropy = 0, meanI = 0, meanJ = 0;
  for (let i = 0; i < nLevels; i++) for (let j = 0; j < nLevels; j++) {
    const p = glcm[i][j];
    contrast += p * (i - j) ** 2;
    energy += p * p;
    if (p > 0) entropy -= p * Math.log2(p);
    meanI += i * p; meanJ += j * p;
  }
  let varI = 0, varJ = 0, correlation = 0, homogeneity = 0;
  for (let i = 0; i < nLevels; i++) for (let j = 0; j < nLevels; j++) {
    const p = glcm[i][j];
    varI += p * (i - meanI) ** 2; varJ += p * (j - meanJ) ** 2;
    homogeneity += p / (1 + Math.abs(i - j));
  }
  const stdI = Math.sqrt(varI), stdJ = Math.sqrt(varJ);
  if (stdI > 0 && stdJ > 0) {
    for (let i = 0; i < nLevels; i++) for (let j = 0; j < nLevels; j++)
      correlation += (glcm[i][j] * (i - meanI) * (j - meanJ)) / (stdI * stdJ);
  }

  return { GLCM_Contrast: contrast, GLCM_Correlation: correlation, GLCM_Homogeneity: homogeneity, GLCM_Energy: energy, GLCM_Entropy: entropy };
}

function glrlmFeatures(raw, mask, width, height, nLevels = 12) {
  const bb = bbox(mask, width, height);
  const empty = { SRE: NaN, LRE: NaN, GLN: NaN };
  if (!bb) return empty;
  const roiVals = [];
  for (let y = bb.minY; y <= bb.maxY; y++) for (let x = bb.minX; x <= bb.maxX; x++)
    if (mask[y * width + x]) roiVals.push(raw[y * width + x]);
  if (roiVals.length === 0) return empty;
  const min = Math.min(...roiVals), max = Math.max(...roiVals);
  if (max === min) return empty;

  const q = new Int16Array(width * height).fill(0);
  for (let y = bb.minY; y <= bb.maxY; y++) for (let x = bb.minX; x <= bb.maxX; x++) {
    if (!mask[y * width + x]) continue;
    let lvl = Math.floor(((raw[y * width + x] - min) / (max - min)) * (nLevels - 1)) + 1;
    lvl = Math.max(1, Math.min(nLevels, lvl));
    q[y * width + x] = lvl;
  }

  const maxRun = Math.max(bb.maxX - bb.minX + 1, bb.maxY - bb.minY + 1);
  const P = Array.from({ length: nLevels + 1 }, () => new Float64Array(maxRun + 1));
  const dirs = [[0, 1], [1, 0], [1, 1], [1, -1]];

  for (const [dx, dy] of dirs) {
    const visited = new Uint8Array(width * height);
    for (let y = bb.minY; y <= bb.maxY; y++) for (let x = bb.minX; x <= bb.maxX; x++) {
      const idx = y * width + x;
      if (q[idx] === 0 || visited[idx]) continue;
      const level = q[idx];
      let runLen = 1; visited[idx] = 1;
      let px = x + dx, py = y + dy;
      while (px >= bb.minX && px <= bb.maxX && py >= bb.minY && py <= bb.maxY && q[py * width + px] === level) {
        visited[py * width + px] = 1; runLen++; px += dx; py += dy;
      }
      P[level][Math.min(runLen, maxRun)]++;
    }
  }
  let Nr = 0;
  for (let l = 1; l <= nLevels; l++) for (let r = 1; r <= maxRun; r++) Nr += P[l][r];
  if (Nr === 0) return empty;

  let SRE = 0, LRE = 0, GLN = 0;
  for (let l = 1; l <= nLevels; l++) {
    let rowSum = 0;
    for (let r = 1; r <= maxRun; r++) {
      SRE += P[l][r] / (r * r);
      LRE += P[l][r] * (r * r);
      rowSum += P[l][r];
    }
    GLN += rowSum * rowSum;
  }
  return { SRE: SRE / Nr, LRE: LRE / Nr, GLN: GLN / Nr };
}

function liverContextFeatures(raw, mask, width, height, tumorMean) {
  const n = width * height;
  let thresh = 0;
  { const sorted = Array.from(raw).sort((a, b) => a - b); thresh = sorted[Math.floor(n * 0.1)]; }
  const contextVals = [];
  for (let i = 0; i < n; i++) if (!mask[i] && raw[i] > thresh) contextVals.push(raw[i]);
  const liverEntropy = shannonEntropy(contextVals);
  const liverMean = contextVals.length ? contextVals.reduce((a, b) => a + b, 0) / contextVals.length : NaN;
  const tumorLiverContrast = Math.abs(tumorMean - liverMean) / (Math.abs(liverMean) + 1e-6);
  return { LiverEntropy: liverEntropy, TumorLiverContrast: tumorLiverContrast };
}

function extractAllFeatures(raw, mask, width, height) {
  const roiVals = [];
  for (let i = 0; i < width * height; i++) if (mask[i]) roiVals.push(raw[i]);
  const fo = firstOrderStats(roiVals);
  const shape = shapeStats(mask, width, height);
  const glcm = glcmFeatures(raw, mask, width, height);
  const rlm = glrlmFeatures(raw, mask, width, height);
  const liverCtx = liverContextFeatures(raw, mask, width, height, fo.Mean);
  return { ...shape, ...fo, ...glcm, ...rlm, ...liverCtx };
}

// ============================================================================
// REAL MODEL INFERENCE — calls the local FastAPI server (hcc_inference_api.py)
// that loads your actual trained .joblib models. A browser cannot execute
// those pickles directly, so this API is required; see the README note in
// the UI if the call fails (most likely cause: API not running).
// ============================================================================
const STAGING_MODEL_OPTIONS = ["XGBoost", "LightGBM", "RandomForest", "GradientBoosting"];

async function callPredictAPI(apiBase, path, body) {
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

async function runRealModel(apiBase, stagingModel, features) {
  const [stageRes, necroticRes] = await Promise.all([
    callPredictAPI(apiBase, "/predict/stage", { model: stagingModel, features }),
    callPredictAPI(apiBase, "/predict/necrotic", { features }),
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

function UploadStep({ title, subtitle, onFile, previewCanvasRef, loaded, onNext, canGoBack, onBack }) {
  const inputRef = useRef(null);
  return (
    <div className="rounded-lg p-8 border flex flex-col items-center text-center" style={{ background: COLORS.panel, borderColor: COLORS.hairline }}>
      <div className="w-12 h-12 rounded-full flex items-center justify-center mb-3" style={{ background: COLORS.tealSoft }}>
        <Upload size={22} color={COLORS.teal} />
      </div>
      <div className="text-lg font-semibold mb-1" style={{ color: COLORS.ink }}>{title}</div>
      <div className="text-sm mb-5" style={{ color: COLORS.inkSoft }}>{subtitle}</div>

      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
      />

      <canvas
        ref={previewCanvasRef}
        className="max-w-full rounded border mb-4"
        style={{ borderColor: COLORS.hairline, maxHeight: 320, display: loaded ? "block" : "none" }}
      />

      {!loaded ? (
        <button onClick={() => inputRef.current.click()} className="px-4 py-2 rounded-md text-sm font-medium text-white" style={{ background: COLORS.teal }}>
          Choose image file&hellip;
        </button>
      ) : (
        <div className="w-full flex flex-col items-center">
          <div className="flex gap-2">
            <button onClick={() => inputRef.current.click()} className="px-3 py-1.5 rounded-md text-xs border" style={{ borderColor: COLORS.hairline, color: COLORS.inkSoft }}>
              Replace file
            </button>
            <button onClick={onNext} className="px-4 py-1.5 rounded-md text-xs font-medium text-white flex items-center gap-1" style={{ background: COLORS.teal }}>
              Continue <ArrowRight size={13} />
            </button>
          </div>
        </div>
      )}

      <div className="text-[11px] mt-5 px-3 py-2 rounded flex gap-2 items-start text-left" style={{ background: "#FBFAF7", color: COLORS.inkSoft }}>
        <Info size={13} className="shrink-0 mt-0.5" />
        Accepts PNG/JPEG/BMP. DICOM files must be exported/converted to a
        standard image format first — this prototype cannot parse raw DICOM
        in-browser.
      </div>

      {canGoBack && (
        <button onClick={onBack} className="text-xs mt-4" style={{ color: COLORS.inkSoft }}>
          &larr; Back
        </button>
      )}
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

  const postCanvasRef = useRef(null);
  const preCanvasRef = useRef(null);
  const subCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);

  const [postData, setPostData] = useState(null);
  const [preData, setPreData] = useState(null);
  const [sub, setSub] = useState(null);
  const [polygon, setPolygon] = useState([]);
  const [drawing, setDrawing] = useState(false);
  const [features, setFeatures] = useState(null);

  useEffect(() => {
    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = "https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap";
    document.head.appendChild(link);
    return () => document.head.removeChild(link);
  }, []);

  const handlePostFile = async (file) => {
    const data = await loadImageToCanvas(file, postCanvasRef.current);
    setPostData(data);
  };
  const handlePreFile = async (file) => {
    const data = await loadImageToCanvas(file, preCanvasRef.current);
    setPreData(data);
  };

  useEffect(() => {
    if (step === 3 && postData && preData && !sub) {
      const w = Math.min(postData.width, preData.width);
      const h = Math.min(postData.height, preData.height);
      const result = subtractImages(
        { data: postData.data, width: w, height: h },
        { data: preData.data, width: w, height: h }
      );
      setSub(result);
    }
  }, [step, postData, preData, sub]);

  useEffect(() => {
    if (!sub || !subCanvasRef.current || !overlayCanvasRef.current) return;
    const c = subCanvasRef.current;
    c.width = sub.width; c.height = sub.height;
    c.getContext("2d").putImageData(sub.display, 0, 0);

    const o = overlayCanvasRef.current;
    o.width = sub.width; o.height = sub.height;
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
  }, [sub, polygon, drawing]);

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

  const clearROI = () => { setPolygon([]); setFeatures(null); };
  const undoROI = () => { setPolygon([]); };

  const finalizeROIAndExtract = () => {
    if (!sub || polygon.length < 3) return;
    const mask = rasterizeMask(polygon, sub.width, sub.height);
    const f = extractAllFeatures(sub.raw, mask, sub.width, sub.height);
    setFeatures(f);
    setStep(4);
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
          "Could not reach the inference API. Make sure hcc_inference_api.py is running " +
            "(uvicorn hcc_inference_api:app --port 8000) and the API base URL above is correct."
      );
    } finally {
      setPredictLoading(false);
    }
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
              <div className="text-[11px]" style={{ color: "#9AA7B8", fontFamily: "'IBM Plex Mono', monospace" }}>RESEARCH PROTOTYPE &middot; v0.2</div>
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
          <UploadStep
            title="Step 1 — Select post-contrast image"
            subtitle="Upload the post-contrast phase slice for this patient."
            onFile={handlePostFile}
            previewCanvasRef={postCanvasRef}
            loaded={!!postData}
            onNext={() => setStep(2)}
            canGoBack={false}
          />
        )}

        {step === 2 && (
          <UploadStep
            title="Step 2 — Select pre-contrast image"
            subtitle="Upload the matching pre-contrast phase slice (same patient, same slice position)."
            onFile={handlePreFile}
            previewCanvasRef={preCanvasRef}
            loaded={!!preData}
            onNext={() => setStep(3)}
            canGoBack={true}
            onBack={() => setStep(1)}
          />
        )}

        {step === 3 && (
          <div className="rounded-lg p-6 border" style={{ background: COLORS.panel, borderColor: COLORS.hairline }}>
            <SectionHeader eyebrow="Step 3" title="Subtracted image — draw tumor ROI" />
            {!sub ? (
              <div className="text-sm" style={{ color: COLORS.inkSoft }}>Computing subtraction (post &minus; pre)&hellip;</div>
            ) : (
              <>
                <div className="text-xs mb-3 px-3 py-2 rounded flex gap-2 items-start" style={{ background: COLORS.tealSoft, color: COLORS.teal }}>
                  <Info size={14} className="shrink-0 mt-0.5" />
                  Click and drag to draw a smooth freehand outline around the
                  tumor on the subtracted (enhancement) image below. Release
                  to close the outline.
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
                <div className="flex items-center justify-center gap-3 mt-4">
                  <button onClick={undoROI} className="px-3 py-1.5 rounded-md text-xs border flex items-center gap-1" style={{ borderColor: COLORS.hairline, color: COLORS.inkSoft }}>
                    <Undo2 size={13} /> Redraw
                  </button>
                  <button onClick={clearROI} className="px-3 py-1.5 rounded-md text-xs border flex items-center gap-1" style={{ borderColor: COLORS.hairline, color: COLORS.inkSoft }}>
                    <Eraser size={13} /> Clear
                  </button>
                  <button
                    onClick={finalizeROIAndExtract}
                    disabled={polygon.length < 3}
                    className="px-4 py-1.5 rounded-md text-xs font-medium text-white flex items-center gap-1 disabled:opacity-40"
                    style={{ background: COLORS.teal }}
                  >
                    Extract features from ROI <ArrowRight size={13} />
                  </button>
                </div>
                <div className="text-center mt-3">
                  <button onClick={() => setStep(2)} className="text-xs" style={{ color: COLORS.inkSoft }}>&larr; Back</button>
                </div>
              </>
            )}
          </div>
        )}

        {step === 4 && features && (
          <div className="rounded-lg p-6 border" style={{ background: COLORS.panel, borderColor: COLORS.hairline }}>
            <SectionHeader eyebrow="Step 4" title="Features extracted from your drawn ROI" />
            <div className="text-xs mb-4 px-3 py-2 rounded flex gap-2 items-start" style={{ background: COLORS.tealSoft, color: COLORS.teal }}>
              <Info size={14} className="shrink-0 mt-0.5" />
              These 25 values were computed directly from the pixels inside
              the ROI you drew on the subtracted image (genuine calculation —
              not a placeholder). <b className="mx-1">TOP</b> marks the 12
              SHAP-selected features. Liver-context features (*) use the rest
              of the image, excluding background, as a proxy for whole-liver
              context.
            </div>
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
            <div className="rounded-lg p-4 border text-xs leading-relaxed" style={{ background: "#FBFAF7", borderColor: COLORS.hairline, color: COLORS.inkSoft }}>
              <div className="font-semibold mb-1" style={{ color: COLORS.ink }}>Served by your real trained models</div>
              Staging prediction: <b>{stagingModel}</b> (your uploaded
              <code> {stagingModel === "XGBoost" ? "best_model_XGBoost.joblib" : `model_${stagingModel}.joblib`}</code>).
              Necrotic prediction: your uploaded
              <code> best_necrotic_vs_others_model_RandomForest.joblib</code>.
              Both ran via the local inference API — feature extraction,
              scaling, and classification above are all genuine, using the
              actual models from your study. Evidence values are real SHAP
              contributions computed server-side with <code>shap.TreeExplainer</code>.
            </div>
            <div className="flex justify-between">
              <button onClick={() => setStep(4)} className="text-xs" style={{ color: COLORS.inkSoft }}>&larr; Back to features</button>
              <button
                onClick={() => { setStep(1); setPostData(null); setPreData(null); setSub(null); setPolygon([]); setFeatures(null); setResult(null); setPredictError(null); }}
                className="flex items-center gap-1 text-xs px-3 py-1.5 rounded border"
                style={{ borderColor: COLORS.hairline, color: COLORS.inkSoft }}
              >
                <RotateCcw size={12} /> Start new case
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="w-full px-6 py-3 flex items-center gap-3 border-t" style={{ background: "#FBEAE8", borderColor: "#E8C4BE" }}>
        <AlertTriangle size={18} color={COLORS.advanced} className="shrink-0" />
        <div className="text-xs" style={{ color: "#7A2A21" }}>
          <b>Research prototype — not a diagnostic device.</b> Feature
          extraction and predictions now use your real trained models via
          the local inference API, but this pipeline has not undergone
          clinical/regulatory validation (external cohort testing, prospective
          evaluation, calibration review). All outputs must be confirmed by
          a qualified radiologist before any clinical use.
        </div>
        <FileWarning size={16} color={COLORS.advanced} className="ml-auto shrink-0 opacity-60" />
      </div>
    </div>
  );
}
