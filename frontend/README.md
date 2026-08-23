# Frontend — React + Vite

The browser-side app: image upload, pixel subtraction, freehand ROI drawing, radiomic feature extraction, and the results UI. This is the original research prototype, unmodified except for two small changes (see below).

## Run it

See [SETUP.md](SETUP.md). Once dependencies are installed: `npm run dev`, then open the printed `http://localhost:5173`.

## Layout

```
frontend/
├── index.html
├── vite.config.js
├── tailwind.config.js
├── .env.example              VITE_API_BASE_URL default
├── src/
│   ├── main.jsx               React entry point
│   ├── index.css               Tailwind directives
│   └── App.jsx                 Everything else — see below
```

Almost the entire app lives in one file, `App.jsx`, organized top-to-bottom into clearly commented sections:

1. **Design tokens** (`COLORS`, `STAGES`, `STEPS`)
2. **Feature schema** (`FEATURE_META`) — the 25 radiomic features, grouped and labeled
3. **Image helpers** — loading a file to canvas, grayscale conversion, pixel subtraction
4. **Feature extraction** — `shannonEntropy`, `firstOrderStats`, `shapeStats`, `glcmFeatures`, `glrlmFeatures`, `liverContextFeatures`, `extractAllFeatures`
5. **Real model inference** — `callPredictAPI`, `runRealModel` (calls the backend)
6. **UI components** — `Stepper`, `UploadStep`, `StageGauge`, `NecroticReadout`, `EvidenceStrip`
7. **Main `App` component** — wires the 5-step wizard together

## What was changed from the original prototype

Only one line, to satisfy the "environment-configurable API URL" requirement from the original spec, without touching the feature-extraction or UI logic:

```diff
- const [apiBase, setApiBase] = useState("http://localhost:8000");
+ const [apiBase, setApiBase] = useState(import.meta.env.VITE_API_BASE_URL || "http://localhost:8000");
```

The UI's own "API base URL" field (visible in Step 4) still lets you override this at runtime without editing `.env` — that was already built into the prototype.

**Do not modify the feature-extraction functions without re-running the parity test** — see [../docs/PARITY_TESTING.md](../docs/PARITY_TESTING.md). They must stay numerically identical to the Python feature extraction used at training time.

## Where it talks to the backend

`callPredictAPI` and `runRealModel`, both defined just above the UI components section in `App.jsx`. Two requests fire in parallel on "Run prediction": `POST /predict/stage` and `POST /predict/necrotic`. See [../backend/API.md](../backend/API.md) for the exact contract.
