# Frontend Setup

## 1. Install dependencies

```bash
cd frontend
npm install
```

## 2. (Optional) Configure the API URL

```bash
cp .env.example .env
```

Default (`http://localhost:8000`) matches the backend's default — no edit needed unless you changed `API_PORT` in the backend's `.env`.

## 3. Run

```bash
npm run dev
```

Vite prints the local URL (default `http://localhost:5173`). Open it in a browser.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Blank page, console shows import errors | `npm install` didn't finish or failed | Delete `node_modules`, re-run `npm install` |
| "API unreachable" when clicking "Test connection" | Backend not running, wrong port, or CORS mismatch | Confirm backend is up at the URL shown; check backend's `CORS_ORIGINS` includes `http://localhost:5173` |
| ROI drawing feels laggy on a large image | Expected — feature extraction (GLCM/GLRLM) is O(pixels × directions) in JS, unoptimized | Not a bug; use smaller images for quick testing |
