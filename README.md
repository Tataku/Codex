# Codex

## CLI (SSOT engine)

Core simulation/backtest logic lives in `mc_core.py`, and both the CLI and Streamlit UI call this module.

### Simulate

```bash
python monte_carlo_divergence.py simulate \
  --as-of-date 2024-12-31 \
  --paths 5000 \
  --horizon-days 365 \
  --ma-window 200 \
  --ma-regime bull \
  --ma-filter-mode both \
  --summary-out mc_summary.csv
```

### Backtest

```bash
python monte_carlo_divergence.py backtest \
  --start-date 2023-01-01 \
  --end-date 2023-06-30 \
  --horizon-days 30 \
  --step-days 30 \
  --train-lookback-days 730 \
  --paths 400 \
  --ma-window 200 \
  --ma-regime bull \
  --ma-filter-mode both \
  --min-median-return -0.02 \
  --min-p10-return -0.30 \
  --non-overlap true \
  --include-baselines true \
  --permutation-runs 200 \
  --out backtest_no_overlap.csv \
  --summary-json backtest_no_overlap.json
```

## Streamlit Simulator UI

Run locally:

```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

UI features:
- Simulate/Backtest mode selector
- Full parameter controls (paths/horizon/divergence/MA/decision/backtest window)
- Progressive updates via batch execution, progress bars, live charts, and diagnostics
- Download buttons for summary/path/backtest CSV + backtest JSON

## Deployment

### Streamlit Community Cloud (recommended)
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create app using `streamlit_app/app.py`.
3. Set Python dependencies from `requirements.txt`.

### Vercel integration (redirect only)
Vercel does not host Streamlit runtime directly. This repo serves a root `index.html` redirect page.

1. Deploy from repository root.
2. Keep `vercel.json` at repository root so rewrites are applied.
3. Redirect target is set to your provided URL: `https://dgrkeqbdszcqxhzy92u93v.streamlit.app/`.
4. Re-deploy.

### Vercel 404 troubleshooting
If Vercel shows `404: NOT_FOUND` after deploy, it usually means the project root has no default route configured.
This repo includes `vercel.json` to rewrite all routes to root `index.html`.

- Ensure Vercel project root is the repo root (not a subfolder).
- Ensure `vercel.json` is included in the deployed commit.
- Re-deploy after merge conflict resolution.
- Keep the Streamlit URL in `index.html` updated if the deployment URL changes.
