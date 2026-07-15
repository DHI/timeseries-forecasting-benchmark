# Results-matrix web app

React + TypeScript + Vite port of `app/app.py` (the Dash results-matrix viewer), built as static
files so it can be hosted on GitHub Pages. `app/app.py` still exists and works as a live Python app;
this is an additive, separately-deployed alternative — see `research_plan.md` / the repo root
`README.md` for project context.

## Development

```
npm install
npm run dev
```

## Data

Static JSON under `public/data/` is what the app reads at runtime — there is no backend. It's
generated from the committed/regenerable benchmark results by a Python export script (see
`scripts/export_web_data.py` at the repo root, added in a later milestone) rather than hand-edited.
`public/data/fixture.json` is a small hand-written placeholder used only during initial
scaffolding, before the export script exists.

## Checks

```
npm run lint          # oxlint
npm run format:check  # prettier --check
npm run test           # vitest
npm run build          # tsc -b && vite build
```

## Deployment

Built via GitHub Actions into GitHub Pages, at the repo's root Pages URL (the existing marimo EDA
site moves to `/eda/`). `vite.config.ts`'s `base` is set to the repo's Pages path — see that file's
comment before changing it.
