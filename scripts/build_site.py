"""Build a static, browser-viewable EDA site — one page per case + an index.

Each page is a `marimo export html` of notebooks/eda.py with EDA_CASE pinned, so the
plots are interactive (plotly zoom/pan/hover) without needing a running kernel. The
result is a plain folder of HTML you can open locally or publish to GitHub Pages.

    uv sync --extra notebooks
    uv run python scripts/build_site.py            # writes ./site, open site/index.html
    uv run python scripts/build_site.py --out _site

The GitHub Pages workflow (.github/workflows/pages.yml) runs exactly this and deploys
the output.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EDA_NB = REPO_ROOT / "notebooks" / "eda.py"


def _rel(path: Path, base: Path) -> str:
    """relative_to for display, falling back to the absolute path when not a subpath."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def _cases() -> list[str]:
    from tsforecastbench.data.registry import discover_cases

    return list(discover_cases())


def _export_case(case_id: str, out_html: Path) -> None:
    env = {**os.environ, "EDA_CASE": case_id}
    out_html.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "marimo", "export", "html", str(EDA_NB), "-o", str(out_html)],
        check=True,
        env=env,
        cwd=REPO_ROOT,
    )


def _index_html(case_ids: list[str]) -> str:
    items = "\n".join(
        f'      <li><a href="eda/{cid}.html">{cid}</a></li>' for cid in case_ids
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>tsforecastbench — dataset EDA</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 3rem auto;
            padding: 0 1rem; line-height: 1.5; color: #222; }}
    h1 {{ margin-bottom: 0.2rem; }}
    p.sub {{ color: #666; margin-top: 0; }}
    ul {{ list-style: none; padding: 0; }}
    li {{ margin: 0.4rem 0; }}
    a {{ text-decoration: none; color: #1a6; font-weight: 600; }}
    a:hover {{ text-decoration: underline; }}
    code {{ background: #f3f3f3; padding: 0.1rem 0.3rem; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>tsforecastbench — dataset EDA</h1>
  <p class="sub">One interactive page per case: map, splits, completeness, distribution,
     autocorrelation, and exogenous cross-correlation / mutual information.</p>
  <p>Generated from <code>notebooks/eda.py</code>. For the live, reactive version run
     <code>uv run marimo run notebooks/eda.py</code>.</p>
  <h2>Cases</h2>
  <ul>
{items}
  </ul>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="site", help="output directory (default: site)")
    p.add_argument("--cases", nargs="*", default=None, help="case ids (default: all)")
    args = p.parse_args(argv)

    out_dir = (REPO_ROOT / args.out).resolve()
    case_ids = args.cases or _cases()

    print(f"[site] building {len(case_ids)} EDA page(s) into {_rel(out_dir, REPO_ROOT)}/")
    for cid in case_ids:
        target = out_dir / "eda" / f"{cid}.html"
        print(f"[site]   {cid} -> {_rel(target, out_dir)}")
        _export_case(cid, target)

    index = out_dir / "index.html"
    index.write_text(_index_html(case_ids), encoding="utf-8")
    print(f"[site] wrote {_rel(index, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
