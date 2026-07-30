"""Streamlit app to orchestrate the pilot → merge → export pipeline.

Launch with:
    uv run streamlit run scripts/orchestrator.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
PARTIAL_DIR = REPO_ROOT / "docs" / "results" / "partial"
RESULTS_DIR = REPO_ROOT / "results"
WEB_DATA_DIR = REPO_ROOT / "web" / "public" / "data"

st.set_page_config(page_title="Pilot Orchestrator", layout="wide")
st.title("🚀 Pilot Orchestrator")


def _run_command(cmd: list[str], label: str) -> tuple[int, str]:
    """Run a subprocess and stream output to a Streamlit expander."""
    with st.expander(f"📋 {label} — output", expanded=True):
        placeholder = st.empty()
        start = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        lines: list[str] = []
        for line in proc.stdout:  # type: ignore[union-attr]
            lines.append(line)
            placeholder.code("".join(lines[-200:]), language="text")
        proc.wait()
        elapsed = time.time() - start
        placeholder.code("".join(lines[-200:]), language="text")
        if proc.returncode == 0:
            st.success(f"✅ Completed in {elapsed:.1f}s")
        else:
            st.error(f"❌ Failed (exit {proc.returncode}) after {elapsed:.1f}s")
        return proc.returncode, "".join(lines)


# ---------------------------------------------------------------------------
# Section 1: Run Pilot
# ---------------------------------------------------------------------------
st.header("1️⃣ Run Pilot")

col1, col2 = st.columns(2)
with col1:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from tsforecastbench.data.registry import discover_cases

    available_cases = sorted(discover_cases().keys())
    selected_cases = st.multiselect(
        "Cases (empty = all)", available_cases, default=[], key="pilot_cases"
    )
    origins = st.number_input("Rolling origins", min_value=1, max_value=200, value=40)
    fast_mode = st.checkbox("Fast mode (single hold-out)", value=False)

with col2:
    st.markdown("**Method families**")
    with_baseline = not st.checkbox("Skip baseline methods", value=False)
    with_dl = st.checkbox("Include DL (N-HiTS, PatchTST)", value=False)
    with_nf = st.checkbox("Include NF (TFT, iTransformer, TimesNet)", value=False)
    with_fm = st.checkbox("Include FM (Chronos, TimesFM, Moirai)", value=False)
    tag = st.text_input("Tag (partial run name, leave empty for full run)", value="")

if st.button("▶️ Run Pilot", type="primary"):
    cmd = [sys.executable, "scripts/run_pilot.py"]
    if selected_cases:
        cmd += ["--cases", *selected_cases]
    cmd += ["--origins", str(origins)]
    if fast_mode:
        cmd.append("--fast")
    if not with_baseline:
        cmd.append("--skip-baseline")
    if with_dl:
        cmd.append("--with-dl")
    if with_nf:
        cmd.append("--with-nf")
    if with_fm:
        cmd.append("--with-fm")
    if tag:
        cmd += ["--tag", tag]

    st.session_state["last_pilot_cmd"] = " ".join(cmd)
    _run_command(cmd, "run_pilot.py")

if "last_pilot_cmd" in st.session_state:
    st.caption(f"`{st.session_state['last_pilot_cmd']}`")

# ---------------------------------------------------------------------------
# Section 2: Merge Partial Results
# ---------------------------------------------------------------------------
st.header("2️⃣ Merge Partial Results")

partials = sorted(PARTIAL_DIR.glob("pilot_metrics*.parquet")) if PARTIAL_DIR.exists() else []
if partials:
    partial_labels = [p.name for p in partials]
    selected_partials = st.multiselect(
        "Select partial metric files to merge",
        partial_labels,
        default=partial_labels,
        key="merge_partials",
    )

    if st.button("🔀 Merge Selected", type="primary"):
        paths = [str(PARTIAL_DIR / name) for name in selected_partials]
        if len(paths) < 2:
            st.warning("Select at least 2 partial files to merge.")
        else:
            cmd = [sys.executable, "scripts/merge_pilot_results.py", *paths]
            _run_command(cmd, "merge_pilot_results.py")
else:
    st.info(f"No partial files found in `{PARTIAL_DIR.relative_to(REPO_ROOT)}/`")

# ---------------------------------------------------------------------------
# Section 3: Export Web Data
# ---------------------------------------------------------------------------
st.header("3️⃣ Export Web Data")

st.caption(f"Output directory: `{WEB_DATA_DIR.relative_to(REPO_ROOT)}`")

if st.button("📦 Export Web Data", type="primary"):
    cmd = [sys.executable, "scripts/export_web_data.py", "--out", str(WEB_DATA_DIR)]
    _run_command(cmd, "export_web_data.py")

# ---------------------------------------------------------------------------
# Section 4: Full Pipeline (run all steps sequentially)
# ---------------------------------------------------------------------------
st.header("4️⃣ Full Pipeline")
st.caption("Run pilot (full, no tag) → export web data. Use when a single machine covers all methods.")

if st.button("🏁 Run Full Pipeline", type="primary"):
    # Step 1: pilot (full run, no tag)
    cmd_pilot = [sys.executable, "scripts/run_pilot.py"]
    if selected_cases:
        cmd_pilot += ["--cases", *selected_cases]
    cmd_pilot += ["--origins", str(origins)]
    if fast_mode:
        cmd_pilot.append("--fast")
    if with_dl:
        cmd_pilot.append("--with-dl")
    if with_nf:
        cmd_pilot.append("--with-nf")
    if with_fm:
        cmd_pilot.append("--with-fm")

    rc, _ = _run_command(cmd_pilot, "run_pilot.py (full)")
    if rc != 0:
        st.error("Pipeline aborted — pilot failed.")
        st.stop()

    # Step 2: export
    cmd_export = [sys.executable, "scripts/export_web_data.py", "--out", str(WEB_DATA_DIR)]
    _run_command(cmd_export, "export_web_data.py")
    st.balloons()

# ---------------------------------------------------------------------------
# Sidebar: current state
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("📁 Current Artifacts")

    scores_path = REPO_ROOT / "docs" / "results" / "scores.parquet"
    if scores_path.exists():
        mtime = scores_path.stat().st_mtime
        st.write(f"**scores.parquet**: {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}")
    else:
        st.write("**scores.parquet**: not found")

    metrics_path = RESULTS_DIR / "pilot_metrics.parquet"
    if metrics_path.exists():
        mtime = metrics_path.stat().st_mtime
        st.write(
            f"**pilot_metrics.parquet**: {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}"
        )
    else:
        st.write("**pilot_metrics.parquet**: not found")

    st.divider()
    st.subheader("Partial Files")
    if partials:
        for p in partials:
            mtime = p.stat().st_mtime
            st.write(f"• `{p.name}` ({time.strftime('%m-%d %H:%M', time.localtime(mtime))})")
    else:
        st.write("None")
