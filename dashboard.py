#!/usr/bin/env python3
"""
Training dashboard for PAN 2026 style change detection — built with Plotly.

Reads JSONL log files from logs/history/ and produces an interactive HTML
dashboard with six panels:

  Row 1: DeBERTa step loss            | Validation F1  (all models)
  Row 2: DeBERTa epoch train+val loss | LightGBM log-loss per round
  Row 3: SSPC train loss + val F1     | DeBERTa learning-rate schedule

Usage:
    python dashboard.py                      # generate logs/charts/dashboard.html
    python dashboard.py --live               # serve on http://localhost:8050 with auto-refresh
    python dashboard.py --live --port 8051   # custom port
    python dashboard.py --interval 8         # refresh every 8 s (--live only)
    python dashboard.py --log-dir logs       # custom log directory
    python dashboard.py --output out.html    # custom output path

For richer per-step monitoring also run:
    tensorboard --logdir logs/tensorboard/
"""
import argparse
import json
import sys
import time
import threading
from pathlib import Path
from typing import Dict, List, Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
except ImportError:
    print("ERROR: plotly not installed.  Run: pip install plotly")
    sys.exit(1)


# ─── Constants ────────────────────────────────────────────────────────────────

DIFFICULTIES = ["easy", "medium", "hard"]
COLORS = {
    "easy":   "#2196F3",   # blue
    "medium": "#FF9800",   # orange
    "hard":   "#4CAF50",   # green
}
# Lighter versions for secondary series (val loss, etc.)
COLORS_LIGHT = {
    "easy":   "#90CAF9",
    "medium": "#FFCC80",
    "hard":   "#A5D6A7",
}
MODEL_DASH = {
    "deberta": "solid",
    "sspc":    "dash",
}

SUBPLOT_TITLES = (
    "DeBERTa — Training Loss (per optimizer step)",
    "Validation F1-macro (all models)",
    "DeBERTa — Epoch Losses (train & val)",
    "LightGBM — Log-loss per Boosting Round",
    "SSPC — Epoch Loss (solid) & Val F1 (dashed)",
    "DeBERTa — Learning-rate Schedule",
)


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_run(path: Path) -> List[dict]:
    events = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except FileNotFoundError:
        pass
    return events


def load_runs(log_dir: Path) -> Dict[str, List[dict]]:
    history = log_dir / "history"
    if not history.exists():
        return {}
    return {p.stem: _load_run(p) for p in sorted(history.glob("*.jsonl"))}


def _filter(events: List[dict], etype: str) -> List[dict]:
    return [e for e in events if e.get("type") == etype]


# ─── Trace builders ───────────────────────────────────────────────────────────

def _line(x, y, name: str, color: str, dash: str = "solid",
          opacity: float = 1.0, mode: str = "lines+markers",
          marker_size: int = 4, **kwargs) -> go.Scatter:
    return go.Scatter(
        x=x, y=y, name=name, mode=mode,
        line=dict(color=color, dash=dash, width=2),
        marker=dict(size=marker_size, color=color),
        opacity=opacity,
        connectgaps=True,
        **kwargs,
    )


# ─── Six panels ───────────────────────────────────────────────────────────────

def _panel_deberta_step(runs: dict) -> List[go.BaseTraceType]:
    traces = []
    for diff in DIFFICULTIES:
        evts = _filter(runs.get(f"deberta_{diff}", []), "step")
        if not evts:
            continue
        x = [e["global_step"] for e in evts]
        y = [e["loss"]        for e in evts]
        traces.append(_line(x, y, name=f"{diff}", color=COLORS[diff]))
    return traces


def _panel_val_f1(runs: dict) -> List[go.BaseTraceType]:
    traces = []
    for model, dash in MODEL_DASH.items():
        for diff in DIFFICULTIES:
            evts = _filter(runs.get(f"{model}_{diff}", []), "epoch")
            if not evts:
                continue
            x = [e["epoch"]  for e in evts]
            y = [e["val_f1"] for e in evts]
            traces.append(_line(
                x, y,
                name=f"{model}/{diff}",
                color=COLORS[diff],
                dash=dash,
                marker_size=5,
            ))
    return traces


def _panel_deberta_epoch(runs: dict) -> List[go.BaseTraceType]:
    traces = []
    for diff in DIFFICULTIES:
        evts = _filter(runs.get(f"deberta_{diff}", []), "epoch")
        if not evts:
            continue
        x = [e["epoch"]            for e in evts]
        tl = [e["train_loss"]      for e in evts]
        vl = [e.get("val_loss")    for e in evts]
        traces.append(_line(x, tl, name=f"{diff} train",
                            color=COLORS[diff], dash="solid"))
        if any(v is not None for v in vl):
            traces.append(_line(x, vl, name=f"{diff} val",
                                color=COLORS_LIGHT[diff], dash="dot"))
    return traces


def _panel_lgbm(runs: dict) -> List[go.BaseTraceType]:
    traces = []
    for diff in DIFFICULTIES:
        evts = _filter(runs.get(f"lgbm_{diff}", []), "lgbm_round")
        if not evts:
            continue
        x  = [e["round"]                   for e in evts]
        tr = [e.get("train_logloss")       for e in evts]
        vl = [e.get("val_logloss")         for e in evts]
        if any(t is not None for t in tr):
            traces.append(_line(x, tr, name=f"{diff} train",
                                color=COLORS[diff], dash="solid",
                                mode="lines", marker_size=3))
        if any(v is not None for v in vl):
            traces.append(_line(x, vl, name=f"{diff} val",
                                color=COLORS_LIGHT[diff], dash="dash",
                                mode="lines", marker_size=3))
    return traces


def _panel_sspc(runs: dict) -> List[go.BaseTraceType]:
    traces = []
    for diff in DIFFICULTIES:
        evts = _filter(runs.get(f"sspc_{diff}", []), "epoch")
        if not evts:
            continue
        x   = [e["epoch"]      for e in evts]
        tl  = [e["train_loss"] for e in evts]
        f1s = [e["val_f1"]     for e in evts]
        traces.append(_line(x, tl,  name=f"{diff} loss",
                            color=COLORS[diff],       dash="solid"))
        traces.append(_line(x, f1s, name=f"{diff} F1",
                            color=COLORS_LIGHT[diff], dash="dash"))
    return traces


def _panel_lr(runs: dict) -> List[go.BaseTraceType]:
    traces = []
    for diff in DIFFICULTIES:
        evts = [e for e in _filter(runs.get(f"deberta_{diff}", []), "step")
                if "lr" in e]
        if not evts:
            continue
        x = [e["global_step"] for e in evts]
        y = [e["lr"]          for e in evts]
        traces.append(_line(x, y, name=diff, color=COLORS[diff],
                            mode="lines", marker_size=3))
    return traces


# ─── Best-metrics summary ─────────────────────────────────────────────────────

def _best_metrics(runs: dict) -> str:
    parts = []
    for diff in DIFFICULTIES:
        row = []
        for model in ("deberta", "lgbm", "sspc"):
            evts = runs.get(f"{model}_{diff}", [])
            final = next((e for e in reversed(evts) if e.get("type") == "final"), None)
            if final and "best_f1" in final:
                row.append(f"{model}={final['best_f1']:.4f}")
        if row:
            parts.append(f"<b>{diff}</b>: " + "  ".join(row))
    return "  |  ".join(parts) if parts else "Training not started yet — logs will appear here."


# ─── Build full figure ────────────────────────────────────────────────────────

def build_figure(log_dir: Path) -> go.Figure:
    runs = load_runs(log_dir)

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=SUBPLOT_TITLES,
        vertical_spacing=0.10,
        horizontal_spacing=0.08,
    )

    panels = [
        (1, 1, _panel_deberta_step(runs)),
        (1, 2, _panel_val_f1(runs)),
        (2, 1, _panel_deberta_epoch(runs)),
        (2, 2, _panel_lgbm(runs)),
        (3, 1, _panel_sspc(runs)),
        (3, 2, _panel_lr(runs)),
    ]

    # Track which legend entries have been shown to avoid duplicates
    legend_seen = set()
    for row, col, traces in panels:
        for tr in traces:
            if tr.name in legend_seen:
                tr.showlegend = False
            else:
                legend_seen.add(tr.name)
            fig.add_trace(tr, row=row, col=col)

    # Axis labels
    axis_labels = {
        "xaxis":  "Optimizer step",  "yaxis":  "Loss",
        "xaxis2": "Epoch",            "yaxis2": "F1-macro",
        "xaxis3": "Epoch",            "yaxis3": "Loss",
        "xaxis4": "Boosting round",   "yaxis4": "Log-loss",
        "xaxis5": "Epoch",            "yaxis5": "Value",
        "xaxis6": "Optimizer step",   "yaxis6": "LR",
    }
    for axis, label in axis_labels.items():
        fig.update_layout(**{axis: dict(title_text=label, title_font_size=11)})

    # Scientific notation for LR axis
    fig.update_layout(yaxis6=dict(
        title_text="LR", title_font_size=11, exponentformat="e"
    ))

    summary = _best_metrics(runs)
    fig.update_layout(
        height=950,
        title=dict(
            text=(
                "<b>PAN 2026 — Training Dashboard</b><br>"
                f"<span style='font-size:11px;color:#666'>{summary}</span>"
            ),
            x=0.5, xanchor="center", font_size=16,
        ),
        legend=dict(
            orientation="v",
            x=1.02, y=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ccc",
            borderwidth=1,
            font_size=10,
        ),
        margin=dict(l=60, r=200, t=100, b=50),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        hovermode="x unified",
    )

    # Subtle grid on all axes
    for i in range(1, 7):
        fig.update_layout(**{
            f"xaxis{'' if i == 1 else i}": dict(
                showgrid=True, gridcolor="#E0E0E0", gridwidth=1
            ),
            f"yaxis{'' if i == 1 else i}": dict(
                showgrid=True, gridcolor="#E0E0E0", gridwidth=1
            ),
        })

    return fig


# ─── HTML with auto-reload ────────────────────────────────────────────────────

def render_html(fig: go.Figure, interval_s: int = 0) -> str:
    """
    Render Plotly figure to a full HTML string.
    If interval_s > 0, inject a JavaScript auto-reload snippet.
    """
    html = pio.to_html(
        fig,
        full_html=True,
        include_plotlyjs="cdn",
        config={"scrollZoom": True, "displayModeBar": True},
    )
    if interval_s > 0:
        snippet = (
            f"<script>"
            f"setTimeout(function(){{location.reload();}},{interval_s * 1000});"
            f"</script>"
        )
        html = html.replace("</body>", snippet + "\n</body>")
    return html


# ─── Live server ──────────────────────────────────────────────────────────────

def _serve_live(log_dir: Path, out_path: Path, interval: int, port: int):
    """
    Regenerate the dashboard HTML every `interval` seconds and serve it via
    a minimal HTTP server.  The page auto-reloads in the browser.
    """
    import http.server
    import socketserver
    import os

    serve_dir = out_path.parent

    def _regen():
        while True:
            try:
                fig  = build_figure(log_dir)
                html = render_html(fig, interval_s=interval)
                out_path.write_text(html, encoding="utf-8")
            except Exception as exc:
                print(f"[dashboard] regeneration error: {exc}")
            time.sleep(interval)

    regen_thread = threading.Thread(target=_regen, daemon=True)
    regen_thread.start()

    # Initial render before server starts
    try:
        fig  = build_figure(log_dir)
        html = render_html(fig, interval_s=interval)
        out_path.write_text(html, encoding="utf-8")
    except Exception:
        pass

    class _Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(serve_dir), **kw)

        def log_message(self, fmt, *args):
            pass   # suppress per-request noise

    url = f"http://localhost:{port}/{out_path.name}"
    print(f"\nDashboard served at: {url}")
    print(f"Auto-refreshes every {interval}s.  Press Ctrl+C to stop.")
    print(f"TensorBoard (richer): tensorboard --logdir {log_dir / 'tensorboard'}\n")

    with socketserver.TCPServer(("", port), _Handler) as httpd:
        httpd.serve_forever()


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Plotly training dashboard for PAN 2026 style change detection"
    )
    parser.add_argument(
        "--log-dir", type=Path,
        default=Path(__file__).parent / "logs",
        help="Root log directory (default: logs/)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output HTML file (default: logs/charts/dashboard.html)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Start an HTTP server and auto-refresh the dashboard while training",
    )
    parser.add_argument(
        "--interval", type=int, default=5,
        help="Auto-refresh interval in seconds for --live mode (default: 5)",
    )
    parser.add_argument(
        "--port", type=int, default=8050,
        help="HTTP port for --live mode (default: 8050)",
    )
    args = parser.parse_args()

    log_dir  = args.log_dir
    out_path = args.output or (log_dir / "charts" / "dashboard.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.live:
        _serve_live(log_dir, out_path, args.interval, args.port)
    else:
        fig  = build_figure(log_dir)
        html = render_html(fig)
        out_path.write_text(html, encoding="utf-8")
        print(f"Dashboard saved to: {out_path}")
        print(f"Open in browser: file:///{out_path.as_posix()}")
        print(f"TensorBoard:     tensorboard --logdir {log_dir / 'tensorboard'}")


if __name__ == "__main__":
    main()