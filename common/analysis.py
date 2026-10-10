"""Small helpers used by the analysis notebooks (tables, figures, text excerpts)."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from common.data import REPO_ROOT, read_jsonl, repo_path

FIG_DIR = REPO_ROOT / "report" / "figures"

# Colour-blind-safe palette (Okabe-Ito) used consistently across all figures.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.precision", 4)


def savefig(fig, name: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight")
    print("saved figure:", (FIG_DIR / f"{name}.pdf").relative_to(REPO_ROOT))


def jl(path) -> pd.DataFrame:
    return pd.DataFrame(read_jsonl(path))


def js(path):
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def excerpt(text: str, n: int = 350, width: int = 110) -> str:
    t = " ".join(str(text).split())
    if len(t) > n:
        t = t[: n // 2] + " [...] " + t[-n // 2 :]
    return textwrap.fill(t, width=width)


def show_log_tail(path, n: int = 25):
    p = repo_path(path)
    if not p.exists():
        print("(no log yet)", p)
        return
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(lines[-n:]))


def grep_log(path, pattern: str, limit: int = 40):
    p = repo_path(path)
    if not p.exists():
        print("(no log yet)", p)
        return
    lines = [l for l in p.read_text(encoding="utf-8", errors="replace").splitlines() if pattern in l]
    print("\n".join(lines[:limit]))
    if len(lines) > limit:
        print(f"... ({len(lines) - limit} more lines)")
