"""Run each training/evaluation stage in a fresh Python process.

A fresh process guarantees that all host RAM and VRAM from the previous model is released
before the next one is loaded (the laptop has 16 GB RAM / 8 GB VRAM), and makes every stage
individually reproducible from the command line.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from common.data import REPO_ROOT, repo_path


LOG_FILE: Path | None = None  # when set, every streamed line is also appended here


def set_log_file(path: str | Path | None) -> None:
    global LOG_FILE
    LOG_FILE = repo_path(path) if path else None
    if LOG_FILE:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def _emit(line: str) -> None:
    print(line, end="", flush=True)
    if LOG_FILE is not None:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line)


def run_module(module: str, *args: str, skip_if_exists: str | Path | None = None) -> None:
    if skip_if_exists is not None and repo_path(skip_if_exists).exists():
        _emit(f"[skip] {module} {' '.join(args)}  (found {skip_if_exists})\n")
        return
    cmd = [sys.executable, "-u", "-m", module, *args]
    _emit("[run] python -m " + " ".join(cmd[3:]) + "\n")
    env = dict(os.environ)
    env.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    for line in proc.stdout:
        # Drop noisy library progress bars / warnings; keep our own log lines.
        if any(s in line for s in ("it/s]", "s/it]", "Loading checkpoint shards", "UserWarning", "warnings.warn")):
            continue
        _emit(line)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"{module} failed with exit code {rc}")
