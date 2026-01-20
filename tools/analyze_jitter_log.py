#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_json(path: Path) -> Any:
    with path.open("r") as f:
        return json.load(f)


def _as_array(steps: list[dict[str, Any]], key: str) -> np.ndarray | None:
    vals = [s.get(key) for s in steps if isinstance(s, dict) and key in s]
    if not vals:
        return None
    try:
        arr = np.asarray(vals, dtype=np.float32)
    except Exception:
        return None
    if arr.ndim == 0:
        return None
    return arr


def _dominant_freqs(x: np.ndarray, dt: float, top_k: int = 8) -> list[dict[str, float]]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size < 8:
        return []
    x = x - np.mean(x)
    n = x.size
    window = np.hanning(n).astype(np.float32)
    y = np.fft.rfft(x * window)
    mag = np.abs(y).astype(np.float32)
    mag[0] = 0.0
    freqs = np.fft.rfftfreq(n, d=dt).astype(np.float32)
    if mag.size <= 1:
        return []
    idx = np.argsort(mag)[-top_k:][::-1]
    out = []
    for i in idx:
        out.append({"hz": float(freqs[i]), "mag": float(mag[i])})
    return out


def _summ_stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to play.py --record_log json")
    ap.add_argument("--meta", default="", help="Optional path to *_meta.json")
    ap.add_argument("--out", default="", help="Output directory (default: /tmp/twist2_jitter_analysis/<log_stem>)")
    ap.add_argument("--dt", type=float, default=0.0, help="Fallback dt if meta missing")
    args = ap.parse_args()

    log_path = Path(args.log).expanduser().resolve()
    if not log_path.exists():
        raise SystemExit(f"Log not found: {log_path}")

    meta_path = Path(args.meta).expanduser().resolve() if args.meta else log_path.with_name(log_path.stem + "_meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = _load_json(meta_path)
        except Exception:
            meta = {}

    dt = float(meta.get("dt", 0.0) or 0.0)
    if dt <= 0:
        dt = float(args.dt or 0.0)
    if dt <= 0:
        dt = 1.0 / 50.0

    raw = _load_json(log_path)
    if isinstance(raw, dict) and "steps" in raw:
        steps = raw["steps"]
    else:
        steps = raw
    if not isinstance(steps, list) or not steps:
        raise SystemExit(f"Unexpected log format in {log_path}")

    actions = _as_array(steps, "action")
    torques = _as_array(steps, "torque")
    dof_vel = _as_array(steps, "dof vel")
    dof_pos = _as_array(steps, "dof pos")

    action_rate_l2 = _as_array(steps, "action_rate_l2")
    torque_l2 = _as_array(steps, "torque_l2")
    dof_acc_l2 = _as_array(steps, "dof_acc_l2")
    base_acc_l2 = _as_array(steps, "base_acc_l2")

    def l2(x: np.ndarray | None) -> np.ndarray | None:
        if x is None:
            return None
        if x.ndim == 1:
            return x
        return np.linalg.norm(x, axis=-1).astype(np.float32)

    summary: dict[str, Any] = {
        "log": str(log_path),
        "meta": meta,
        "dt": float(dt),
        "n_steps": int(len(steps)),
    }

    for name, arr in [
        ("action", actions),
        ("torque", torques),
        ("dof_pos", dof_pos),
        ("dof_vel", dof_vel),
        ("action_rate_l2", action_rate_l2),
        ("torque_l2", torque_l2),
        ("dof_acc_l2", dof_acc_l2),
        ("base_acc_l2", base_acc_l2),
    ]:
        if arr is None:
            continue
        summary[name] = {"shape": list(arr.shape), "stats": _summ_stats(l2(arr))}

    if actions is not None and actions.shape[0] >= 3:
        da = np.diff(actions, axis=0)
        da_l2 = np.linalg.norm(da, axis=-1).astype(np.float32)
        summary["delta_action_l2"] = {"stats": _summ_stats(da_l2), "dominant_freqs_hz": _dominant_freqs(da_l2, dt)}

    if dof_vel is not None:
        dof_vel_l2 = l2(dof_vel)
        summary["dof_vel_l2"] = {"stats": _summ_stats(dof_vel_l2), "dominant_freqs_hz": _dominant_freqs(dof_vel_l2, dt)}

    if torques is not None:
        tq_l2 = l2(torques)
        summary["torque_l2_series"] = {"stats": _summ_stats(tq_l2), "dominant_freqs_hz": _dominant_freqs(tq_l2, dt)}

    out_dir = Path(args.out).expanduser() if args.out else Path("/tmp") / "twist2_jitter_analysis" / log_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "summary.json"
    out_txt = out_dir / "summary.txt"
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    lines = []
    lines.append(f"log: {log_path}")
    if meta_path.exists():
        lines.append(f"meta: {meta_path}")
    lines.append(f"dt: {dt:.6f} s")
    lines.append(f"n_steps: {len(steps)}")
    for k in ["action", "torque", "dof_pos", "dof_vel", "action_rate_l2", "dof_acc_l2", "base_acc_l2"]:
        if k in summary:
            st = summary[k]["stats"]
            lines.append(f"{k}: mean={st['mean']:.4f} std={st['std']:.4f} min={st['min']:.4f} max={st['max']:.4f}")
    if "dof_vel_l2" in summary:
        freqs = summary["dof_vel_l2"].get("dominant_freqs_hz", [])[:5]
        if freqs:
            lines.append("dof_vel_l2 dominant freqs (hz): " + ", ".join(f"{f['hz']:.2f}" for f in freqs))
    if "delta_action_l2" in summary:
        freqs = summary["delta_action_l2"].get("dominant_freqs_hz", [])[:5]
        if freqs:
            lines.append("delta_action_l2 dominant freqs (hz): " + ", ".join(f"{f['hz']:.2f}" for f in freqs))

    out_txt.write_text("\n".join(lines) + "\n")
    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_txt}")


if __name__ == "__main__":
    main()

