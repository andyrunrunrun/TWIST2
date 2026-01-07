#!/usr/bin/env python3
"""
Extract AMASS motions that exist in a PHC joblib/pickle dataset but are missing from TWIST2_dataset,
and save them as TWIST2-style *_stageii.pkl motion files.

This script targets the 3 missing sources previously identified:
  - BioMotionLab_NTroje  -> renamed to BMLrub_*
  - BMLhandball
  - TCD_handMocap

It writes TWIST2 stageii dicts with keys:
  fps, root_pos, root_rot, dof_pos, local_body_pos, link_body_list

local_body_pos is computed via MuJoCo FK from dof_pos only (base at origin, identity quat),
using a slightly patched g1_mocap_29dof.xml (toe Z offset aligned to TWIST2 stageii).
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    joblib = None

try:
    import mujoco  # type: ignore
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "MuJoCo python package is required to compute local_body_pos. "
        "Install mujoco in the current env."
    ) from e


_PHC_KEY_RE = re.compile(r"^\d+-(.+)_poses$")


@dataclass(frozen=True)
class StageII:
    fps: float
    root_pos: np.ndarray  # (T,3) float64
    root_rot_xyzw: np.ndarray  # (T,4) float64
    dof_pos: np.ndarray  # (T,29) float64
    local_body_pos: np.ndarray  # (T,B,3) float32
    link_body_list: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "fps": float(self.fps),
            "root_pos": self.root_pos,
            "root_rot": self.root_rot_xyzw,
            "dof_pos": self.dof_pos,
            "local_body_pos": self.local_body_pos,
            "link_body_list": list(self.link_body_list),
        }


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_phc_dataset(path: str) -> Dict[str, Any]:
    path = os.path.abspath(os.path.expanduser(path))
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if joblib is None:
        raise RuntimeError("joblib is not available; cannot reliably load PHC joblib-style dataset")
    # NOTE: For this PHC file, mmap_mode="r" is extremely slow; mmap_mode=None loads much faster.
    # This will materialize arrays in memory (expect multi-GB RAM usage).
    return joblib.load(path, mmap_mode=None)


def _get_twist2_stageii_ids(twist2_amass_dir: str) -> set[str]:
    ids: set[str] = set()
    for name in os.listdir(twist2_amass_dir):
        if not name.endswith("_stageii.pkl"):
            continue
        ids.add(name[: -len("_stageii.pkl")])
    return ids


def _load_link_body_list(reference_stageii_pkl: str) -> List[str]:
    d = _load_pickle(reference_stageii_pkl)
    if not isinstance(d, dict) or "link_body_list" not in d:
        raise TypeError(f"Unexpected stageii pkl structure: {reference_stageii_pkl}")
    link_body_list = d["link_body_list"]
    if not isinstance(link_body_list, list) or not all(isinstance(x, str) for x in link_body_list):
        raise TypeError(f"Invalid link_body_list in {reference_stageii_pkl}")
    return link_body_list


def _maybe_quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    if q.ndim != 2 or q.shape[1] < 4:
        raise ValueError(f"Expected quaternion array (T,>=4), got {q.shape}")
    q4 = q[:, :4].astype(np.float64, copy=False)
    mean_abs = np.mean(np.abs(q4), axis=0)
    # Heuristic: w component typically has the largest magnitude.
    # If it looks like w is stored first, convert to xyzw.
    if mean_abs[0] > mean_abs[3] * 1.2:
        return np.ascontiguousarray(np.stack([q4[:, 1], q4[:, 2], q4[:, 3], q4[:, 0]], axis=1))
    return np.ascontiguousarray(q4)


def _extract_phc_arrays(entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    dof = entry.get("dof", None)
    if not isinstance(dof, np.ndarray) or dof.ndim != 2 or int(dof.shape[1]) < 29:
        raise KeyError("Missing/invalid 'dof' array")
    dof29 = np.asarray(dof[:, :29], dtype=np.float64)

    trans = entry.get("root_trans_offset", entry.get("root_trans", entry.get("root_pos", None)))
    if not isinstance(trans, np.ndarray) or trans.ndim != 2 or int(trans.shape[1]) < 3:
        raise KeyError("Missing/invalid root translation array (root_trans_offset/root_trans/root_pos)")
    root_pos = np.asarray(trans[:, :3], dtype=np.float64)

    rot = entry.get("root_rot", entry.get("root_quat", entry.get("root_orient", None)))
    if not isinstance(rot, np.ndarray) or rot.ndim != 2 or int(rot.shape[1]) < 4:
        raise KeyError("Missing/invalid root rotation array (root_rot/root_quat/root_orient)")
    root_rot_xyzw = _maybe_quat_wxyz_to_xyzw(rot)

    fps = entry.get("fps", None)
    if fps is None:
        dt = entry.get("dt", None)
        fps = 1.0 / float(dt) if dt else 30.0
    fps_f = float(fps)
    return dof29, root_pos, root_rot_xyzw, fps_f


def _build_mujoco_model(xml_path: str) -> mujoco.MjModel:
    xml_path = os.path.abspath(xml_path)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(xml_path)
    meshdir = os.path.abspath(os.path.join(os.path.dirname(xml_path), "meshes"))
    with open(xml_path, "r", encoding="utf-8") as f:
        xml = f.read()

    # Ensure MuJoCo can resolve meshes when loading from string.
    xml = xml.replace('meshdir="meshes"', f'meshdir="{meshdir}"')

    # Align toe-link Z offset with TWIST2 stageii local_body_pos convention.
    xml = xml.replace('body name="left_toe_link" pos="0.1 0 -0.035"', 'body name="left_toe_link" pos="0.1 0 -0.02"')
    xml = xml.replace('body name="right_toe_link" pos="0.1 0 -0.035"', 'body name="right_toe_link" pos="0.1 0 -0.02"')

    return mujoco.MjModel.from_xml_string(xml)


def _compute_local_body_pos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_ids: np.ndarray,
    dof_pos: np.ndarray,
) -> np.ndarray:
    dof_pos = np.asarray(dof_pos, dtype=np.float64)
    if dof_pos.ndim != 2 or int(dof_pos.shape[1]) != 29:
        raise ValueError(f"Expected dof_pos (T,29), got {dof_pos.shape}")
    if int(model.nq) < 7 + 29:
        raise ValueError(f"Model nq too small for free joint + 29 dof: nq={model.nq}")

    T = int(dof_pos.shape[0])
    B = int(body_ids.shape[0])
    out = np.empty((T, B, 3), dtype=np.float32)

    # Base at origin, identity quaternion (wxyz) for local body positions.
    data.qpos[:] = 0.0
    data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for t in range(T):
        data.qpos[7 : 7 + 29] = dof_pos[t]
        # Position-only forward pass is enough and noticeably faster than mj_forward.
        mujoco.mj_fwdPosition(model, data)
        out[t] = data.xpos[body_ids].astype(np.float32, copy=False)
    return out


def _resample_uniform_linear(arr: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array, got {arr.shape}")
    T = int(arr.shape[0])
    if T <= 1 or float(src_fps) <= 0.0 or float(dst_fps) <= 0.0:
        return np.ascontiguousarray(arr)

    duration = (T - 1) / float(src_fps)
    T_new = int(round(duration * float(dst_fps))) + 1
    if T_new <= 1:
        T_new = 2

    x = (np.arange(T_new, dtype=np.float64) * float(src_fps) / float(dst_fps)).clip(0.0, float(T - 1))
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, T - 1)
    alpha = (x - i0.astype(np.float64)).reshape(-1, 1)

    out = (1.0 - alpha) * arr[i0] + alpha * arr[i1]
    return np.ascontiguousarray(out)


def _normalize_quat_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 4:
        raise ValueError(f"Expected quaternion array (T,>=4), got {q.shape}")
    q4 = q[:, :4]
    n = np.linalg.norm(q4, axis=1, keepdims=True)
    return np.ascontiguousarray(q4 / np.maximum(n, 1e-12))


def _slerp_xyzw(q0: np.ndarray, q1: np.ndarray, a: np.ndarray) -> np.ndarray:
    q0 = _normalize_quat_xyzw(q0)
    q1 = _normalize_quat_xyzw(q1)
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if q0.shape != q1.shape or q0.shape[0] != a.shape[0]:
        raise ValueError(f"Shape mismatch: q0={q0.shape} q1={q1.shape} a={a.shape}")

    dot = np.sum(q0 * q1, axis=1)
    flip = dot < 0.0
    if np.any(flip):
        q1 = q1.copy()
        q1[flip] *= -1.0
        dot = np.sum(q0 * q1, axis=1)
    dot = np.clip(dot, -1.0, 1.0)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-8

    w0 = np.empty_like(dot)
    w1 = np.empty_like(dot)
    # Linear fallback for tiny angles
    w0[small] = 1.0 - a[small]
    w1[small] = a[small]
    # True slerp
    ok = ~small
    w0[ok] = np.sin((1.0 - a[ok]) * theta[ok]) / sin_theta[ok]
    w1[ok] = np.sin(a[ok] * theta[ok]) / sin_theta[ok]

    out = (w0[:, None] * q0) + (w1[:, None] * q1)
    return _normalize_quat_xyzw(out)


def _resample_uniform_quat_xyzw(q: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    q = _normalize_quat_xyzw(q)
    T = int(q.shape[0])
    if T <= 1 or float(src_fps) <= 0.0 or float(dst_fps) <= 0.0:
        return np.ascontiguousarray(q)

    duration = (T - 1) / float(src_fps)
    T_new = int(round(duration * float(dst_fps))) + 1
    if T_new <= 1:
        T_new = 2

    x = (np.arange(T_new, dtype=np.float64) * float(src_fps) / float(dst_fps)).clip(0.0, float(T - 1))
    i0 = np.floor(x).astype(np.int64)
    i1 = np.minimum(i0 + 1, T - 1)
    alpha = (x - i0.astype(np.float64))
    return _slerp_xyzw(q[i0], q[i1], alpha)


def _resample_to_fps(
    dof_pos: np.ndarray,
    root_pos: np.ndarray,
    root_rot_xyzw: np.ndarray,
    src_fps: float,
    dst_fps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if float(dst_fps) <= 0.0:
        raise ValueError("dst_fps must be > 0")
    if abs(float(src_fps) - float(dst_fps)) < 1e-6:
        return (
            np.ascontiguousarray(np.asarray(dof_pos, dtype=np.float64)),
            np.ascontiguousarray(np.asarray(root_pos, dtype=np.float64)),
            _normalize_quat_xyzw(root_rot_xyzw),
            float(src_fps),
        )
    dof_new = _resample_uniform_linear(dof_pos, src_fps, dst_fps)
    pos_new = _resample_uniform_linear(root_pos, src_fps, dst_fps)
    rot_new = _resample_uniform_quat_xyzw(root_rot_xyzw, src_fps, dst_fps)
    T = int(min(dof_new.shape[0], pos_new.shape[0], rot_new.shape[0]))
    return dof_new[:T], pos_new[:T], rot_new[:T], float(dst_fps)


def _iter_missing_motion_keys(
    dataset: Dict[str, Any],
    *,
    twist2_ids: set[str],
) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    """
    Yield (new_stageii_id, phc_key, entry) for PHC motions that are not in TWIST2.
    """
    for phc_key, entry in dataset.items():
        if not isinstance(phc_key, str) or not isinstance(entry, dict):
            continue
        m = _PHC_KEY_RE.match(phc_key)
        if not m:
            continue
        motion_id = m.group(1)

        if motion_id.startswith("BioMotionLab_NTroje_"):
            new_id = "BMLrub_" + motion_id[len("BioMotionLab_NTroje_") :]
        elif motion_id.startswith("BMLhandball_"):
            new_id = motion_id
        elif motion_id.startswith("TCD_handMocap_"):
            new_id = motion_id
        else:
            continue

        if new_id in twist2_ids:
            continue
        yield new_id, phc_key, entry


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_stageii(path: str, stageii: StageII) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(stageii.as_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phc_pkl",
        type=str,
        required=True,
        help="PHC postprocessed dataset .pkl (joblib) path.",
    )
    parser.add_argument(
        "--twist2_amass_dir",
        type=str,
        default="TWIST2_dataset/AMASS_g1_GMR8",
        help="TWIST2 AMASS stageii folder to compare against.",
    )
    parser.add_argument(
        "--reference_stageii_pkl",
        type=str,
        default="TWIST2_dataset/AMASS_g1_GMR8/KIT_10_LeftTurn01_stageii.pkl",
        help="A TWIST2 stageii pkl used to read link_body_list.",
    )
    parser.add_argument(
        "--mujoco_xml",
        type=str,
        default="assets/g1/g1_mocap_29dof.xml",
        help="MuJoCo XML model used for FK.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output folder for extracted *_stageii.pkl files.",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=30.0,
        help="Resample outputs to this FPS (default: 30).",
    )
    parser.add_argument(
        "--no_resample",
        action="store_true",
        help="Disable resampling; keep source fps and frames as-is.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, only write the first N motions (for debugging).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print counts and a few example names; do not write files.",
    )

    args = parser.parse_args(argv)

    twist2_amass_dir = os.path.abspath(os.path.expanduser(args.twist2_amass_dir))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))

    twist2_ids = _get_twist2_stageii_ids(twist2_amass_dir)
    link_body_list = _load_link_body_list(os.path.abspath(os.path.expanduser(args.reference_stageii_pkl)))

    model = _build_mujoco_model(os.path.abspath(os.path.expanduser(args.mujoco_xml)))
    data = mujoco.MjData(model)
    body_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in link_body_list],
        dtype=np.int32,
    )
    if np.any(body_ids < 0):
        missing = [n for n, i in zip(link_body_list, body_ids.tolist()) if i < 0]
        raise RuntimeError(f"MuJoCo model missing bodies from link_body_list: {missing}")

    dataset = _load_phc_dataset(args.phc_pkl)
    if not isinstance(dataset, dict):
        raise TypeError(f"Expected dataset dict in {args.phc_pkl}, got {type(dataset)}")

    motions = list(_iter_missing_motion_keys(dataset, twist2_ids=twist2_ids))
    motions.sort(key=lambda x: x[0])

    by_source: Dict[str, int] = {"BMLrub": 0, "BMLhandball": 0, "TCD_handMocap": 0}
    for new_id, _, _ in motions:
        if new_id.startswith("BMLrub_"):
            by_source["BMLrub"] += 1
        elif new_id.startswith("BMLhandball_"):
            by_source["BMLhandball"] += 1
        elif new_id.startswith("TCD_handMocap_"):
            by_source["TCD_handMocap"] += 1

    print(f"Found missing motions: total={len(motions)} by_source={by_source}")
    print("Example outputs:")
    for new_id, phc_key, _ in motions[:10]:
        print(f"  {new_id}_stageii.pkl  <-  {phc_key}")

    if args.dry_run:
        return 0

    _safe_mkdir(out_dir)

    limit = int(args.limit)
    written = 0
    skipped_exists = 0
    for idx, (new_id, phc_key, entry) in enumerate(motions):
        if limit > 0 and written >= limit:
            break

        out_path = os.path.join(out_dir, f"{new_id}_stageii.pkl")
        if os.path.exists(out_path):
            skipped_exists += 1
            continue

        try:
            dof_pos, root_pos, root_rot, fps = _extract_phc_arrays(entry)
            T0 = int(min(dof_pos.shape[0], root_pos.shape[0], root_rot.shape[0]))
            if T0 < 2:
                print(f"[skip] too short: {new_id} T={T0}")
                continue
            dof_pos = dof_pos[:T0]
            root_pos = root_pos[:T0]
            root_rot = root_rot[:T0]

            if not bool(args.no_resample):
                dof_pos, root_pos, root_rot, fps = _resample_to_fps(
                    dof_pos,
                    root_pos,
                    root_rot,
                    src_fps=fps,
                    dst_fps=float(args.target_fps),
                )
            local_body_pos = _compute_local_body_pos(model, data, body_ids, dof_pos)
            stageii = StageII(
                fps=fps,
                root_pos=root_pos,
                root_rot_xyzw=root_rot,
                dof_pos=dof_pos,
                local_body_pos=local_body_pos,
                link_body_list=link_body_list,
            )
            _write_stageii(out_path, stageii)
            written += 1
            if written <= 5 or written % 100 == 0:
                print(f"[{written}/{len(motions)}] wrote {os.path.basename(out_path)} (src={phc_key})")
        except Exception as e:
            print(f"[error] {new_id} from {phc_key}: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"Done. written={written} skipped_exists={skipped_exists} out_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
