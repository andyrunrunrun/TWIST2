#!/usr/bin/env python3
import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import redis
from rich import print

from data_utils.params import DEFAULT_MIMIC_OBS
from data_utils.rot_utils import euler_from_quaternion_np, quat_rotate_inverse_np

try:
    import joblib  # type: ignore
except Exception:
    joblib = None


class _SpaceKeyReader:
    def __init__(self) -> None:
        self.enabled = bool(sys.stdin.isatty())
        self._is_windows = os.name == "nt"
        self._fd: Optional[int] = None
        self._old_settings = None

    def __enter__(self) -> "_SpaceKeyReader":
        if not self.enabled:
            return self
        if self._is_windows:
            return self
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._fd = fd
        self._old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        termios.tcflush(fd, termios.TCIFLUSH)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.enabled:
            return
        if self._is_windows:
            return
        if self._fd is None or self._old_settings is None:
            return
        import termios

        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def flush(self) -> None:
        if not self.enabled:
            return
        if self._is_windows:
            import msvcrt

            while msvcrt.kbhit():
                msvcrt.getwch()
            return
        if self._fd is None:
            return
        import termios

        termios.tcflush(self._fd, termios.TCIFLUSH)

    def wait_for_space(self) -> None:
        if not self.enabled:
            return
        if self._is_windows:
            import msvcrt

            while True:
                if msvcrt.getwch() == " ":
                    return
        while True:
            if sys.stdin.read(1) == " ":
                return

    def poll_space(self) -> bool:
        if not self.enabled:
            return False
        if self._is_windows:
            import msvcrt

            if not msvcrt.kbhit():
                return False
            return msvcrt.getwch() == " "

        import select

        rlist, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not rlist:
            return False
        return sys.stdin.read(1) == " "


def _rate_sleep(last_time: float, period_s: float) -> float:
    now = time.time()
    elapsed = now - last_time
    if elapsed < period_s:
        time.sleep(period_s - elapsed)
        return last_time + period_s
    return now


def _wrap_to_pi(angles: np.ndarray) -> np.ndarray:
    angles = np.asarray(angles, dtype=np.float32)
    return ((angles + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32, copy=False)


def _normalize_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    if q.ndim != 2 or int(q.shape[1]) < 4:
        raise ValueError(f"Expected quaternion array shape (T,>=4), got {q.shape}")
    q4 = q[:, :4]
    n = np.linalg.norm(q4, axis=1, keepdims=True)
    q4 = q4 / np.maximum(n, 1e-8)
    return q4.astype(np.float32, copy=False)


def _load_dataset(dataset_path: str) -> Dict[str, Any]:
    dataset_path = os.path.expanduser(str(dataset_path))
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.abspath(dataset_path)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    if joblib is not None:
        print(f"[PHC] Loading dataset with joblib (no mmap): {dataset_path}")
        data = joblib.load(dataset_path, mmap_mode=None)
    else:
        print(f"[PHC] joblib not available; falling back to pickle.load: {dataset_path}")
        with open(dataset_path, "rb") as f:
            data = pickle.load(f)

    if not isinstance(data, dict):
        raise TypeError(f"Expected dataset dict at {dataset_path}, got {type(data)}")
    return data


def _extract_entry_arrays(entry: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dof = entry.get("dof", None)
    if not isinstance(dof, np.ndarray):
        raise KeyError("Missing numpy array field 'dof' in dataset entry")
    if dof.ndim != 2 or int(dof.shape[1]) < 29:
        raise ValueError(f"Expected dof shape (T,>=29), got {dof.shape}")

    trans = entry.get("root_trans_offset", None)
    if trans is None:
        trans = entry.get("root_trans", None)
    if trans is None:
        trans = entry.get("root_pos", None)
    if not isinstance(trans, np.ndarray):
        raise KeyError("Missing numpy array field 'root_trans_offset' (or 'root_trans'/'root_pos') in dataset entry")
    if trans.ndim != 2 or int(trans.shape[1]) < 3:
        raise ValueError(f"Expected root translation shape (T,>=3), got {trans.shape}")

    rot = entry.get("root_rot", None)
    if rot is None:
        rot = entry.get("root_quat", None)
    if rot is None:
        rot = entry.get("root_orient", None)
    if not isinstance(rot, np.ndarray):
        raise KeyError("Missing numpy array field 'root_rot' (or 'root_quat'/'root_orient') in dataset entry")
    if rot.ndim != 2 or int(rot.shape[1]) < 4:
        raise ValueError(f"Expected root rotation shape (T,>=4), got {rot.shape}")

    T = int(min(dof.shape[0], trans.shape[0], rot.shape[0]))
    if T <= 0:
        raise ValueError("Empty sequence after length alignment")

    dof29 = np.asarray(dof[:T, :29], dtype=np.float32)
    trans3 = np.asarray(trans[:T, :3], dtype=np.float32)
    rot4 = _normalize_quat(rot[:T, :4])
    return dof29, trans3, rot4


def _build_mimic_obs_35d(
    dof29: np.ndarray,
    trans3: np.ndarray,
    rot4: np.ndarray,
    *,
    dt: float,
    quat_scalar_first: bool,
    override_root_z: Optional[float],
) -> np.ndarray:
    T = int(dof29.shape[0])
    if T <= 0:
        raise ValueError("Empty clip")

    dt = float(dt)
    if dt <= 0.0:
        raise ValueError("dt must be > 0")

    root_vel_world = np.zeros((T, 3), dtype=np.float32)
    if T > 1:
        root_vel_world[:-1] = (trans3[1:] - trans3[:-1]) / dt
        root_vel_world[-1] = root_vel_world[-2]

    root_vel_local = quat_rotate_inverse_np(rot4, root_vel_world, scalar_first=quat_scalar_first)

    roll, pitch, yaw = euler_from_quaternion_np(rot4, scalar_first=quat_scalar_first)
    yaw = np.asarray(yaw, dtype=np.float32)
    yaw_rate = np.zeros((T,), dtype=np.float32)
    if T > 1:
        yaw_rate[:-1] = _wrap_to_pi(yaw[1:] - yaw[:-1]) / dt
        yaw_rate[-1] = yaw_rate[-2]

    root_pos_z = trans3[:, 2:3].astype(np.float32, copy=False)
    if override_root_z is not None:
        root_pos_z = np.full((T, 1), float(override_root_z), dtype=np.float32)

    mimic_obs = np.concatenate(
        [
            root_vel_local[:, :2],
            root_pos_z,
            roll.reshape(T, 1).astype(np.float32, copy=False),
            pitch.reshape(T, 1).astype(np.float32, copy=False),
            yaw_rate.reshape(T, 1),
            dof29,
        ],
        axis=1,
    )
    if mimic_obs.shape[1] != 35:
        raise RuntimeError(f"Expected mimic_obs 35D, got {mimic_obs.shape}")
    return mimic_obs.astype(np.float32, copy=False)


def _iter_valid_keys(dataset: Dict[str, Any]) -> Iterable[str]:
    for k, entry in dataset.items():
        if not isinstance(entry, dict):
            continue
        dof = entry.get("dof", None)
        if not isinstance(dof, np.ndarray) or dof.ndim != 2 or int(dof.shape[1]) < 29 or int(dof.shape[0]) < 2:
            continue
        trans = entry.get("root_trans_offset", entry.get("root_trans", entry.get("root_pos", None)))
        rot = entry.get("root_rot", entry.get("root_quat", entry.get("root_orient", None)))
        if not isinstance(trans, np.ndarray) or trans.ndim != 2 or int(trans.shape[1]) < 3 or int(trans.shape[0]) < 2:
            continue
        if not isinstance(rot, np.ndarray) or rot.ndim != 2 or int(rot.shape[1]) < 4 or int(rot.shape[0]) < 2:
            continue
        yield str(k)


def _publish_mimic_obs(
    pipe: "redis.client.Pipeline",
    *,
    robot: str,
    mimic_obs_35d: np.ndarray,
    t_action_ms: int,
) -> None:
    pipe.set(f"action_body_{robot}", json.dumps(mimic_obs_35d.tolist()))
    pipe.set(f"action_hand_left_{robot}", json.dumps(np.zeros(7, dtype=np.float32).tolist()))
    pipe.set(f"action_hand_right_{robot}", json.dumps(np.zeros(7, dtype=np.float32).tolist()))
    pipe.set(f"action_neck_{robot}", json.dumps(np.zeros(2, dtype=np.float32).tolist()))
    pipe.set("t_action", int(t_action_ms))


def _publish_interp(
    pipe: "redis.client.Pipeline",
    *,
    robot: str,
    obs0: np.ndarray,
    obs1: np.ndarray,
    steps: int,
    period_s: float,
    last_time: float,
) -> float:
    if steps <= 0:
        return last_time
    obs0 = np.asarray(obs0, dtype=np.float32)
    obs1 = np.asarray(obs1, dtype=np.float32)
    if obs0.shape != (35,) or obs1.shape != (35,):
        raise ValueError(f"Expected obs shapes (35,), got {obs0.shape} and {obs1.shape}")
    for i in range(int(steps)):
        alpha = float(i + 1) / float(steps)
        interp = (1.0 - alpha) * obs0 + alpha * obs1
        _publish_mimic_obs(pipe, robot=robot, mimic_obs_35d=interp, t_action_ms=int(time.time() * 1000))
        pipe.execute()
        last_time = _rate_sleep(last_time, period_s)
    return last_time


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream PHC dataset sequences as TWIST2 35D mimic_obs via Redis.")
    parser.add_argument("--dataset_path", type=str, required=True, help="PHC dataset .pkl path (joblib/pickle dict).")
    parser.add_argument("--redis_ip", type=str, default="localhost", help="Redis host/IP")
    parser.add_argument("--redis_port", type=int, default=6379)
    parser.add_argument("--redis_db", type=int, default=0)
    parser.add_argument("--robot", type=str, default="unitree_g1_with_hands", choices=["unitree_g1", "unitree_g1_with_hands"])

    parser.add_argument("--rate_hz", type=float, default=50.0, help="Publish rate (Hz).")
    parser.add_argument(
        "--quat_order",
        type=str,
        default="xyzw",
        choices=["xyzw", "wxyz"],
        help="Quaternion storage order in dataset.",
    )
    parser.add_argument(
        "--override_root_z",
        type=float,
        default=None,
        help="If set, force mimic_obs root_pos_z to this constant value (meters).",
    )

    parser.add_argument("--dataset_key", type=str, default="", help="If set, always play this dataset key.")
    parser.add_argument("--sample_mode", type=str, default="sequential", choices=["sequential", "random"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=0, help="Start frame index (clamped).")
    parser.add_argument("--random_start", action="store_true", help="Randomize start frame each time (overrides --start).")
    parser.add_argument("--clip_len", type=int, default=0, help="Frames to play (0 means play until end).")
    parser.add_argument("--loop", action="store_true", help="Loop (and in sequential mode, advance to next key).")

    parser.add_argument(
        "--start_interp_seconds",
        type=float,
        default=2.0,
        help="Interpolate into the first frame of each clip over this duration (seconds).",
    )
    parser.add_argument(
        "--wait_for_space",
        dest="wait_for_space",
        action="store_true",
        help="Start paused; press SPACE to start/pause/resume streaming.",
    )
    parser.add_argument(
        "--no_wait_for_space",
        dest="wait_for_space",
        action="store_false",
        help="Disable SPACE-to-start pause on launch.",
    )
    parser.set_defaults(wait_for_space=True)
    parser.add_argument(
        "--exit_interp_seconds",
        type=float,
        default=2.0,
        help="Interpolate back to default pose on exit (seconds).",
    )
    parser.add_argument("--print_keys", action="store_true", help="Print valid dataset keys and exit.")
    parser.add_argument("--print_first_n", type=int, default=50, help="When --print_keys, print at most N keys.")

    args = parser.parse_args()

    if args.rate_hz <= 0:
        raise ValueError("--rate_hz must be > 0")
    period_s = 1.0 / float(args.rate_hz)
    quat_scalar_first = str(args.quat_order) == "wxyz"

    dataset = _load_dataset(args.dataset_path)

    valid_keys: List[str] = []
    if args.dataset_key:
        if args.dataset_key not in dataset:
            raise KeyError(f"--dataset_key={args.dataset_key!r} not found in dataset")
        valid_keys = [str(args.dataset_key)]
    else:
        valid_keys = sorted(list(_iter_valid_keys(dataset)))
        if not valid_keys:
            raise ValueError("No valid sequences found in dataset (need dof/root_trans/root_rot with T>=2).")

    if args.print_keys:
        n = int(max(0, args.print_first_n))
        print(f"[PHC] Valid sequences: {len(valid_keys)}")
        for k in valid_keys[:n]:
            print(k)
        if len(valid_keys) > n:
            print(f"... (showing first {n})")
        return

    r = redis.Redis(host=args.redis_ip, port=int(args.redis_port), db=int(args.redis_db))
    r.ping()
    pipe = r.pipeline()

    rng = np.random.default_rng(int(args.seed))
    seq_cursor = 0

    default_mimic = np.asarray(DEFAULT_MIMIC_OBS[args.robot], dtype=np.float32)
    if default_mimic.shape != (35,):
        raise RuntimeError(f"Unexpected DEFAULT_MIMIC_OBS shape: {default_mimic.shape}")

    last_mimic = default_mimic.copy()
    _publish_mimic_obs(pipe, robot=args.robot, mimic_obs_35d=last_mimic, t_action_ms=int(time.time() * 1000))
    pipe.execute()

    last_time = time.time()
    print(f"[PHC] Ready. Redis={args.redis_ip}:{args.redis_port}/{args.redis_db} robot={args.robot} rate={args.rate_hz:g}Hz")
    print(f"[PHC] Keys: {len(valid_keys)}  sample_mode={args.sample_mode}  loop={bool(args.loop)}")
    print("[PHC] Hotkey: SPACE toggles pause/resume (Ctrl+C to exit)")

    paused = bool(args.wait_for_space)
    key_reader = _SpaceKeyReader()
    if paused and not key_reader.enabled:
        print("[PHC] NOTE: stdin is not a TTY; cannot read SPACE. Starting immediately.")
        paused = False

    interp_s = float(max(0.0, args.start_interp_seconds))
    interp_steps = int(round(interp_s * float(args.rate_hz))) if interp_s > 0.0 else 0

    try:
        with key_reader:
            if paused:
                print("[PHC] Paused. Press SPACE to start.")
                key_reader.wait_for_space()
                key_reader.flush()
                paused = False

            while True:
                if args.sample_mode == "random":
                    key = valid_keys[int(rng.integers(low=0, high=len(valid_keys)))]
                else:
                    key = valid_keys[int(seq_cursor % len(valid_keys))]

                entry = dataset.get(key, None)
                if not isinstance(entry, dict):
                    raise RuntimeError(f"Dataset entry for key={key!r} is not a dict")

                dof29, trans3, rot4 = _extract_entry_arrays(entry)
                T = int(dof29.shape[0])
                if T < 2:
                    raise RuntimeError(f"Sequence too short for key={key!r}: T={T}")

                if args.random_start:
                    start = int(rng.integers(low=0, high=T))
                else:
                    start = int(max(0, args.start))
                if start >= T:
                    start = max(0, T - 1)

                end = T
                if int(args.clip_len) > 0:
                    end = min(T, start + int(args.clip_len))
                if end - start < 2:
                    print(f"[PHC] Skipping too-short clip: key={key!r} start={start} end={end} (T={T})")
                    seq_cursor += 1
                    if not args.loop and args.sample_mode == "sequential" and seq_cursor >= len(valid_keys):
                        break
                    continue

                dof_clip = dof29[start:end]
                trans_clip = trans3[start:end]
                rot_clip = rot4[start:end]

                mimic_traj = _build_mimic_obs_35d(
                    dof_clip,
                    trans_clip,
                    rot_clip,
                    dt=period_s,
                    quat_scalar_first=quat_scalar_first,
                    override_root_z=args.override_root_z,
                )

                # Smooth transition into the first frame of this clip.
                if interp_steps > 0:
                    last_time = _publish_interp(
                        pipe,
                        robot=args.robot,
                        obs0=last_mimic,
                        obs1=mimic_traj[0],
                        steps=interp_steps,
                        period_s=period_s,
                        last_time=last_time,
                    )
                    last_mimic = mimic_traj[0].copy()

                print(f"[PHC] Streaming key={key!r} frames=[{start}:{end}) ({end-start} frames)")
                frame_i = 0
                while frame_i < int(mimic_traj.shape[0]):
                    if key_reader.poll_space():
                        key_reader.flush()
                        if not paused:
                            # Pause: interpolate to default and hold.
                            if interp_steps > 0:
                                last_time = _publish_interp(
                                    pipe,
                                    robot=args.robot,
                                    obs0=last_mimic,
                                    obs1=default_mimic,
                                    steps=interp_steps,
                                    period_s=period_s,
                                    last_time=last_time,
                                )
                            last_mimic = default_mimic.copy()
                            paused = True
                            print("[PHC] Paused.")
                        else:
                            # Resume: interpolate to current frame target.
                            target = mimic_traj[frame_i]
                            if interp_steps > 0:
                                last_time = _publish_interp(
                                    pipe,
                                    robot=args.robot,
                                    obs0=last_mimic,
                                    obs1=target,
                                    steps=interp_steps,
                                    period_s=period_s,
                                    last_time=last_time,
                                )
                            last_mimic = target.copy()
                            paused = False
                            print("[PHC] Resumed.")

                    if paused:
                        _publish_mimic_obs(pipe, robot=args.robot, mimic_obs_35d=default_mimic, t_action_ms=int(time.time() * 1000))
                        pipe.execute()
                        last_time = _rate_sleep(last_time, period_s)
                        continue

                    mimic = mimic_traj[frame_i]
                    _publish_mimic_obs(pipe, robot=args.robot, mimic_obs_35d=mimic, t_action_ms=int(time.time() * 1000))
                    pipe.execute()
                    last_mimic = mimic
                    last_time = _rate_sleep(last_time, period_s)
                    frame_i += 1

                seq_cursor += 1
                if not args.loop:
                    break
                # After finishing one clip, enter pause mode and wait for SPACE to continue
                # (consistent with "step-through" dataset browsing).
                if key_reader.enabled:
                    if interp_steps > 0:
                        last_time = _publish_interp(
                            pipe,
                            robot=args.robot,
                            obs0=last_mimic,
                            obs1=default_mimic,
                            steps=interp_steps,
                            period_s=period_s,
                            last_time=last_time,
                        )
                    last_mimic = default_mimic.copy()
                    paused = True
                    key_reader.flush()
                    print("[PHC] Clip finished. Paused. Press SPACE to continue.")

                    while True:
                        if key_reader.poll_space():
                            key_reader.flush()
                            paused = False
                            print("[PHC] Continuing...")
                            break
                        _publish_mimic_obs(
                            pipe,
                            robot=args.robot,
                            mimic_obs_35d=default_mimic,
                            t_action_ms=int(time.time() * 1000),
                        )
                        pipe.execute()
                        last_time = _rate_sleep(last_time, period_s)

    except KeyboardInterrupt:
        print("\n[PHC] KeyboardInterrupt, exiting...")
    finally:
        # Interpolate back to default pose for safety.
        back_s = float(max(0.0, args.exit_interp_seconds))
        back_steps = int(round(back_s * float(args.rate_hz))) if back_s > 0.0 else 0
        if back_steps > 0:
            last_time = _publish_interp(
                pipe,
                robot=args.robot,
                obs0=last_mimic,
                obs1=default_mimic,
                steps=back_steps,
                period_s=period_s,
                last_time=last_time,
            )
        _publish_mimic_obs(pipe, robot=args.robot, mimic_obs_35d=default_mimic, t_action_ms=int(time.time() * 1000))
        pipe.execute()


if __name__ == "__main__":
    main()
