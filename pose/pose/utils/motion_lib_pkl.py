import os, pickle, yaml
from pathlib import Path
from itertools import islice
import torch
from pose.utils.torch_utils import quat_diff, quat_to_exp_map, slerp, euler_from_quaternion
from tqdm import tqdm
from pose.utils.isaacgym_torch_utils import quat_rotate_inverse, quat_mul, quat_conjugate
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

def smooth(x, box_pts, device):
    box = torch.ones(box_pts, device=device) / box_pts
    num_channels = x.shape[1]
    x_reshaped = x.T.unsqueeze(0)
    smoothed = torch.nn.functional.conv1d(
        x_reshaped,
        box.view(1, 1, -1).expand(num_channels, 1, -1),
        groups=num_channels,
        padding='same'
    )
    return smoothed.squeeze(0).T


class MotionLib:
    def __init__(self, motion_file, device, 
                 motion_decompose=False, 
                 motion_smooth=True, 
                 motion_height_adjust=False,
                 sample_ratio=1.0, # only sample a portion of the motion
                 max_motions: int = -1, # for YAML configs: only load first N after filtering
                 motion_ids: str = "", # for YAML configs: select a subset by indices, e.g. "0,3,10-20"
                 shuffle_motions: bool = False, # for YAML configs: shuffle before applying max_motions (ignored if motion_ids given)
                 shuffle_seed: int = 0,
                 store_on_cpu: bool = True, # keep dataset tensors on CPU, move slices to GPU on demand
                 gpu_cache_gib: float = 4.0, # cache active motions on GPU up to this budget (GiB); 0 disables
                 # Optional HY-Motion single-stream feature support (loaded per motion on demand)
                 hy_feat_cache_motions: int = 0,  # CPU LRU cache size (motions). 0 disables.
                 ):
        self._device = torch.device(device)
        self._store_on_cpu = bool(store_on_cpu)
        self._storage_device = torch.device("cpu") if self._store_on_cpu else self._device
        self._gpu_cache_gib = float(gpu_cache_gib)

        # motion augmentation by decomposing long motion into short motions
        self._motion_decompose = motion_decompose
        # motion smoothing
        self._motion_smooth = motion_smooth
        # motion height adjustment
        self._motion_height_adjust = motion_height_adjust
        # sample a portion of the motion
        self._sample_ratio = sample_ratio

        self._max_motions = int(max_motions) if max_motions is not None else -1
        self._motion_ids_spec = str(motion_ids) if motion_ids is not None else ""
        self._shuffle_motions = bool(shuffle_motions)
        self._shuffle_seed = int(shuffle_seed) if shuffle_seed is not None else 0

        # Optional HY-Motion single-stream features (.npz alongside a separate root).
        self._hy_feat_files: List[str] = []
        self._hy_feat_dim = 0
        self._hy_feat_t = 1.0
        self._hy_feat_cache_motions = int(hy_feat_cache_motions) if hy_feat_cache_motions is not None else 0
        self._hy_feat_cache_cpu: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        
        # load motions
        self._load_motions(motion_file)
        
        self._init_gpu_cache()
        
        
    def _load_motions(self, motion_file):
        self._motion_names = []
        self._motion_weights = []
        self._motion_fps = []
        self._motion_dt = []
        self._motion_num_frames = []
        self._motion_lengths = []
        self._motion_files = []
        
        self._motion_root_pos_delta = []
        self._motion_root_pos = []
        self._motion_root_rot = []
        self._motion_root_vel = []
        self._motion_root_ang_vel = []
        self._motion_dof_pos = []
        self._motion_root_pos_delta_local = []
        self._motion_root_rot_delta_local = []
        self._motion_dof_vel = []
        self._motion_local_body_pos = []
        self._body_link_list = []
        
        motion_files, motion_weights, hy_feat_files, hy_feat_dim, hy_feat_t = self._fetch_motion_files(motion_file)
        self._hy_feat_files = [str(p) for p in hy_feat_files]
        self._hy_feat_dim = int(hy_feat_dim)
        self._hy_feat_t = float(hy_feat_t)
        num_motion_files = len(motion_files)
        
        num_sub_motions_total = 0
            
        for i in tqdm(range(num_motion_files), desc="[MotionLib] Loading motions"):
            if torch.rand(1) > self._sample_ratio and num_motion_files > 1:
                continue
            
            curr_file = motion_files[i]
            if not os.path.exists(curr_file):
                print(f"Motion file {curr_file} does not exist")
                continue

            try:
                motion_data = self._load_motion_data(curr_file)
                if motion_data is None:
                    continue
                fps = motion_data["fps"]
            except Exception as e:
                # NumPy 2.x pickles are not compatible with NumPy 1.x (e.g. py38 IsaacGym env).
                # Avoid attempting module hacks here as they can hard-crash the interpreter.
                if isinstance(e, ModuleNotFoundError) and "numpy._core" in str(e):
                    print(
                        "Error loading motion file (NumPy 2.x pickle detected). "
                        "Please convert motions to .npz first and re-run.\n"
                        f"  file: {curr_file}\n"
                        f"  error: {e}"
                    )
                else:
                    print(f"Error loading motion file {curr_file}: {e}")
                continue
            curr_weight = motion_weights[i]
            # Create tensors on CPU first then move to target device.
            # This avoids some CUDA-side conversion paths that may hard-crash (segfault) in certain setups.
            root_pos = self._to_storage_tensor(motion_data["root_pos"], dtype=torch.float)
            root_rot = self._to_storage_tensor(motion_data["root_rot"], dtype=torch.float)
            dof_pos = self._to_storage_tensor(motion_data["dof_pos"], dtype=torch.float)
            local_body_pos = self._to_storage_tensor(motion_data["local_body_pos"], dtype=torch.float)
            if self._body_link_list is None or len(self._body_link_list) == 0:
                self._body_link_list = motion_data["link_body_list"]
            num_frames = root_pos.shape[0]
            motion_len_s = 1.0 / fps * (num_frames - 1)
            
            if self._motion_height_adjust:
                # compute the lowest body part in reference motion
                body_pos = local_body_pos + root_pos.unsqueeze(1)
                lowest_body_part = torch.min(body_pos[..., 2])
                # adjust the height of the root position
                root_pos[..., 2] -= lowest_body_part
                
            try:
                self._add_motions(root_pos, root_rot, dof_pos, local_body_pos, fps, curr_weight, curr_file)
            except Exception as e:
                print(f"Error adding motion {curr_file}: {e}")
                continue
            
            
            if self._motion_decompose:
                # Decompose long motion into short motions
                base_motion_len_s = 10.0 # 10 seconds for each sub-motion
                # base_motion_len_s = 20.0 # 20 seconds for each sub-motion
                # base_motion_len_s = 30.0 # 30 seconds for each sub-motion
                if motion_len_s < base_motion_len_s:
                    continue
                # divide motion into sub-motions of base_motion_len
                num_sub_motions = int(motion_len_s / base_motion_len_s)
                # if the motion is longer than the base_motion_len, add one more sub-motion
                if motion_len_s > base_motion_len_s * num_sub_motions:
                    num_sub_motions += 1
                
                num_sub_motions_total += num_sub_motions
                for i in range(num_sub_motions):
                    start_idx = int(i * base_motion_len_s * fps)
                    end_idx = int(start_idx + base_motion_len_s * fps)
                    
                    # get the sub-motion
                    sub_root_pos = root_pos[start_idx:end_idx]
                    sub_root_rot = root_rot[start_idx:end_idx]
                    sub_dof_pos = dof_pos[start_idx:end_idx]
                    sub_local_body_pos = local_body_pos[start_idx:end_idx]
                    # sub_weight = curr_weight + i # we increase the weight of the sub-motion by i
                    sub_weight = curr_weight
                    self._add_motions(sub_root_pos, sub_root_rot, sub_dof_pos, sub_local_body_pos, fps, sub_weight, curr_file)
                # print(f"Decomposed {curr_file} into {num_sub_motions} sub-motions")
        
        print(f"Total number of sub-motions: {num_sub_motions_total}")
                        
        assert len(self._motion_weights) == len(self._motion_names), f"len(self._motion_weights) = {len(self._motion_weights)}, len(self._motion_names) = {len(self._motion_names)}"
        assert len(self._motion_weights) == len(self._motion_files), f"len(self._motion_weights) = {len(self._motion_weights)}, len(self._motion_files) = {len(self._motion_files)}"
        assert len(self._motion_weights) == len(self._motion_fps), f"len(self._motion_weights) = {len(self._motion_weights)}, len(self._motion_fps) = {len(self._motion_fps)}"

        if len(self._motion_weights) == 0:
            raise RuntimeError(
                f"No valid motions loaded from {motion_file}. "
                "If you are using a dataset generated with NumPy 2.x, convert all *.pkl to *.npz first."
            )
        
        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float, device=self._device)
        self._motion_weights /= torch.sum(self._motion_weights)
        
        self._motion_fps = torch.tensor(self._motion_fps, dtype=torch.float, device=self._device)
        self._motion_dt = torch.tensor(self._motion_dt, dtype=torch.float, device=self._device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, dtype=torch.long, device=self._device)
        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float, device=self._device)

        # Per-motion deltas are small; keep them on compute device.
        self._motion_root_pos_delta = torch.stack(self._motion_root_pos_delta, dim=0).to(self._device)
        
        # Large per-frame tensors stay on storage device (CPU by default).
        self._motion_root_pos = torch.cat(self._motion_root_pos, dim=0).to(self._storage_device)
        self._motion_root_rot = torch.cat(self._motion_root_rot, dim=0).to(self._storage_device)
        self._motion_root_vel = torch.cat(self._motion_root_vel, dim=0).to(self._storage_device)
        self._motion_root_ang_vel = torch.cat(self._motion_root_ang_vel, dim=0).to(self._storage_device)
        self._motion_dof_pos = torch.cat(self._motion_dof_pos, dim=0).to(self._storage_device)
        self._motion_dof_vel = torch.cat(self._motion_dof_vel, dim=0).to(self._storage_device)
        self._motion_local_body_pos = torch.cat(self._motion_local_body_pos, dim=0).to(self._storage_device)
        self._motion_root_pos_delta_local = torch.cat(self._motion_root_pos_delta_local, dim=0).to(self._storage_device)
        self._motion_root_rot_delta_local = torch.cat(self._motion_root_rot_delta_local, dim=0).to(self._storage_device)
        
        lengths_shifted = self._motion_num_frames.roll(1)
        lengths_shifted[0] = 0
        self._motion_start_idx = lengths_shifted.cumsum(0)
        self._motion_start_idx_cpu = self._motion_start_idx.to("cpu")
        self._motion_num_frames_cpu = self._motion_num_frames.to("cpu")
        
        num_motions = self.num_motions()
        self._motion_ids = torch.arange(num_motions, dtype=torch.long, device=self._device)
        
        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

    def _add_motions(self, root_pos, root_rot, dof_pos, local_body_pos, fps, curr_weight, curr_file):
        dt = 1.0 / fps
        num_frames = root_pos.shape[0]
        curr_len = dt * (num_frames - 1)
        
        root_pos_delta = root_pos[-1] - root_pos[0]
        root_pos_delta[..., -1] = 0.0
        
        root_vel = self._finite_difference(root_pos, dt)
        
        # compute the delta pos per frame
        root_pos_delta_local = torch.zeros_like(root_pos)
        root_pos_delta_local[1:, :] = root_pos[1:, :] - root_pos[:-1, :] # cur frame delta pos = cur frame pos - last frame pos
        root_pos_delta_local[0, :] = 0.0 # first frame delta pos = 0
        root_pos_delta_local[1:, :] = quat_rotate_inverse(root_rot[:-1, :], root_pos_delta_local[1:, :]) # rotate the delta pos to local frame via last frame rot
        
        # compute the delta rot per frame
        root_rot_delta_local = torch.zeros_like(root_pos)
        root_rot_delta_local[1:, :] = euler_from_quaternion(quat_diff(root_rot[1:, :], root_rot[:-1, :])) # cur frame delta rot = cur frame rot - last frame rot
        root_rot_delta_local[0, :] = 0.0
        root_rot_delta_local[1:, :] = quat_rotate_inverse(root_rot[:-1, :], root_rot_delta_local[1:, :]) # rotate the delta rot to local frame via last frame rot
        
        root_ang_vel = self._compute_so3_derivative(root_rot, dt)
        
        dof_vel = self._finite_difference(dof_pos, dt)
        
        self._motion_weights.append(curr_weight)
        self._motion_fps.append(fps)
        self._motion_dt.append(dt)
        self._motion_num_frames.append(num_frames)
        self._motion_lengths.append(curr_len)
        self._motion_files.append(curr_file)
        
        self._motion_root_pos_delta.append(root_pos_delta)
        self._motion_root_pos.append(root_pos)
        self._motion_root_rot.append(root_rot)
        self._motion_root_vel.append(root_vel)
        self._motion_root_ang_vel.append(root_ang_vel)
        self._motion_dof_pos.append(dof_pos)
        self._motion_root_pos_delta_local.append(root_pos_delta_local)
        self._motion_root_rot_delta_local.append(root_rot_delta_local)
        self._motion_dof_vel.append(dof_vel)
        self._motion_local_body_pos.append(local_body_pos)
        self._motion_names.append(os.path.basename(curr_file))

    def _to_storage_tensor(self, x, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=self._storage_device, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=self._storage_device)

    def _cache_bytes_per_frame(self) -> int:
        D = int(self._motion_dof_pos.shape[-1])
        B = int(self._motion_local_body_pos.shape[1])
        floats_per_frame = 19 + 2 * D + 3 * B
        bytes_per_frame = int(floats_per_frame * 4)
        if self._hy_feat_dim > 0:
            # HY features are cached as float16 to reduce VRAM pressure.
            bytes_per_frame += int(self._hy_feat_dim * 2)
        return bytes_per_frame

    def _init_gpu_cache(self) -> None:
        self._gpu_cache_enabled = (
            self._device.type == "cuda"
            and self._store_on_cpu
            and self._gpu_cache_gib > 0.0
        )
        if not self._gpu_cache_enabled:
            self._cache_capacity_frames = 0
            return

        max_bytes = int(self._gpu_cache_gib * (1024 ** 3))
        bytes_per_frame = self._cache_bytes_per_frame()
        self._cache_max_frames = max(1, max_bytes // bytes_per_frame)
        self._cache_bytes_per_frame_val = bytes_per_frame

        self._cache_capacity_frames = 0
        self._cache_free: List[Tuple[int, int]] = []
        self._cache_meta: Dict[int, Tuple[int, int]] = {}
        self._cache_lru: "OrderedDict[int, None]" = OrderedDict()
        self._cache_frames_used = 0

        self._cache_offset = torch.full(
            (self.num_motions(),), -1, device=self._device, dtype=torch.int32
        )
        self._cache_len = torch.zeros(
            (self.num_motions(),), device=self._device, dtype=torch.int32
        )

        self._cache_root_pos = None
        self._cache_root_rot = None
        self._cache_root_vel = None
        self._cache_root_ang_vel = None
        self._cache_dof_pos = None
        self._cache_dof_vel = None
        self._cache_local_body_pos = None
        self._cache_root_pos_delta_local = None
        self._cache_root_rot_delta_local = None
        self._cache_hy_feat = None

    def _cache_is_initialized(self) -> bool:
        return self._cache_capacity_frames > 0

    def _cache_grow_to(self, new_capacity_frames: int) -> None:
        new_capacity_frames = int(min(new_capacity_frames, self._cache_max_frames))
        if new_capacity_frames <= self._cache_capacity_frames:
            return
        old_cap = int(self._cache_capacity_frames)

        D = int(self._motion_dof_pos.shape[-1])
        B = int(self._motion_local_body_pos.shape[1])

        def alloc(shape_tail, *, dtype: torch.dtype = torch.float32):
            return torch.empty((new_capacity_frames, *shape_tail), device=self._device, dtype=dtype)

        new_root_pos = alloc((3,))
        new_root_rot = alloc((4,))
        new_root_vel = alloc((3,))
        new_root_ang_vel = alloc((3,))
        new_dof_pos = alloc((D,))
        new_dof_vel = alloc((D,))
        new_local_body_pos = alloc((B, 3))
        new_root_pos_delta_local = alloc((3,))
        new_root_rot_delta_local = alloc((3,))
        new_hy_feat = None
        if self._hy_feat_dim > 0:
            new_hy_feat = alloc((int(self._hy_feat_dim),), dtype=torch.float16)

        if old_cap > 0:
            new_root_pos[:old_cap].copy_(self._cache_root_pos)
            new_root_rot[:old_cap].copy_(self._cache_root_rot)
            new_root_vel[:old_cap].copy_(self._cache_root_vel)
            new_root_ang_vel[:old_cap].copy_(self._cache_root_ang_vel)
            new_dof_pos[:old_cap].copy_(self._cache_dof_pos)
            new_dof_vel[:old_cap].copy_(self._cache_dof_vel)
            new_local_body_pos[:old_cap].copy_(self._cache_local_body_pos)
            new_root_pos_delta_local[:old_cap].copy_(self._cache_root_pos_delta_local)
            new_root_rot_delta_local[:old_cap].copy_(self._cache_root_rot_delta_local)
            if new_hy_feat is not None and self._cache_hy_feat is not None:
                new_hy_feat[:old_cap].copy_(self._cache_hy_feat)

        self._cache_root_pos = new_root_pos
        self._cache_root_rot = new_root_rot
        self._cache_root_vel = new_root_vel
        self._cache_root_ang_vel = new_root_ang_vel
        self._cache_dof_pos = new_dof_pos
        self._cache_dof_vel = new_dof_vel
        self._cache_local_body_pos = new_local_body_pos
        self._cache_root_pos_delta_local = new_root_pos_delta_local
        self._cache_root_rot_delta_local = new_root_rot_delta_local
        if new_hy_feat is not None:
            self._cache_hy_feat = new_hy_feat

        self._cache_capacity_frames = new_capacity_frames
        self._cache_free_segment_add(old_cap, new_capacity_frames - old_cap)

    def _cache_free_segment_add(self, start: int, length: int) -> None:
        if length <= 0:
            return
        start = int(start)
        length = int(length)
        end = start + length
        free = self._cache_free
        free.append((start, length))
        free.sort(key=lambda x: x[0])

        merged: List[Tuple[int, int]] = []
        for s, l in free:
            if not merged:
                merged.append((s, l))
                continue
            ps, pl = merged[-1]
            pe = ps + pl
            if s <= pe:
                ne = max(pe, s + l)
                merged[-1] = (ps, ne - ps)
            else:
                merged.append((s, l))
        self._cache_free = merged

    def _cache_alloc_segment(self, length: int) -> Optional[int]:
        length = int(length)
        for i, (s, l) in enumerate(self._cache_free):
            if l >= length:
                off = s
                if l == length:
                    del self._cache_free[i]
                else:
                    self._cache_free[i] = (s + length, l - length)
                return off
        return None

    def _cache_evict_one(self) -> bool:
        if not self._cache_lru:
            return False
        motion_id, _ = self._cache_lru.popitem(last=False)
        seg = self._cache_meta.pop(motion_id, None)
        if seg is None:
            return True
        off, length = seg
        self._cache_frames_used -= int(length)
        self._cache_free_segment_add(int(off), int(length))
        self._cache_offset[motion_id] = -1
        self._cache_len[motion_id] = 0
        return True

    def _cache_motion_to_gpu(self, motion_id: int) -> bool:
        if motion_id in self._cache_meta:
            return True
        length = int(self._motion_num_frames_cpu[motion_id].item())
        if length <= 0:
            return False
        if length > self._cache_max_frames:
            return False

        while (self._cache_frames_used + length) > self._cache_max_frames:
            if not self._cache_evict_one():
                return False

        off = self._cache_alloc_segment(length)
        while off is None:
            if self._cache_capacity_frames < self._cache_max_frames:
                if self._cache_capacity_frames == 0:
                    # Avoid a large one-time allocation spike during warmup.
                    grow_step = 250_000
                    grow_to = min(self._cache_max_frames, max(int(length), grow_step))
                else:
                    # Grow by at least `length`, but avoid doubling to prevent large temporary spikes
                    # from (old buffers + new buffers) during reallocation.
                    grow_step = 250_000
                    grow_to = min(
                        self._cache_max_frames,
                        max(
                            int(self._cache_capacity_frames + length),
                            int(self._cache_capacity_frames + grow_step),
                        ),
                    )
                self._cache_grow_to(grow_to)
            else:
                if not self._cache_evict_one():
                    return False
            off = self._cache_alloc_segment(length)

        start = int(self._motion_start_idx_cpu[motion_id].item())
        end = start + length

        end_off = off + length
        # Direct CPU->GPU copy into cache slices.
        # This avoids allocating a temporary GPU tensor for each field (which can cause peak VRAM spikes).
        self._cache_root_pos[off:end_off].copy_(self._motion_root_pos[start:end], non_blocking=True)
        self._cache_root_rot[off:end_off].copy_(self._motion_root_rot[start:end], non_blocking=True)
        self._cache_root_vel[off:end_off].copy_(self._motion_root_vel[start:end], non_blocking=True)
        self._cache_root_ang_vel[off:end_off].copy_(self._motion_root_ang_vel[start:end], non_blocking=True)
        self._cache_dof_pos[off:end_off].copy_(self._motion_dof_pos[start:end], non_blocking=True)
        self._cache_dof_vel[off:end_off].copy_(self._motion_dof_vel[start:end], non_blocking=True)
        self._cache_local_body_pos[off:end_off].copy_(self._motion_local_body_pos[start:end], non_blocking=True)
        self._cache_root_pos_delta_local[off:end_off].copy_(self._motion_root_pos_delta_local[start:end], non_blocking=True)
        self._cache_root_rot_delta_local[off:end_off].copy_(self._motion_root_rot_delta_local[start:end], non_blocking=True)
        if self._hy_feat_dim > 0:
            hy = self._load_hy_feat_motion_cpu(motion_id, expected_len=length)
            if hy is None:
                raise RuntimeError(f"HY feature missing for motion_id={motion_id} file={self._motion_files[motion_id]!r}")
            self._cache_hy_feat[off:end_off].copy_(hy, non_blocking=True)

        self._cache_meta[motion_id] = (off, length)
        self._cache_lru[motion_id] = None
        self._cache_lru.move_to_end(motion_id, last=True)
        self._cache_frames_used += length
        self._cache_offset[motion_id] = int(off)
        self._cache_len[motion_id] = int(length)
        return True

    def prefetch(self, motion_ids: torch.Tensor) -> None:
        if not self._gpu_cache_enabled:
            return
        if motion_ids.numel() == 0:
            return

        motion_ids_dev = motion_ids.detach().to(self._device) if motion_ids.device != self._device else motion_ids.detach()
        # Fast path: if everything is already cached, avoid any GPU->CPU sync.
        off = self._cache_offset[motion_ids_dev]
        missing = motion_ids_dev[off < 0]
        if missing.numel() == 0:
            return

        # Only synchronize the missing subset.
        missing_list = missing.to("cpu").tolist()
        seen = set()
        uniq_missing = []
        for mid in missing_list:
            if mid in seen:
                continue
            seen.add(mid)
            uniq_missing.append(int(mid))

        for mid in uniq_missing:
            if mid in self._cache_meta:
                self._cache_lru.move_to_end(mid, last=True)
                continue
            self._cache_motion_to_gpu(mid)

    def _gather_frames(self, tensor: torch.Tensor, frame_idx: torch.Tensor) -> torch.Tensor:
        if tensor.device.type == "cuda":
            return tensor[frame_idx]
        frame_idx_cpu = frame_idx if frame_idx.device.type == "cpu" else frame_idx.to("cpu")
        out = tensor[frame_idx_cpu]
        if self._device.type == "cuda":
            out = out.to(self._device, non_blocking=False)
        return out

    @staticmethod
    def _load_motion_npz(path: str):
        with np.load(path, allow_pickle=False) as z:
            return {
                "fps": float(z["fps"]),
                "root_pos": z["root_pos"],
                "root_rot": z["root_rot"],
                "dof_pos": z["dof_pos"],
                "local_body_pos": z["local_body_pos"],
                "link_body_list": z["link_body_list"].tolist(),
            }

    def _load_motion_data(self, path: str):
        if path.endswith(".npz"):
            return self._load_motion_npz(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def _finite_difference(x: torch.Tensor, dt: float) -> torch.Tensor:
        """Compute per-timestep derivative using central differences.

        torch.gradient on CUDA has been observed to segfault in some environments;
        this implementation avoids that code path while preserving similar behavior.
        """
        T = int(x.shape[0])
        if T <= 1:
            return torch.zeros_like(x)
        if T == 2:
            v = (x[1:2] - x[0:1]) / dt
            return torch.cat([v, v], dim=0)

        out = torch.empty_like(x)
        out[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        out[0] = (x[1] - x[0]) / dt
        out[-1] = (x[-1] - x[-2]) / dt
        return out
    
    def _compute_so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations using central differences.
        
        Args:
            rotations: Quaternion rotations with shape (T, 4).
            dt: Time step.
        Returns:
            Angular velocities with shape (T, 3).
        """
        if rotations.shape[0] < 3:
            # For very short sequences, fall back to forward differences
            root_drot = quat_diff(rotations[:-1], rotations[1:])
            omega = quat_to_exp_map(root_drot) / dt
            omega = torch.cat([omega, omega[-1:]], dim=0)  # Repeat last
            return omega
        
        # Use central differences for interior points
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega_interior = quat_to_exp_map(q_rel) / (2.0 * dt)
        
        # Handle boundaries with forward/backward differences
        q_start_rel = quat_mul(rotations[1], quat_conjugate(rotations[0]))
        omega_start = quat_to_exp_map(q_start_rel) / dt
        
        q_end_rel = quat_mul(rotations[-1], quat_conjugate(rotations[-2]))
        omega_end = quat_to_exp_map(q_end_rel) / dt
        
        # Combine all parts
        omega = torch.cat([omega_start.unsqueeze(0), omega_interior, omega_end.unsqueeze(0)], dim=0)
        return omega
    
    def get_motion_length(self, motion_ids):
        return self._motion_lengths[motion_ids]
        
    def num_motions(self):
        return self._motion_weights.shape[0]
    
    def get_total_length(self):
        return torch.sum(self._motion_lengths).item()
    
    def sample_motions(self, n, motion_difficulty=None, max_key_body_error=None, 
                      use_error_aware_sampling=False, error_sampling_power=5.0, 
                      error_sampling_threshold=0.15):
        if motion_difficulty is not None:
            if use_error_aware_sampling and max_key_body_error is not None:
                # Apply error aware sampling formula
                error_aware_prob = torch.ones_like(motion_difficulty)
                
                # Apply error aware probability only when motion_difficulty == 1
                difficulty_one_mask = (motion_difficulty == 1.0)
                if difficulty_one_mask.any():
                    normalized_error = torch.clamp(max_key_body_error / error_sampling_threshold, max=1.0)
                    error_prob = normalized_error ** error_sampling_power
                    error_aware_prob[difficulty_one_mask] = error_prob[difficulty_one_mask]
                
                # For motion_difficulty > 1, use original difficulty
                difficulty_gt_one_mask = (motion_difficulty > 1.0)
                error_aware_prob[difficulty_gt_one_mask] = motion_difficulty[difficulty_gt_one_mask]
                
                motion_prob = self._motion_weights * error_aware_prob
            else:
                motion_prob = self._motion_weights * motion_difficulty
        else:
            motion_prob = self._motion_weights
        
        motion_ids = torch.multinomial(motion_prob, num_samples=n, replacement=True)
        return motion_ids
    
    def sample_time(self, motion_ids):
        phase = torch.rand(motion_ids.shape, device=self._device)
        motion_len = self._motion_lengths[motion_ids]
        
        motion_time = motion_len * phase
        return motion_time
                
    def _fetch_motion_files(self, motion_file: str):
        if motion_file.endswith(".yaml"):
            motion_files = []
            motion_weights = []
            hy_feat_files: List[str] = []
            hy_feat_dim = 0
            hy_feat_t = 1.0
            with open(motion_file, "r") as f:
                motion_config = yaml.load(f, Loader=yaml.SafeLoader)
            
            motion_root_path = os.path.expandvars(os.path.expanduser(str(motion_config["root_path"])))
            hy_feat_root = motion_config.get("hy_feat_root", None)
            if hy_feat_root is not None:
                hy_feat_root = os.path.expandvars(os.path.expanduser(str(hy_feat_root)))
            hy_feat_t = float(motion_config.get("hy_feat_t", 1.0))
            hy_feat_dim = int(motion_config.get("hy_feat_dim", 1280 if hy_feat_root is not None else 0))
            if hy_feat_root is None:
                hy_feat_dim = 0

            if bool(motion_config.get("auto_discover", False)):
                pattern = str(motion_config.get("auto_discover_glob", "**/motion.pkl"))
                weight = float(motion_config.get("auto_discover_weight", 1.0))
                base = motion_root_path

                base_path = Path(base)

                def _iter_auto_files():
                    # Fast deterministic scan for the common HYMotion layout: <id>/<seed>/motion.pkl.
                    if pattern == "**/motion.pkl":
                        try:
                            first_level = [e for e in os.scandir(base_path) if e.is_dir()]
                        except FileNotFoundError:
                            return
                        for e1 in sorted(first_level, key=lambda e: e.name):
                            try:
                                second_level = [e for e in os.scandir(e1.path) if e.is_dir()]
                            except FileNotFoundError:
                                continue
                            for e2 in sorted(second_level, key=lambda e: e.name):
                                p = Path(e2.path) / "motion.pkl"
                                if p.is_file():
                                    yield p
                    else:
                        yield from base_path.glob(pattern)

                # Avoid enumerating the full dataset when only a small deterministic prefix is needed.
                early_limit = None
                if (pattern == "**/motion.pkl") and (self._max_motions > 0) and (not self._motion_ids_spec.strip()) and (not self._shuffle_motions):
                    early_limit = self._max_motions

                if early_limit is not None:
                    files = [str(p) for p in islice(_iter_auto_files(), early_limit)]
                else:
                    files = sorted(str(p) for p in _iter_auto_files())

                motion_list = [{"file": os.path.relpath(p, base), "weight": weight, "description": "auto"} for p in files]
            else:
                motion_list = motion_config["motions"]

            # Optional subset selection for faster visualization/debug.
            if self._motion_ids_spec.strip():
                indices = self._parse_index_spec(self._motion_ids_spec, len(motion_list))
                motion_list = [motion_list[i] for i in indices]
            elif self._shuffle_motions:
                rng = np.random.RandomState(self._shuffle_seed)
                order = rng.permutation(len(motion_list)).tolist()
                motion_list = [motion_list[i] for i in order]

            if self._max_motions > 0:
                motion_list = motion_list[: self._max_motions]

            # DDP: shard motion list across ranks to avoid each process loading the full dataset.
            # Sharding happens after motion_ids/shuffle/max_motions so every rank applies the same
            # deterministic preprocessing before splitting.
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    if world_size > 1:
                        rank = dist.get_rank()

                        # In DDP, random subsampling during loading makes each rank load a different
                        # unpredictable subset. Force full loading per-rank for reproducibility.
                        if self._sample_ratio != 1.0:
                            if rank == 0:
                                print(
                                    f"[MotionLib] Warning: sample_ratio={self._sample_ratio} with DDP sharding; "
                                    "forcing sample_ratio=1.0 to keep per-rank motion sets deterministic."
                                )
                            self._sample_ratio = 1.0

                        # Ensure each rank gets at least one motion even when world_size > num_motions.
                        if len(motion_list) < world_size:
                            repeats = (world_size + len(motion_list) - 1) // len(motion_list)
                            motion_list = (motion_list * repeats)[:world_size]

                        # Strided sharding (rank, rank+W, ...).
                        motion_list = motion_list[rank::world_size]
            except Exception:
                pass

            if len(motion_list) == 0:
                raise ValueError(
                    f"No motions selected from YAML. motion_file={motion_file}, "
                    f"motion_ids='{self._motion_ids_spec}', max_motions={self._max_motions}"
                )

            for motion_entry in motion_list:
                curr_file = os.path.join(motion_root_path, motion_entry['file'])
                if curr_file.endswith(".pkl"):
                    npz_file = curr_file[:-4] + ".npz"
                    if os.path.exists(npz_file):
                        curr_file = npz_file
                curr_weight = motion_entry['weight']
                assert(curr_weight >= 0)

                motion_weights.append(curr_weight)
                motion_files.append(curr_file)
                if hy_feat_root is not None:
                    rel = Path(curr_file).relative_to(Path(motion_root_path))
                    parts = rel.parts
                    if len(parts) < 3:
                        raise ValueError(f"Cannot derive HY feature path from motion file: {curr_file}")
                    sample_id = parts[0]
                    seed = parts[1]
                    hy_feat_files.append(str(Path(hy_feat_root) / sample_id / seed / "single_stream_feat.npz"))
                else:
                    hy_feat_files.append("")
        else:
            curr_file = motion_file
            if curr_file.endswith(".pkl"):
                npz_file = curr_file[:-4] + ".npz"
                if os.path.exists(npz_file):
                    curr_file = npz_file
            motion_files = [curr_file]
            motion_weights = [1.0]
            hy_feat_files = [""]
            hy_feat_dim = 0
            hy_feat_t = 1.0
        
        return motion_files, motion_weights, hy_feat_files, hy_feat_dim, hy_feat_t

    def _load_hy_feat_motion_cpu(self, motion_id: int, *, expected_len: int) -> Optional[torch.Tensor]:
        if self._hy_feat_dim <= 0:
            return None
        mid = int(motion_id)
        if mid in self._hy_feat_cache_cpu:
            self._hy_feat_cache_cpu.move_to_end(mid, last=True)
            return self._hy_feat_cache_cpu[mid]

        path = str(self._hy_feat_files[mid])
        if not path or not os.path.exists(path):
            return None

        with np.load(path, allow_pickle=False) as z:
            t_arr = z["t"].astype(np.float32)
            feat = z["feat"]
        if feat.ndim != 3:
            raise ValueError(f"Expected feat.ndim==3 in {path}, got {feat.shape}")

        # Select t=1.0 stream by default.
        t_target = float(self._hy_feat_t)
        idx = int(np.argmin(np.abs(t_arr - t_target)))
        feat_sel = feat[idx]
        if feat_sel.shape[-1] != int(self._hy_feat_dim):
            raise ValueError(f"Unexpected HY feat dim in {path}: {feat_sel.shape} (expected last dim {self._hy_feat_dim})")
        if feat_sel.shape[0] != int(expected_len):
            raise ValueError(
                f"HY feat length mismatch for motion_id={mid}: feat_T={feat_sel.shape[0]} expected={expected_len} file={path}"
            )

        out = torch.as_tensor(feat_sel, device="cpu", dtype=torch.float16)
        if self._hy_feat_cache_motions > 0:
            self._hy_feat_cache_cpu[mid] = out
            self._hy_feat_cache_cpu.move_to_end(mid, last=True)
            while len(self._hy_feat_cache_cpu) > int(self._hy_feat_cache_motions):
                self._hy_feat_cache_cpu.popitem(last=False)
        return out

    def calc_hy_feat_frame(self, motion_ids: torch.Tensor, motion_times: torch.Tensor) -> torch.Tensor:
        """Return HY-Motion single-stream feature at t≈hy_feat_t, interpolated to motion_times.

        The feature files are expected to be aligned to the same per-motion frame grid as the motion data:
          single_stream_feat.npz: feat[K, T, D] where T == num_frames of the motion.
        """
        if self._hy_feat_dim <= 0:
            raise RuntimeError("HY features are not enabled for this MotionLib instance (hy_feat_dim==0).")
        motion_ids = motion_ids.to(self._device)
        motion_times = motion_times.to(self._device)

        motion_loop_num = torch.floor(motion_times / self._motion_lengths[motion_ids])
        motion_times = motion_times - motion_loop_num * self._motion_lengths[motion_ids]

        _fi0, _fi1, fi0_local, fi1_local, blend = self._calc_frame_blend(motion_ids, motion_times)

        n = int(motion_ids.shape[0])
        D = int(self._hy_feat_dim)

        feat0 = torch.empty((n, D), device=self._device, dtype=torch.float16)
        feat1 = torch.empty((n, D), device=self._device, dtype=torch.float16)

        use_cache = self._gpu_cache_enabled and (self._cache_hy_feat is not None)
        if use_cache:
            cache_off = self._cache_offset[motion_ids].to(torch.int64)
            cached_mask = cache_off >= 0
            if not bool(cached_mask.all()):
                self.prefetch(motion_ids[cached_mask.logical_not()])
                cache_off = self._cache_offset[motion_ids].to(torch.int64)
                cached_mask = cache_off >= 0

            if bool(cached_mask.any()):
                idx = cached_mask.nonzero(as_tuple=False).flatten()
                cache_off_c = cache_off[idx]
                cache_idx0 = cache_off_c + fi0_local[idx].to(torch.int64)
                cache_idx1 = cache_off_c + fi1_local[idx].to(torch.int64)
                feat0[idx] = self._cache_hy_feat[cache_idx0]
                feat1[idx] = self._cache_hy_feat[cache_idx1]

            if bool(cached_mask.logical_not().any()):
                # Fallback: load from disk + per-motion cache on demand.
                idx = cached_mask.logical_not().nonzero(as_tuple=False).flatten()
                for mid in motion_ids[idx].unique().tolist():
                    mid = int(mid)
                    mask = (motion_ids[idx] == mid).nonzero(as_tuple=False).flatten()
                    ii = idx[mask]
                    seq = self._load_hy_feat_motion_cpu(mid, expected_len=int(self._motion_num_frames_cpu[mid]))
                    if seq is None:
                        raise RuntimeError(f"HY feature missing for motion_id={mid} file={self._motion_files[mid]!r}")
                    seq_dev = seq.to(self._device, dtype=torch.float16, non_blocking=False) if self._device.type == "cuda" else seq
                    feat0[ii] = seq_dev[fi0_local[ii]]
                    feat1[ii] = seq_dev[fi1_local[ii]]
        else:
            # No GPU cache: group by motion_id and use per-motion CPU LRU cache.
            for mid in motion_ids.unique().tolist():
                mid = int(mid)
                idx = (motion_ids == mid).nonzero(as_tuple=False).flatten()
                seq = self._load_hy_feat_motion_cpu(mid, expected_len=int(self._motion_num_frames_cpu[mid]))
                if seq is None:
                    raise RuntimeError(f"HY feature missing for motion_id={mid} file={self._motion_files[mid]!r}")
                seq_dev = seq.to(self._device, dtype=torch.float16, non_blocking=False) if self._device.type == "cuda" else seq
                feat0[idx] = seq_dev[fi0_local[idx]]
                feat1[idx] = seq_dev[fi1_local[idx]]

        # Interpolate in float32 for stability.
        blend_f = blend.to(dtype=torch.float32).unsqueeze(-1)
        out = (1.0 - blend_f) * feat0.to(dtype=torch.float32) + blend_f * feat1.to(dtype=torch.float32)
        return out

    @staticmethod
    def _parse_index_spec(spec: str, n: int) -> List[int]:
        spec = (spec or "").strip()
        if not spec:
            return list(range(n))

        indices: List[int] = []
        parts = [p.strip() for p in spec.replace(" ", "").split(",") if p.strip()]
        for part in parts:
            if "-" in part:
                a_s, b_s = part.split("-", 1)
                if a_s == "" or b_s == "":
                    raise ValueError(f"Invalid range token '{part}' in motion_ids='{spec}'")
                a = int(a_s)
                b = int(b_s)
                if a < 0:
                    a = n + a
                if b < 0:
                    b = n + b
                if a > b:
                    a, b = b, a
                for i in range(a, b + 1):
                    if i < 0 or i >= n:
                        raise IndexError(f"motion_ids index {i} out of range [0, {n-1}]")
                    indices.append(i)
            else:
                i = int(part)
                if i < 0:
                    i = n + i
                if i < 0 or i >= n:
                    raise IndexError(f"motion_ids index {i} out of range [0, {n-1}]")
                indices.append(i)

        # Deduplicate while preserving order
        seen = set()
        out: List[int] = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                out.append(i)
        return out
    
    def _calc_frame_blend(self, motion_ids, times):
        num_frames = self._motion_num_frames[motion_ids]
        
        phase = times / self._motion_lengths[motion_ids]
        phase = torch.clip(phase, 0.0, 1.0)
        
        frame_idx0_local = (phase * (num_frames - 1)).long()
        frame_idx1_local = torch.min(frame_idx0_local + 1, num_frames - 1)
        blend = phase * (num_frames - 1) - frame_idx0_local.float()
        
        frame_start_idx = self._motion_start_idx[motion_ids]
        frame_idx0 = frame_idx0_local + frame_start_idx
        frame_idx1 = frame_idx1_local + frame_start_idx
        
        return frame_idx0, frame_idx1, frame_idx0_local, frame_idx1_local, blend
        
    def calc_motion_frame(self, motion_ids, motion_times):
        motion_ids = motion_ids.to(self._device)
        motion_times = motion_times.to(self._device)

        motion_loop_num = torch.floor(motion_times / self._motion_lengths[motion_ids])
        motion_times -= motion_loop_num * self._motion_lengths[motion_ids]

        frame_idx0, frame_idx1, frame_idx0_local, frame_idx1_local, blend = self._calc_frame_blend(motion_ids, motion_times)

        use_cache = self._gpu_cache_enabled
        cache_off = None
        cached_mask = None
        if use_cache:
            cache_off = self._cache_offset[motion_ids].to(torch.int64)
            cached_mask = cache_off >= 0
            if not bool(cached_mask.all()):
                # Only synchronize if there are true misses.
                self.prefetch(motion_ids[cached_mask.logical_not()])
                cache_off = self._cache_offset[motion_ids].to(torch.int64)
                cached_mask = cache_off >= 0

        # Allocate outputs once and fill from cache/CPU as available (avoid all-or-nothing fallback).
        n = int(motion_ids.shape[0])
        D = int(self._motion_dof_pos.shape[-1])
        B = int(self._motion_local_body_pos.shape[1])

        root_pos0 = torch.empty((n, 3), device=self._device, dtype=torch.float32)
        root_pos1 = torch.empty((n, 3), device=self._device, dtype=torch.float32)
        root_rot0 = torch.empty((n, 4), device=self._device, dtype=torch.float32)
        root_rot1 = torch.empty((n, 4), device=self._device, dtype=torch.float32)
        root_vel = torch.empty((n, 3), device=self._device, dtype=torch.float32)
        root_ang_vel = torch.empty((n, 3), device=self._device, dtype=torch.float32)
        dof_pos0 = torch.empty((n, D), device=self._device, dtype=torch.float32)
        dof_pos1 = torch.empty((n, D), device=self._device, dtype=torch.float32)
        local_key_body_pos0 = torch.empty((n, B, 3), device=self._device, dtype=torch.float32)
        local_key_body_pos1 = torch.empty((n, B, 3), device=self._device, dtype=torch.float32)
        dof_vel = torch.empty((n, D), device=self._device, dtype=torch.float32)

        if use_cache and bool(cached_mask.any()):
            idx = cached_mask.nonzero(as_tuple=False).flatten()
            cache_off_c = cache_off[idx]
            cache_idx0 = cache_off_c + frame_idx0_local[idx].to(torch.int64)
            cache_idx1 = cache_off_c + frame_idx1_local[idx].to(torch.int64)

            root_pos0[idx] = self._cache_root_pos[cache_idx0]
            root_pos1[idx] = self._cache_root_pos[cache_idx1]
            root_rot0[idx] = self._cache_root_rot[cache_idx0]
            root_rot1[idx] = self._cache_root_rot[cache_idx1]
            root_vel[idx] = self._cache_root_vel[cache_idx0]
            root_ang_vel[idx] = self._cache_root_ang_vel[cache_idx0]
            dof_pos0[idx] = self._cache_dof_pos[cache_idx0]
            dof_pos1[idx] = self._cache_dof_pos[cache_idx1]
            local_key_body_pos0[idx] = self._cache_local_body_pos[cache_idx0]
            local_key_body_pos1[idx] = self._cache_local_body_pos[cache_idx1]
            dof_vel[idx] = self._cache_dof_vel[cache_idx0]

        if (not use_cache) or bool(cached_mask.logical_not().any()):
            idx = torch.arange(n, device=self._device) if (not use_cache) else cached_mask.logical_not().nonzero(as_tuple=False).flatten()
            fi0 = frame_idx0[idx]
            fi1 = frame_idx1[idx]

            root_pos0[idx] = self._gather_frames(self._motion_root_pos, fi0)
            root_pos1[idx] = self._gather_frames(self._motion_root_pos, fi1)
            root_rot0[idx] = self._gather_frames(self._motion_root_rot, fi0)
            root_rot1[idx] = self._gather_frames(self._motion_root_rot, fi1)
            root_vel[idx] = self._gather_frames(self._motion_root_vel, fi0)
            root_ang_vel[idx] = self._gather_frames(self._motion_root_ang_vel, fi0)
            dof_pos0[idx] = self._gather_frames(self._motion_dof_pos, fi0)
            dof_pos1[idx] = self._gather_frames(self._motion_dof_pos, fi1)
            local_key_body_pos0[idx] = self._gather_frames(self._motion_local_body_pos, fi0)
            local_key_body_pos1[idx] = self._gather_frames(self._motion_local_body_pos, fi1)
            dof_vel[idx] = self._gather_frames(self._motion_dof_vel, fi0)
        
        blend_unsqueeze = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsqueeze) * root_pos0 + blend_unsqueeze * root_pos1
        root_pos += motion_loop_num.unsqueeze(-1) * self._motion_root_pos_delta[motion_ids]
        root_rot = slerp(root_rot0, root_rot1, blend)
        
        dof_pos = (1.0 - blend_unsqueeze) * dof_pos0 + blend_unsqueeze * dof_pos1
        
        local_key_body_pos = (1.0 - blend_unsqueeze.unsqueeze(1)) * local_key_body_pos0 + blend_unsqueeze.unsqueeze(1) * local_key_body_pos1
        
        # compute the root pos delta compared to last frame
        root_pos_delta_local0 = torch.empty((n, 3), device=self._device, dtype=torch.float32)
        root_pos_delta_local1 = torch.empty((n, 3), device=self._device, dtype=torch.float32)
        if use_cache and bool(cached_mask.any()):
            idx = cached_mask.nonzero(as_tuple=False).flatten()
            cache_off_c = cache_off[idx]
            cache_idx0 = cache_off_c + frame_idx0_local[idx].to(torch.int64)
            cache_idx1 = cache_off_c + frame_idx1_local[idx].to(torch.int64)
            root_pos_delta_local0[idx] = self._cache_root_pos_delta_local[cache_idx0]
            root_pos_delta_local1[idx] = self._cache_root_pos_delta_local[cache_idx1]
        if (not use_cache) or bool(cached_mask.logical_not().any()):
            idx = torch.arange(n, device=self._device) if (not use_cache) else cached_mask.logical_not().nonzero(as_tuple=False).flatten()
            root_pos_delta_local0[idx] = self._gather_frames(self._motion_root_pos_delta_local, frame_idx0[idx])
            root_pos_delta_local1[idx] = self._gather_frames(self._motion_root_pos_delta_local, frame_idx1[idx])
        root_pos_delta_local = (1.0 - blend_unsqueeze) * root_pos_delta_local0 + blend_unsqueeze * root_pos_delta_local1

        # compute the root rot delta compared to last frame 
        root_rot_delta_local0 = torch.empty((n, 3), device=self._device, dtype=torch.float32)
        root_rot_delta_local1 = torch.empty((n, 3), device=self._device, dtype=torch.float32)
        if use_cache and bool(cached_mask.any()):
            idx = cached_mask.nonzero(as_tuple=False).flatten()
            cache_off_c = cache_off[idx]
            cache_idx0 = cache_off_c + frame_idx0_local[idx].to(torch.int64)
            cache_idx1 = cache_off_c + frame_idx1_local[idx].to(torch.int64)
            root_rot_delta_local0[idx] = self._cache_root_rot_delta_local[cache_idx0]
            root_rot_delta_local1[idx] = self._cache_root_rot_delta_local[cache_idx1]
        if (not use_cache) or bool(cached_mask.logical_not().any()):
            idx = torch.arange(n, device=self._device) if (not use_cache) else cached_mask.logical_not().nonzero(as_tuple=False).flatten()
            root_rot_delta_local0[idx] = self._gather_frames(self._motion_root_rot_delta_local, frame_idx0[idx])
            root_rot_delta_local1[idx] = self._gather_frames(self._motion_root_rot_delta_local, frame_idx1[idx])
        # we use linear interpolation for root rot delta, as it is euler angle
        root_rot_delta_local = (1.0 - blend_unsqueeze) * root_rot_delta_local0 + blend_unsqueeze * root_rot_delta_local1

        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, root_pos_delta_local, root_rot_delta_local
    
    def get_key_body_idx(self, key_body_names):
        key_body_idx = []
        for key_body_name in key_body_names:
            key_body_idx.append(self._body_link_list.index(key_body_name))
        return key_body_idx # list
    
    def get_motion_names(self):
        return self._motion_names
        
