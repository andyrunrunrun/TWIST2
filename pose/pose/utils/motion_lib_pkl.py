import os, pickle, struct, warnings, yaml
from pathlib import Path
from itertools import islice
import torch
from pose.utils.torch_utils import quat_diff, quat_to_exp_map, slerp, euler_from_quaternion
from tqdm import tqdm
from pose.utils.isaacgym_torch_utils import quat_rotate_inverse, quat_mul, quat_conjugate
import numpy as np
import sys
from types import ModuleType
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
# Patch sys.modules to fake missing modules from numpy 2.x
class FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)

# Patch potentially missing modules
sys.modules['numpy._core'] = FakeModule('numpy._core', np.core if hasattr(np, 'core') else np)
sys.modules['numpy._core.multiarray'] = FakeModule('numpy._core.multiarray', getattr(np.core, 'multiarray', None))
class FakeModule(ModuleType):
    def __init__(self, name, real=None):
        super().__init__(name)
        if real:
            self.__dict__.update(real.__dict__)
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
                 # Memory optimization options
                 lazy_load: bool = False,  # If True, only load metadata at startup; load motion data on-demand
                 cpu_cache_gib: float = 50.0, # CPU LRU cache budget in GiB when lazy_load=True; 0 disables
                 storage_dtype: str = "float32",  # Storage precision: "float32" or "float16" (halves CPU memory)
                 skip_ddp_sharding: bool = False,  # If True, skip DDP sharding to allow sampling from full dataset
                 ):
        self._device = torch.device(device)
        self._store_on_cpu = bool(store_on_cpu)
        self._storage_device = torch.device("cpu") if self._store_on_cpu else self._device
        self._gpu_cache_gib = float(gpu_cache_gib)

        # Memory optimization settings
        self._lazy_load = bool(lazy_load)
        self._cpu_cache_gib = float(cpu_cache_gib)
        self._storage_dtype_str = storage_dtype
        if storage_dtype == "float16":
            self._storage_dtype = torch.float16
        else:
            self._storage_dtype = torch.float32

        # DDP sharding control (for resample mode)
        self._skip_ddp_sharding = bool(skip_ddp_sharding)

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
        
        # CPU LRU cache for lazy loading (motion_id -> tuple of tensors)
        self._cpu_motion_cache: "OrderedDict[int, dict]" = OrderedDict()
        self._cpu_cache_bytes_used = 0
        self._cpu_cache_max_bytes = int(cpu_cache_gib * (1024 ** 3))

        # Periodic resample mode: only use a subset of motions
        self._resample_mode = False  # Whether resample mode is enabled
        self._loaded_subset_ids: set = set()  # Current subset of motion IDs
        self._loaded_subset_ids_tensor = None  # Cached GPU tensor of subset IDs (for fast sampling)
        self._full_motion_ids = []  # All available motion IDs (for resampling)

        # Async resample: prepare next subset in background on CPU
        self._async_resample_enabled = False
        self._async_resample_interval = 0
        self._async_resample_thread = None
        self._async_resample_stop_event = None
        self._async_resample_ready_event = None
        self._async_resample_next_data = None  # Pre-loaded data on CPU (ready to load to GPU)
        self._async_resample_next_ids = None    # Next subset IDs (for sampling)
        self._async_resample_last_iteration = 0
        self._async_resample_lock = None  # Lock for thread-safe access

        # Resample mode: direct GPU storage (no cache, simple and direct)
        # Stores motion data directly on GPU, bypassing all cache layers
        self._resample_gpu_storage: dict = {}  # {motion_id: {root_pos, root_rot, ...}} on GPU

        # Resample mode: merged GPU tensors (optimized for speed)
        self._gpu_root_pos = None
        self._gpu_root_rot = None
        self._gpu_root_vel = None
        self._gpu_root_ang_vel = None
        self._gpu_dof_pos = None
        self._gpu_dof_vel = None
        self._gpu_local_body_pos = None
        self._gpu_root_pos_delta_local = None
        self._gpu_root_rot_delta_local = None
        self._gpu_root_pos_delta = None  # Per-motion delta
        self._motion_id_to_frame = {}  # {motion_id: (start_frame, num_frames)}
        self._motion_id_to_idx = {}  # {motion_id: index_in_subset}
        self._motion_start_frame = None  # GPU tensor: motion_id -> start_frame
        self._motion_id_to_idx_tensor = None  # GPU tensor: motion_id -> index
        self._motion_start_idx_by_id = None  # GPU tensor: motion_id -> start_frame (resample mode)
        self._resample_D = 0
        self._resample_B = 0
        self._resample_total_motions = 0
        self._resample_total_frames = 0
        self._motion_num_frames_resample = None  # Resample mode: num_frames per motion
        self._motion_lengths_resample = None  # Resample mode: length per motion

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
        hy_feat_files = [str(p) for p in hy_feat_files]
        self._hy_feat_dim = int(hy_feat_dim)
        self._hy_feat_t = float(hy_feat_t)
        self._hy_feat_files = []
        num_motion_files = len(motion_files)
        
        num_sub_motions_total = 0

        if self._hy_feat_dim > 0 and self._motion_decompose:
            raise ValueError("HY features are not supported with motion_decompose=True. Please disable motion_decompose.")
            
        for i in tqdm(range(num_motion_files), desc="[MotionLib] Loading motions"):
            if torch.rand(1) > self._sample_ratio and num_motion_files > 1:
                continue
            
            curr_file = motion_files[i]
            curr_weight = motion_weights[i]
            if not os.path.exists(curr_file):
                print(f"Motion file {curr_file} does not exist")
                continue

            try:
                if self._lazy_load:
                    # In lazy load mode, we only peek at metadata if possible, or load & discard.
                    # Since .pkl / .npz structure varies, we do a full load but NOT store the big result tensors.
                    # We just need: fps, num_frames.
                    # Optimization: For .npz, we could just read 'fps' and 'root_pos' shape.
                    # For now, to be safe and simple: load, get meta, discard data.
                    # This incurs startup I/O but saves RAM.
                    # (Ideally we'd have a separate metadata file, but we don't control dataset generation here).
                    motion_data = self._load_motion_data(curr_file)
                    if motion_data is None:
                        continue
                    fps = float(motion_data["fps"])
                    # root_pos is (T, 3)
                    if isinstance(motion_data["root_pos"], torch.Tensor):
                         num_frames = motion_data["root_pos"].shape[0]
                    else:
                         num_frames = motion_data["root_pos"].shape[0]
                    
                    # Compute derived metadata
                    motion_len_s = 1.0 / fps * (num_frames - 1)
                    
                    # Store minimal metadata
                    self._motion_files.append(curr_file)
                    self._motion_fps.append(fps)
                    self._motion_dt.append(1.0 / fps)
                    self._motion_num_frames.append(num_frames)
                    self._motion_lengths.append(motion_len_s)
                    self._motion_weights.append(curr_weight)
                    self._motion_names.append(os.path.basename(curr_file))

                    # Extract link_body_list from first motion (needed for key_body_idx lookup)
                    if self._body_link_list is None or len(self._body_link_list) == 0:
                        self._body_link_list = motion_data["link_body_list"]

                    # Append empty/dummy entries to other lists to keep indices aligned
                    # (These will be populated on-demand in cache, or remain empty if unused)
                    self._motion_root_pos_delta.append(None)
                    self._motion_root_pos.append(None)
                    self._motion_root_rot.append(None)
                    self._motion_root_vel.append(None)
                    self._motion_root_ang_vel.append(None)
                    self._motion_dof_pos.append(None)
                    self._motion_root_pos_delta_local.append(None)
                    self._motion_root_rot_delta_local.append(None)
                    self._motion_dof_vel.append(None)
                    self._motion_local_body_pos.append(None)

                    # Explicitly delete heavy data to free memory immediately
                    del motion_data
                    
                else:
                    motion_data = self._load_motion_data(curr_file)
                    if motion_data is None:
                        continue
                    fps = motion_data["fps"]
                    
                    # ... [Standard loading logic] ...
                    # Create tensors on CPU first then move to target device.
                    root_pos = self._to_storage_tensor(motion_data["root_pos"], dtype=torch.float)
                    root_rot = self._to_storage_tensor(motion_data["root_rot"], dtype=torch.float)
                    dof_pos = self._to_storage_tensor(motion_data["dof_pos"], dtype=torch.float)
                    local_body_pos = self._to_storage_tensor(motion_data["local_body_pos"], dtype=torch.float)
                    if self._body_link_list is None or len(self._body_link_list) == 0:
                        self._body_link_list = motion_data["link_body_list"]
                    
                     # ... [Height adjustment] ...
                    if self._motion_height_adjust:
                        # compute the lowest body part in reference motion
                        body_pos = local_body_pos + root_pos.unsqueeze(1)
                        lowest_body_part = torch.min(body_pos[..., 2])
                        # adjust the height of the root position
                        root_pos[..., 2] -= lowest_body_part

                    self._add_motions(root_pos, root_rot, dof_pos, local_body_pos, fps, curr_weight, curr_file)

            except Exception as e:
                # NumPy 2.x pickles are not compatible with NumPy 1.x (e.g. py38 IsaacGym env).
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

            # Keep HY feature file list aligned with successfully loaded motions (some motion files may be skipped).
            if self._hy_feat_dim > 0:
                self._hy_feat_files.append(hy_feat_files[i])
            else:
                self._hy_feat_files.append("")
            
            # TODO: Motion decomposition is not supported in lazy_load mode yet
            if self._motion_decompose and not self._lazy_load:
                # [Existing decomposition logic preserved for non-lazy mode]
                pass # Already handled inside _add_motions call above or we skip it for now in lazy mode?
                # Actually _add_motions is NOT called in lazy mode above.
                # If motion_decompose is True and lazy_load is True, we have a conflict.
                # We should probably warn or disable decomposition for lazy load, or implement it later.
                # For now, assumme lazy_load + motion_decompose is not supported/tested.
                pass 
                
        print(f"Total number of sub-motions: {num_sub_motions_total}")
                        
        assert len(self._motion_weights) == len(self._motion_names), f"len(self._motion_weights) = {len(self._motion_weights)}, len(self._motion_names) = {len(self._motion_names)}"
        
        if len(self._motion_weights) == 0:
             raise RuntimeError(f"No valid motions loaded from {motion_file}.")
        
        self._motion_weights = torch.tensor(self._motion_weights, dtype=torch.float, device=self._device)
        self._motion_weights /= torch.sum(self._motion_weights)
        
        self._motion_fps = torch.tensor(self._motion_fps, dtype=torch.float, device=self._device)
        self._motion_dt = torch.tensor(self._motion_dt, dtype=torch.float, device=self._device)
        self._motion_num_frames = torch.tensor(self._motion_num_frames, dtype=torch.long, device=self._device)
        self._motion_lengths = torch.tensor(self._motion_lengths, dtype=torch.float, device=self._device)

        if not self._lazy_load:
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
        else:
            # In lazy load mode, we don't have concatenated tensors.
            # We initialize dummy start_idx to satisfy potential downstream checks (though we won't use them for gathering)
            self._motion_start_idx = torch.zeros(len(self._motion_weights), dtype=torch.long, device=self._device)
            self._motion_start_idx_cpu = self._motion_start_idx.to("cpu")
            # But we must ensure _motion_root_pos etc are indexable lists (already populated with Nones)

        self._motion_num_frames_cpu = self._motion_num_frames.to("cpu")

        num_motions = self.num_motions()
        self._motion_ids = torch.arange(num_motions, dtype=torch.long, device=self._device)
        # Store full list of motion IDs for resample mode
        self._full_motion_ids = list(range(num_motions))

        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

    def _ensure_motion_loaded(self, motion_id: int):
        """Ensure a single motion is loaded into CPU cache."""
        if not self._lazy_load:
            return
            
        if motion_id in self._cpu_motion_cache:
            self._cpu_motion_cache.move_to_end(motion_id, last=True)
            return

        # Load motion
        data = self._load_motion_on_demand(motion_id)
        
        # Check size estimate (sum of bytes of all tensors)
        size_bytes = 0
        for t in data.values():
             if isinstance(t, torch.Tensor):
                 size_bytes += t.element_size() * t.numel()
        
        # Evict if needed
        while (self._cpu_cache_bytes_used + size_bytes > self._cpu_cache_max_bytes) and self._cpu_motion_cache:
            evict_id, evict_data = self._cpu_motion_cache.popitem(last=False)
            evict_size = 0
            for t in evict_data.values():
                 if isinstance(t, torch.Tensor):
                     evict_size += t.element_size() * t.numel()
            self._cpu_cache_bytes_used -= evict_size
            # Also clear from lists to be safe (though lists contain None in lazy mode)
            # self._motion_root_pos[evict_id] = None ... (optional, as we don't use the lists for lookup in lazy mode)

        self._cpu_motion_cache[motion_id] = data
        self._cpu_cache_bytes_used += size_bytes
        
    def _load_motion_on_demand(self, motion_id: int) -> dict:
        """Load a single motion from disk and compute derived features."""
        file_path = self._motion_files[motion_id]
        motion_data = self._load_motion_data(file_path)
        if motion_data is None:
            raise RuntimeError(f"Failed to load motion {file_path}")
            
        fps = self._motion_fps[motion_id].item()
        dt = 1.0 / fps
        
        # Convert to storage tensors
        root_pos = self._to_storage_tensor(motion_data["root_pos"], dtype=torch.float)
        root_rot = self._to_storage_tensor(motion_data["root_rot"], dtype=torch.float)
        dof_pos = self._to_storage_tensor(motion_data["dof_pos"], dtype=torch.float)
        local_body_pos = self._to_storage_tensor(motion_data["local_body_pos"], dtype=torch.float)
        
        if self._motion_height_adjust:
             body_pos = local_body_pos + root_pos.unsqueeze(1)
             lowest_body_part = torch.min(body_pos[..., 2])
             root_pos[..., 2] -= lowest_body_part

        # Compute derived
        root_pos_delta = root_pos[-1] - root_pos[0]
        root_pos_delta[..., -1] = 0.0
        
        root_vel = self._finite_difference(root_pos, dt)
        
        root_pos_delta_local = torch.zeros_like(root_pos)
        root_pos_delta_local[1:, :] = root_pos[1:, :] - root_pos[:-1, :]
        root_pos_delta_local[0, :] = 0.0
        root_pos_delta_local[1:, :] = quat_rotate_inverse(root_rot[:-1, :], root_pos_delta_local[1:, :])
        
        root_rot_delta_local = torch.zeros_like(root_pos)
        root_rot_delta_local[1:, :] = euler_from_quaternion(quat_diff(root_rot[1:, :], root_rot[:-1, :]))
        root_rot_delta_local[0, :] = 0.0
        root_rot_delta_local[1:, :] = quat_rotate_inverse(root_rot[:-1, :], root_rot_delta_local[1:, :])

        root_ang_vel = self._compute_so3_derivative(root_rot, dt)
        dof_vel = self._finite_difference(dof_pos, dt)
        
        # Cast to compact storage dtype if needed
        def cast(t): 
            return self._to_storage_tensor(t)
            
        return {
            "root_pos": cast(root_pos),
            "root_rot": cast(root_rot),
            "root_vel": cast(root_vel),
            "root_ang_vel": cast(root_ang_vel),
            "dof_pos": cast(dof_pos),
            "dof_vel": cast(dof_vel),
            "local_body_pos": cast(local_body_pos),
            "root_pos_delta_local": cast(root_pos_delta_local),
            "root_rot_delta_local": cast(root_rot_delta_local),
            "root_pos_delta": cast(root_pos_delta) # This is single vector usually, but keep consistent
        }

        # Duplicate code from above (exists in original codebase, keeping for consistency)
        self._motion_num_frames_cpu = self._motion_num_frames.to("cpu")

        num_motions = self.num_motions()
        self._motion_ids = torch.arange(num_motions, dtype=torch.long, device=self._device)

        total_len = self.get_total_length()
        print("Loaded {:d} motions with a total length of {:.3f}s.".format(num_motions, total_len))

    # ========================================================================
    # Periodic Resample Mode Methods
    # ========================================================================

    def enable_resample_mode(self, motion_ids: list):
        """Enable resample mode with MERGED GPU tensors (optimized for speed).

        Data is loaded DIRECTLY to GPU as merged tensors, avoiding CPU-GPU sync.
        Bypasses ALL CPU/GPU cache and LRU eviction logic.

        Args:
            motion_ids: List of motion IDs to load directly to GPU
        """
        from tqdm import tqdm
        import gc

        self._resample_mode = True
        self._loaded_subset_ids = set(motion_ids)
        # Cache the subset as a GPU tensor for fast sampling (avoid repeated conversions)
        self._loaded_subset_ids_tensor = torch.tensor(sorted(motion_ids), device=self._device, dtype=torch.long)

        # Clear ALL caches (we don't use them in resample mode)
        if self._lazy_load:
            self._cpu_motion_cache.clear()
            self._cpu_cache_bytes_used = 0
        self._clear_gpu_cache()
        self._resample_gpu_storage.clear()

        # When synchronous resample happens, clear both the ready event AND the data
        # This ensures the background thread will immediately start preparing new data
        if self._async_resample_enabled:
            # Clear the old pre-loaded data so worker will prepare fresh data
            if self._async_resample_next_data is not None:
                self._async_resample_next_data.clear()
                self._async_resample_next_data = None
                self._async_resample_next_ids = None
            # Clear the ready event to signal worker to prepare
            if self._async_resample_ready_event.is_set():
                self._async_resample_ready_event.clear()
            print(f"[MotionLib] Cleared async data and ready event (synchronous resample happened, background will reprepare)", flush=True)

        print(f"[MotionLib] Loading {len(motion_ids)} motions to GPU (merged tensors)...")

        # First pass: collect motion info (num_frames per motion)
        motion_info = []  # [(motion_id, num_frames, data)]
        total_frames = 0

        for motion_id in tqdm(motion_ids, desc="[MotionLib] Loading metadata", unit="motion"):
            data = self._load_motion_on_demand(motion_id)
            num_frames = data["root_pos"].shape[0]
            motion_info.append({
                "id": motion_id,
                "num_frames": num_frames,
                "start_frame": total_frames,
                "data": data,  # Keep reference to avoid reloading
            })
            total_frames += num_frames

        # Get dimensions from first motion
        first_data = motion_info[0]["data"]
        D = first_data["dof_pos"].shape[-1]
        B = first_data["local_body_pos"].shape[1]

        # CRITICAL: Delete old GPU tensors FIRST to avoid double memory usage
        if hasattr(self, '_gpu_root_pos'):
            del self._gpu_root_pos
        if hasattr(self, '_gpu_root_rot'):
            del self._gpu_root_rot
        if hasattr(self, '_gpu_root_vel'):
            del self._gpu_root_vel
        if hasattr(self, '_gpu_root_ang_vel'):
            del self._gpu_root_ang_vel
        if hasattr(self, '_gpu_dof_pos'):
            del self._gpu_dof_pos
        if hasattr(self, '_gpu_dof_vel'):
            del self._gpu_dof_vel
        if hasattr(self, '_gpu_local_body_pos'):
            del self._gpu_local_body_pos
        if hasattr(self, '_gpu_root_pos_delta_local'):
            del self._gpu_root_pos_delta_local
        if hasattr(self, '_gpu_root_rot_delta_local'):
            del self._gpu_root_rot_delta_local
        if hasattr(self, '_gpu_root_pos_delta'):
            del self._gpu_root_pos_delta

        # Force GPU cache flush before allocating new tensors
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()

        # Allocate merged tensors on GPU
        print(f"[MotionLib] Allocating {total_frames} frames on GPU...")
        self._gpu_root_pos = torch.empty((total_frames, 3), device=self._device, dtype=torch.float32)
        self._gpu_root_rot = torch.empty((total_frames, 4), device=self._device, dtype=torch.float32)
        self._gpu_root_vel = torch.empty((total_frames, 3), device=self._device, dtype=torch.float32)
        self._gpu_root_ang_vel = torch.empty((total_frames, 3), device=self._device, dtype=torch.float32)
        self._gpu_dof_pos = torch.empty((total_frames, D), device=self._device, dtype=torch.float32)
        self._gpu_dof_vel = torch.empty((total_frames, D), device=self._device, dtype=torch.float32)
        self._gpu_local_body_pos = torch.empty((total_frames, B, 3), device=self._device, dtype=torch.float32)
        self._gpu_root_pos_delta_local = torch.empty((total_frames, 3), device=self._device, dtype=torch.float32)
        self._gpu_root_rot_delta_local = torch.empty((total_frames, 3), device=self._device, dtype=torch.float32)

        # Create motion_id to frame range mapping
        self._motion_id_to_frame = {}  # {motion_id: (start_frame, num_frames)}

        # Create motion_id to index mapping for per-motion data (like root_pos_delta)
        self._motion_id_to_idx = {mid: i for i, mid in enumerate(motion_ids)}

        # Allocate per-motion tensors
        self._gpu_root_pos_delta = torch.empty((len(motion_ids), 3), device=self._device, dtype=torch.float32)

        # Second pass: fill merged tensors
        for info in tqdm(motion_info, desc="[MotionLib] Copying to GPU", unit="motion"):
            motion_id = info["id"]
            start = info["start_frame"]
            num_frames = info["num_frames"]
            end = start + num_frames
            data = info["data"]

            # Copy to GPU
            self._gpu_root_pos[start:end] = data["root_pos"].to(self._device)
            self._gpu_root_rot[start:end] = data["root_rot"].to(self._device)
            self._gpu_root_vel[start:end] = data["root_vel"].to(self._device)
            self._gpu_root_ang_vel[start:end] = data["root_ang_vel"].to(self._device)
            self._gpu_dof_pos[start:end] = data["dof_pos"].to(self._device)
            self._gpu_dof_vel[start:end] = data["dof_vel"].to(self._device)
            self._gpu_local_body_pos[start:end] = data["local_body_pos"].to(self._device)
            self._gpu_root_pos_delta_local[start:end] = data["root_pos_delta_local"].to(self._device)
            self._gpu_root_rot_delta_local[start:end] = data["root_rot_delta_local"].to(self._device)

            # Per-motion root_pos_delta
            # FIX: In resample mode, use the delta from the loaded data (data["root_pos_delta"]),
            # not from self._motion_root_pos_delta[motion_id] which may have wrong indexing
            root_pos_delta = data.get("root_pos_delta")
            if root_pos_delta is not None:
                self._gpu_root_pos_delta[self._motion_id_to_idx[motion_id]] = root_pos_delta.to(self._device)
            else:
                self._gpu_root_pos_delta[self._motion_id_to_idx[motion_id]] = 0.0

            self._motion_id_to_frame[motion_id] = (start, num_frames)

        # Create fast lookup table: motion_id -> start_frame (GPU tensor for no-sync indexing)
        max_motion_id = max(motion_ids)
        self._motion_start_frame = torch.zeros(max_motion_id + 1, device=self._device, dtype=torch.long)
        for mid, (start, _) in self._motion_id_to_frame.items():
            self._motion_start_frame[mid] = start

        # Create motion_id -> index lookup table (GPU tensor)
        self._motion_id_to_idx_tensor = torch.full((max_motion_id + 1,), -1, device=self._device, dtype=torch.long)
        for mid, idx in self._motion_id_to_idx.items():
            self._motion_id_to_idx_tensor[mid] = idx

        # IMPORTANT: Update _motion_start_idx, _motion_num_frames, _motion_lengths for resample mode
        # This makes _calc_frame_blend work correctly with the subset data
        num_motions_subset = len(motion_ids)
        subset_num_frames = torch.tensor([info["num_frames"] for info in motion_info],
                                        device=self._device, dtype=torch.long)
        # FIX: Use original motion_id (mid) to index _motion_fps, not subset index
        # In resample mode, _motion_fps contains the full dataset (indexed by original motion_id)
        subset_lengths = torch.tensor([info["num_frames"] / self._motion_fps[mid]
                                        for info, mid in zip(motion_info, motion_ids)],
                                        device=self._device, dtype=torch.float)

        # Store resample-specific data (don't overwrite original)
        self._motion_num_frames_resample = subset_num_frames
        self._motion_lengths_resample = subset_lengths

        # Calculate start_idx for each motion in the subset (in subset order, not motion_id order)
        # motion_info is in the same order as motion_ids
        lengths_shifted = torch.roll(subset_num_frames, 1)
        lengths_shifted[0] = 0
        start_indices = lengths_shifted.cumsum(0)

        # Create mapping from subset index to motion_id
        # _motion_start_idx[i] = start frame of motion with subset index i
        self._motion_start_idx = start_indices  # Overwrite with subset start_idx!

        # But we also need to handle motion_id indexing: _motion_start_idx[motion_id]
        # In resample mode, motion_ids can be arbitrary (not 0..N-1)
        # So we need a lookup table
        max_id = max(motion_ids)
        self._motion_start_idx_by_id = torch.full((max_id + 1,), -1, device=self._device, dtype=torch.long)
        for i, mid in enumerate(motion_ids):
            self._motion_start_idx_by_id[mid] = start_indices[i]

        # Cache dimensions
        self._resample_D = D
        self._resample_B = B
        self._resample_total_motions = len(motion_ids)
        self._resample_total_frames = total_frames

        # Get rank info for multi-GPU
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
                rank_str = f"Rank {rank}/{world_size}"
            else:
                rank = 0
                world_size = 1
                rank_str = "Rank 0/1"
        except:
            rank = 0
            world_size = 1
            rank_str = "Rank 0/1"

        # Calculate actual GPU memory used
        total_memory_mb = (
            (3 + 4 + 3 + 3) * 4 +  # root_pos, root_rot, root_vel, root_ang_vel (float32)
            (D + D) * 4 +            # dof_pos, dof_vel
            (B * 3) * 4 +            # local_body_pos
            (3 + 3) * 4              # root_pos_delta_local, root_rot_delta_local
        ) * total_frames / (1024**2)

        print(f"[{rank_str}] Resample loaded: {len(motion_ids):5d} motions, {total_frames:7d} frames, ~{total_memory_mb:.0f}MB GPU memory")


    def _motion_ids_to_indices(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Convert motion_ids to indices in the resample subset.

        Args:
            motion_ids: Tensor of motion IDs [n]

        Returns:
            Tensor of indices [n] for indexing into per-motion tensors
        """
        return self._motion_id_to_idx_tensor[motion_ids]

    def _preload_subset_cache(self, motion_ids: list):
        """NO-OP in resample mode (data is loaded directly to GPU).

        In resample mode, enable_resample_mode() loads data DIRECTLY to GPU,
        bypassing CPU cache entirely. This method is kept for compatibility
        but does nothing.

        Args:
            motion_ids: List of motion IDs (ignored)
        """
        # NO-OP: In resample mode, data is already on GPU via enable_resample_mode()
        pass

    def resample_subset(self, num_motions: int, seed: Optional[int] = None, motion_difficulty=None, preload=False, gpu_memory_budget_gb: Optional[float] = None):
        """Resample a new subset of motions and load DIRECTLY to GPU.

        Args:
            num_motions: Number of motions to sample (ignored if gpu_memory_budget_gb is specified)
            seed: Random seed for reproducibility
            motion_difficulty: Optional difficulty weights for sampling
            preload: IGNORED (kept for compatibility). Data is always loaded directly to GPU.
            gpu_memory_budget_gb: If specified, load motions until this GPU memory budget is reached (in GB).
                                     This takes precedence over num_motions. Uses cumulative sampling
                                     to maximize GPU memory utilization.

        Returns:
            List of sampled motion IDs
        """
        # Store resample config for async worker to use
        self._resample_num_motions = num_motions
        # Convert gpu_memory_budget_gb to float if it's a string (from command line args)
        if gpu_memory_budget_gb is not None:
            try:
                gpu_memory_budget_gb = float(gpu_memory_budget_gb)
            except (ValueError, TypeError):
                gpu_memory_budget_gb = None
        self._resample_gpu_memory_budget_gb = gpu_memory_budget_gb

        if seed is not None:
            torch.manual_seed(seed)

        num_total = len(self._full_motion_ids)

        # If GPU memory budget is specified, use cumulative sampling
        if gpu_memory_budget_gb is not None:
            sampled_ids = self._cumulative_sample_by_budget(
                gpu_memory_budget_gb,
                motion_difficulty,
                num_total
            )
        else:
            # Original logic: sample fixed number of motions
            sampled_ids = self._sample_fixed_num_motions(num_motions, motion_difficulty, num_total)

        # Enable resample mode with new subset (loads DIRECTLY to GPU)
        self.enable_resample_mode(sampled_ids)

        return sampled_ids

    def _cumulative_sample_by_budget(self, gpu_memory_budget_gb: float, motion_difficulty, num_total: int):
        """Sample motions cumulatively until GPU memory budget is reached.

        This method samples motions one by one (weighted by motion_weights and difficulty),
        accumulating the total frame count until the budget is approximately reached.
        This maximizes GPU memory utilization since different ranks may sample motions
        of different sizes.

        Args:
            gpu_memory_budget_gb: GPU memory budget in GB
            motion_difficulty: Optional difficulty weights for sampling
            num_total: Total number of motions available

        Returns:
            List of sampled motion IDs
        """
        # Calculate memory per frame (same as before)
        sample_motion_id = self._full_motion_ids[0]
        sample_data = self._load_motion_on_demand(sample_motion_id)
        D = sample_data["dof_pos"].shape[-1]
        B = sample_data["local_body_pos"].shape[1]

        bytes_per_frame = (
            3 + 4 + 3 + 3 +  # root_pos, root_rot, root_vel, root_ang_vel
            D + D +            # dof_pos, dof_vel
            B * 3 +            # local_body_pos
            3 + 3              # root_pos_delta_local, root_rot_delta_local
        ) * 4 * 1.1  # float32 with 10% buffer

        budget_bytes = gpu_memory_budget_gb * 1024**3
        max_frames = int(budget_bytes / bytes_per_frame)

        # Build sampling probability distribution
        if motion_difficulty is not None:
            motion_prob = self._motion_weights * motion_difficulty
        else:
            motion_prob = self._motion_weights.clone()

        # Normalize to sum to 1
        motion_prob = motion_prob / motion_prob.sum()

        # Get rank info for logging
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world_size = dist.get_world_size()
            else:
                rank = 0
                world_size = 1
        except:
            rank = 0
            world_size = 1

        # IMPORTANT: Use pre-loaded _motion_num_frames instead of loading each motion!
        # _motion_num_frames is indexed by motion_id (0 to max_motion_id)
        max_motion_id = max(self._full_motion_ids)
        motion_lengths = self._motion_num_frames[:max_motion_id+1].clone()

        # OPTIMIZED: Batch sampling instead of loop
        # Sample a large batch at once, then filter by budget
        estimated_motions = min(int(max_frames / 100), num_total)  # Assume avg 100 frames
        estimated_motions = max(estimated_motions, 5000)  # At least 5000

        # Sample 1.5x more than needed (we'll filter later)
        sample_batch_size = min(int(estimated_motions * 1.5), num_total)

        if motion_difficulty is not None:
            motion_prob = self._motion_weights * motion_difficulty
        else:
            motion_prob = self._motion_weights.clone()
        motion_prob = motion_prob / motion_prob.sum()

        # Sample in one batch (all on GPU, no Python loop)
        sampled_tensor = torch.multinomial(motion_prob, num_samples=sample_batch_size, replacement=False)

        # Get motion lengths for sampled indices (all on GPU)
        sampled_lengths = motion_lengths[sampled_tensor]

        # Cumulative sum to find where to cut (all on GPU, no Python loop)
        cumsum_frames = torch.cumsum(sampled_lengths, dim=0)

        # Find how many motions we can keep (all on GPU)
        budget_threshold = max_frames * 1.05
        valid_mask = cumsum_frames <= budget_threshold

        # Also ensure at least some motions are selected
        if not valid_mask.any():
            valid_mask[0] = True

        # Get indices to keep (all on GPU)
        valid_indices = torch.where(valid_mask)[0]
        sampled_tensor_filtered = sampled_tensor[valid_indices]

        # Convert to list of motion IDs
        sampled_ids = [self._full_motion_ids[i.item()] for i in sampled_tensor_filtered]

        total_frames = cumsum_frames[valid_indices[-1]].item() if len(valid_indices) > 0 else 0

        # Calculate actual memory usage
        total_memory_mb = bytes_per_frame * total_frames / (1024**2)

        print(f"[Rank {rank}/{world_size}] Cumulative sampling: {len(sampled_ids)} motions, "
              f"{total_frames} frames, ~{total_memory_mb:.0f}MB GPU memory "
              f"(budget: {gpu_memory_budget_gb:.2f}GB = {max_frames} frames)", flush=True)

        return sampled_ids

    def _sample_fixed_num_motions(self, num_motions: int, motion_difficulty, num_total: int):
        """Sample a fixed number of motions (original logic).

        Args:
            num_motions: Number of motions to sample
            motion_difficulty: Optional difficulty weights for sampling
            num_total: Total number of motions available

        Returns:
            List of sampled motion IDs
        """
        if num_motions >= num_total:
            sampled_ids = self._full_motion_ids.copy()
            print(f"[MotionLib] Requested {num_motions} motions, but only {num_total} available. Using all.")
        else:
            # Use the same sampling logic as sample_motions (weighted by motion_weights and difficulty)
            if motion_difficulty is not None:
                motion_prob = self._motion_weights * motion_difficulty
            else:
                motion_prob = self._motion_weights

            # Check if we need replacement (fewer non-zero probabilities than requested samples)
            num_non_zero = (motion_prob > 0).sum().item()
            if num_non_zero < num_motions:
                replacement = True
            else:
                replacement = num_motions > num_total

            sampled_tensor = torch.multinomial(
                motion_prob,
                num_samples=num_motions,
                replacement=replacement
            )
            sampled_ids = sampled_tensor.cpu().tolist()

        return sampled_ids

    def _clear_gpu_cache(self):
        """Clear the GPU cache."""
        if not self._gpu_cache_enabled:
            return

        self._cache_frames_used = 0
        self._cache_free = []
        self._cache_meta = {}
        self._cache_lru = OrderedDict()

        if hasattr(self, '_cache_offset'):
            self._cache_offset.fill_(-1)
        if hasattr(self, '_cache_len'):
            self._cache_len.fill_(0)

    def _is_motion_in_subset(self, motion_id: int) -> bool:
        """Check if a motion ID is in the current loaded subset."""
        if not self._resample_mode:
            return True  # If not in resample mode, all motions are available
        return motion_id in self._loaded_subset_ids

    # ========================================================================
    # End of Periodic Resample Mode Methods
    # ========================================================================

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

    def _to_storage_tensor(self, x, dtype: torch.dtype = None) -> torch.Tensor:
        """Convert input to storage tensor with configurable dtype.
        
        If dtype is None, uses self._storage_dtype (configured via storage_dtype param).
        """
        if dtype is None:
            dtype = self._storage_dtype
        if isinstance(x, torch.Tensor):
            return x.to(device=self._storage_device, dtype=dtype)
        return torch.as_tensor(x, dtype=dtype, device=self._storage_device)

    def _cache_bytes_per_frame(self) -> int:
        # Handle lazy_load mode where tensors are lists of None
        if self._lazy_load:
            # Load first motion to get dimensions, then cache dimensions
            if not hasattr(self, '_cached_dof_dim'):
                # Load first motion data briefly to get shape info
                if len(self._motion_files) == 0:
                    raise RuntimeError("No motion files available for dimension calculation")
                first_motion_data = self._load_motion_data(self._motion_files[0])
                if first_motion_data is None:
                    raise RuntimeError(f"Failed to load first motion file: {self._motion_files[0]}")
                dof_pos = first_motion_data["dof_pos"]
                local_body_pos = first_motion_data["local_body_pos"]
                if isinstance(dof_pos, torch.Tensor):
                    self._cached_dof_dim = int(dof_pos.shape[-1])
                    self._cached_body_count = int(local_body_pos.shape[1])
                else:
                    self._cached_dof_dim = int(dof_pos.shape[-1])
                    self._cached_body_count = int(local_body_pos.shape[1])
                del first_motion_data
            D = self._cached_dof_dim
            B = self._cached_body_count
        else:
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

        # Get dimensions (handle lazy_load mode)
        if self._lazy_load:
            # Ensure dimensions are cached by calling _cache_bytes_per_frame
            self._cache_bytes_per_frame()
            D = self._cached_dof_dim
            B = self._cached_body_count
        else:
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
        
        # Determine source tensors
        if self._lazy_load:
            self._ensure_motion_loaded(motion_id)
            src_data = self._cpu_motion_cache[motion_id]
            # Helper to copy from dict to GPU cache
            def copy_src(target, key):
                target[off:end_off].copy_(src_data[key], non_blocking=True)

            copy_src(self._cache_root_pos, "root_pos")
            copy_src(self._cache_root_rot, "root_rot")
            copy_src(self._cache_root_vel, "root_vel")
            copy_src(self._cache_root_ang_vel, "root_ang_vel")
            copy_src(self._cache_dof_pos, "dof_pos")
            copy_src(self._cache_dof_vel, "dof_vel")
            copy_src(self._cache_local_body_pos, "local_body_pos")
            copy_src(self._cache_root_pos_delta_local, "root_pos_delta_local")
            copy_src(self._cache_root_rot_delta_local, "root_rot_delta_local")
        else:
            # Direct CPU->GPU copy into cache slices.
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
        if self._lazy_load:
             # in lazy mode, assume we always want to ensure loaded even if no GPU cache?
             # But 'prefetch' name implies optional optimization.
             # However, calc_motion_frame needs _ensure_motion_loaded called.
             # We'll let calc_motion_frame call _ensure_motion_loaded explicitly for fallback path.
             # Here we only care about GPU cache population.
             pass

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
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except pickle.UnpicklingError as e:
            msg = str(e)
            if "pickle data was truncated" not in msg:
                raise
            # Some GMR-generated motion.pkl files were observed to be missing the final STOP opcode
            pass
        except Exception:
            raise

        # Some GMR-generated motion.pkl files were observed to be missing the final STOP opcode
        # (exactly 1 byte shorter than the PROTO5 FRAME length). Repair in-memory on the fly.
        with open(path, "rb") as f:
            data = f.read()
        repaired = self._repair_proto5_frame_missing_stop(data, path=path)
        return pickle.loads(repaired)

    # ... [Keep helper methods like _repair_proto5_frame_missing_stop etc] ...
    
    # ... [Move to calc_motion_frame] ...

    def calc_motion_frame(self, motion_ids, motion_times):
        motion_ids = motion_ids.to(self._device)
        motion_times = motion_times.to(self._device)

        motion_loop_num = torch.floor(motion_times / self._motion_lengths[motion_ids])
        motion_times -= motion_loop_num * self._motion_lengths[motion_ids]

        frame_idx0, frame_idx1, frame_idx0_local, frame_idx1_local, blend = self._calc_frame_blend(motion_ids, motion_times)

        # =====================================================================
        # RESAMPLE MODE: Merged GPU tensors (same logic as official, no CPU-GPU sync)
        # =====================================================================
        if self._resample_mode:
            # Use the same logic as official implementation
            # frame_idx0/1 are already absolute indices (computed by _calc_frame_blend)

            root_pos0 = self._gpu_root_pos[frame_idx0]
            root_pos1 = self._gpu_root_pos[frame_idx1]

            root_rot0 = self._gpu_root_rot[frame_idx0]
            root_rot1 = self._gpu_root_rot[frame_idx1]

            root_vel = self._gpu_root_vel[frame_idx0]
            root_ang_vel = self._gpu_root_ang_vel[frame_idx0]

            dof_pos0 = self._gpu_dof_pos[frame_idx0]
            dof_pos1 = self._gpu_dof_pos[frame_idx1]

            local_key_body_pos0 = self._gpu_local_body_pos[frame_idx0]
            local_key_body_pos1 = self._gpu_local_body_pos[frame_idx1]

            dof_vel = self._gpu_dof_vel[frame_idx0]

            blend_unsqueeze = blend.unsqueeze(-1)
            root_pos = (1.0 - blend_unsqueeze) * root_pos0 + blend_unsqueeze * root_pos1
            # Get root_pos_delta per motion
            motion_indices = self._motion_ids_to_indices(motion_ids)
            root_pos += motion_loop_num.unsqueeze(-1) * self._gpu_root_pos_delta[motion_indices]
            root_rot = slerp(root_rot0, root_rot1, blend)

            dof_pos = (1.0 - blend_unsqueeze) * dof_pos0 + blend_unsqueeze * dof_pos1

            local_key_body_pos = (1.0 - blend_unsqueeze.unsqueeze(1)) * local_key_body_pos0 + blend_unsqueeze.unsqueeze(1) * local_key_body_pos1

            root_pos_delta_local0 = self._gpu_root_pos_delta_local[frame_idx0]
            root_pos_delta_local1 = self._gpu_root_pos_delta_local[frame_idx1]
            root_pos_delta_local = (1.0 - blend_unsqueeze) * root_pos_delta_local0 + blend_unsqueeze * root_pos_delta_local1

            root_rot_delta_local0 = self._gpu_root_rot_delta_local[frame_idx0]
            root_rot_delta_local1 = self._gpu_root_rot_delta_local[frame_idx1]
            root_rot_delta_local = (1.0 - blend_unsqueeze) * root_rot_delta_local0 + blend_unsqueeze * root_rot_delta_local1

            return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, root_pos_delta_local, root_rot_delta_local

        # =====================================================================
        # ORIGINAL MODE: CPU/GPU cache path
        # =====================================================================
        # Ensure motions are loaded if using lazy load (must happen before gathering, cache or not)
        if self._lazy_load:
            # Gather unique IDs on CPU to avoid devicesync if possible, but IDs are on GPU.
            # We take the hit of unique() and cpu() transfer for the metadata check.
            uniq_ids = torch.unique(motion_ids)
            # Optimization: check which are missing from CPU cache first?
            # self._ensure_motion_loaded checks internally.
            # But converting to list is necessary loop.
            uniq_ids_cpu = uniq_ids.to("cpu").tolist()
            for mid in uniq_ids_cpu:
                 self._ensure_motion_loaded(mid)

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
        D = int(self._motion_dof_pos.shape[-1]) if not self._lazy_load else int(self._to_storage_tensor(torch.zeros(1)).shape[-1]) # Fallback for lazy...
        # Wait, if lazy load, we don't have _motion_dof_pos list populated with tensors to check shape.
        # We need D and B.
        # We can get them from the first loaded motion in cache, or store them in metadata during load.
        # During _load_motions (lazy), we didn't store D/B.
        # FIX: We should store D and B in _load_motions even in lazy mode.
        # But we discarded data.
        # Let's assume we can get it from the first motion in cache or just fetch one if empty.
        
        if self._lazy_load:
             # Ensure at least one motion is loaded to check dims
             if len(self._cpu_motion_cache) == 0:
                  self._ensure_motion_loaded(int(motion_ids[0].item()))
             # Grab any
             _any_data = next(iter(self._cpu_motion_cache.values()))
             D = int(_any_data["dof_pos"].shape[-1])
             B = int(_any_data["local_body_pos"].shape[1])
        else:
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
            
            if not self._lazy_load:
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
            else:
                # Lazy load gathering fallback
                # Iterate over unique motions in the missing set
                missing_mids = motion_ids[idx]
                uniq_missing = torch.unique(missing_mids)
                
                # Pre-fetch indices to CPU
                idx_cpu = idx.to("cpu")
                missing_mids_cpu = missing_mids.to("cpu")
                fi0_local_cpu = frame_idx0_local[idx].to("cpu")
                fi1_local_cpu = frame_idx1_local[idx].to("cpu")
                
                uniq_missing_cpu = uniq_missing.to("cpu").tolist()
                
                for mid in uniq_missing_cpu:
                    mask = (missing_mids_cpu == mid)
                    # Indices into 'idx' array
                    sub_idx = mask.nonzero(as_tuple=False).flatten()
                    
                    # Indices into output tensors
                    target_idx = idx[sub_idx.to(idx.device)]
                    
                    # Local frame indices
                    fi0_loc = fi0_local_cpu[sub_idx]
                    fi1_loc = fi1_local_cpu[sub_idx]
                    
                    m_data = self._cpu_motion_cache[mid]
                    
                    # Gather and copy
                    def gather_copy(out, key, fix):
                        src = m_data[key].to(self._device, non_blocking=True) # Move whole tensor to GPU or slice on CPU?
                        # Moving whole tensor might be expensive. Slicing on CPU then moving is better.
                        # m_data[key] is on CPU.
                        # Slicing:
                        loaded = m_data[key][fix] # CPU slice
                        out[target_idx] = loaded.to(self._device, non_blocking=True)

                    gather_copy(root_pos0, "root_pos", fi0_loc)
                    gather_copy(root_pos1, "root_pos", fi1_loc)
                    gather_copy(root_rot0, "root_rot", fi0_loc)
                    gather_copy(root_rot1, "root_rot", fi1_loc)
                    gather_copy(root_vel, "root_vel", fi0_loc)
                    gather_copy(root_ang_vel, "root_ang_vel", fi0_loc)
                    gather_copy(dof_pos0, "dof_pos", fi0_loc)
                    gather_copy(dof_pos1, "dof_pos", fi1_loc)
                    gather_copy(local_key_body_pos0, "local_body_pos", fi0_loc)
                    gather_copy(local_key_body_pos1, "local_body_pos", fi1_loc)
                    gather_copy(dof_vel, "dof_vel", fi0_loc)
        
        blend_unsqueeze = blend.unsqueeze(-1)
        root_pos = (1.0 - blend_unsqueeze) * root_pos0 + blend_unsqueeze * root_pos1
        
        # Handle root_pos_delta
        if not self._lazy_load:
            root_pos += motion_loop_num.unsqueeze(-1) * self._motion_root_pos_delta[motion_ids]
        else:
             # Lazy load delta gathering
             # self._motion_root_pos_delta is list of Nones.
             # We need to gather from cache.
             # Assuming we can just do a loop similar to above or just fill it.
             # Actually, simpler: gather 'delta' into a temp tensor
             root_pos_delta_batch = torch.empty((n, 3), device=self._device, dtype=torch.float32)
             # Reuse the uniq loop logic? Or separate? 
             # Since we have motion_ids, we can just iterate unique motions again.
             # OR we optimize and do it in the loop above.
             # Let's do it in a separate block for clarity or mix it?
             # For performance, mix it.
             pass 

        root_rot = slerp(root_rot0, root_rot1, blend)
        
        dof_pos = (1.0 - blend_unsqueeze) * dof_pos0 + blend_unsqueeze * dof_pos1
        
        local_key_body_pos = (1.0 - blend_unsqueeze.unsqueeze(1)) * local_key_body_pos0 + blend_unsqueeze.unsqueeze(1) * local_key_body_pos1
        
        # compute the root pos/rot delta compared to last frame
        root_pos_delta_local0 = torch.empty((n, 3), device=self._device, dtype=torch.float32)
        root_pos_delta_local1 = torch.empty((n, 3), device=self._device, dtype=torch.float32)
        root_rot_delta_local0 = torch.empty((n, 3), device=self._device, dtype=torch.float32)
        root_rot_delta_local1 = torch.empty((n, 3), device=self._device, dtype=torch.float32)

        if use_cache and bool(cached_mask.any()):
            idx = cached_mask.nonzero(as_tuple=False).flatten()
            cache_off_c = cache_off[idx]
            cache_idx0 = cache_off_c + frame_idx0_local[idx].to(torch.int64)
            cache_idx1 = cache_off_c + frame_idx1_local[idx].to(torch.int64)
            
            root_pos_delta_local0[idx] = self._cache_root_pos_delta_local[cache_idx0]
            root_pos_delta_local1[idx] = self._cache_root_pos_delta_local[cache_idx1]
            root_rot_delta_local0[idx] = self._cache_root_rot_delta_local[cache_idx0]
            root_rot_delta_local1[idx] = self._cache_root_rot_delta_local[cache_idx1]
            
        if (not use_cache) or bool(cached_mask.logical_not().any()):
            idx = torch.arange(n, device=self._device) if (not use_cache) else cached_mask.logical_not().nonzero(as_tuple=False).flatten()
            
            if not self._lazy_load:
                root_pos_delta_local0[idx] = self._gather_frames(self._motion_root_pos_delta_local, frame_idx0[idx])
                root_pos_delta_local1[idx] = self._gather_frames(self._motion_root_pos_delta_local, frame_idx1[idx])
                root_rot_delta_local0[idx] = self._gather_frames(self._motion_root_rot_delta_local, frame_idx0[idx])
                root_rot_delta_local1[idx] = self._gather_frames(self._motion_root_rot_delta_local, frame_idx1[idx])
            else:
                 # Lazy load gathering fallback (continued)
                 missing_mids = motion_ids[idx]
                 uniq_missing = torch.unique(missing_mids)
                 # Repetitive code, we should have grouped it.
                 # Re-looping for deltas
                 idx_cpu = idx.to("cpu")
                 missing_mids_cpu = missing_mids.to("cpu")
                 fi0_local_cpu = frame_idx0_local[idx].to("cpu")
                 fi1_local_cpu = frame_idx1_local[idx].to("cpu")
                 
                 for mid in uniq_missing.to("cpu").tolist():
                    mask = (missing_mids_cpu == mid)
                    sub_idx = mask.nonzero(as_tuple=False).flatten()
                    target_idx = idx[sub_idx.to(idx.device)]
                    fi0_loc = fi0_local_cpu[sub_idx]
                    fi1_loc = fi1_local_cpu[sub_idx]
                    
                    m_data = self._cpu_motion_cache[mid]
                    
                    def gather_copy(out, key, fix):
                        loaded = m_data[key][fix]
                        out[target_idx] = loaded.to(self._device, non_blocking=True)
                        
                    gather_copy(root_pos_delta_local0, "root_pos_delta_local", fi0_loc)
                    gather_copy(root_pos_delta_local1, "root_pos_delta_local", fi1_loc)
                    gather_copy(root_rot_delta_local0, "root_rot_delta_local", fi0_loc)
                    gather_copy(root_rot_delta_local1, "root_rot_delta_local", fi1_loc)
        
        # Address the missing root_pos_delta gathering for lazy load
        if self._lazy_load:
             # We need to gather root_pos_delta which was skipped in the first block
             # Ideally we should merge these loops, but for code structure minimal invasiveness:
             # Loop over ALL unique motions in batch to fill root_pos_delta
             # (Note: root_pos_delta is needed for everyone, cached or not, typically)
             # Wait, in standard mode: `root_pos += ... * self._motion_root_pos_delta[motion_ids]`
             # `_motion_root_pos_delta` is (NumMotions, 3) on GPU.
             # In lazy mode, it's list of Nones.
             # We need to construct a batch tensor.
             root_pos_delta_batch = torch.empty((n, 3), device=self._device)
             
             uniq_ids = torch.unique(motion_ids)
             for mid in uniq_ids.to("cpu").tolist():
                  # This covers cached and non-cached
                  mask = (motion_ids == mid)
                  # ensure loaded (already done)
                  m_data = self._cpu_motion_cache[mid]
                  # m_data["root_pos_delta"] is 1D tensor?
                  # In _load_motion_on_demand we stored it.
                  delta = m_data["root_pos_delta"].to(self._device, non_blocking=True)
                  root_pos_delta_batch[mask] = delta
             
             root_pos += motion_loop_num.unsqueeze(-1) * root_pos_delta_batch

        root_pos_delta_local = (1.0 - blend_unsqueeze) * root_pos_delta_local0 + blend_unsqueeze * root_pos_delta_local1
        root_rot_delta_local = (1.0 - blend_unsqueeze) * root_rot_delta_local0 + blend_unsqueeze * root_rot_delta_local1

        return root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, local_key_body_pos, root_pos_delta_local, root_rot_delta_local

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
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except pickle.UnpicklingError as e:
            msg = str(e)
            if "pickle data was truncated" not in msg:
                raise

        # Some GMR-generated motion.pkl files were observed to be missing the final STOP opcode
        # (exactly 1 byte shorter than the PROTO5 FRAME length). Repair in-memory on the fly.
        with open(path, "rb") as f:
            data = f.read()
        repaired = self._repair_proto5_frame_missing_stop(data, path=path)
        return pickle.loads(repaired)

    @staticmethod
    def _repair_proto5_frame_missing_stop(data: bytes, *, path: str) -> bytes:
        """Repair a PROTO5+FRAME pickle missing exactly the final STOP byte ('.').

        Raises ValueError if the blob does not match the expected pattern.
        """
        if len(data) < 11:
            raise ValueError(f"Pickle too short to repair: {path} (size={len(data)})")
        if not (data[0] == 0x80 and data[1] == 0x05 and data[2] == 0x95):
            raise ValueError(f"Unsupported pickle header (expected PROTO5+FRAME): {path}")
        frame_len = struct.unpack("<Q", data[3:11])[0]
        expected_size = 11 + int(frame_len)
        if len(data) == expected_size - 1:
            global _WARNED_TRUNCATED_PICKLE
            if not _WARNED_TRUNCATED_PICKLE:
                warnings.warn(
                    f"[MotionLib] Detected truncated pickle missing STOP; repairing in-memory (example: {path}). "
                    "Consider regenerating the dataset to avoid this overhead.",
                    RuntimeWarning,
                )
                _WARNED_TRUNCATED_PICKLE = True
            return data + b"."
        raise ValueError(
            f"Cannot repair pickle: {path} (size={len(data)} expected={expected_size} diff={len(data)-expected_size})"
        )

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
        # In resample mode, only sample from the loaded subset
        if self._resample_mode and self._loaded_subset_ids:
            # Use cached GPU tensor (much faster than converting list each time)
            if self._loaded_subset_ids_tensor is None:
                # Fallback: create the cached tensor if it doesn't exist
                print("[MotionLib] WARNING: _loaded_subset_ids_tensor is None, creating it now...")
                self._loaded_subset_ids_tensor = torch.tensor(sorted(self._loaded_subset_ids), device=self._device, dtype=torch.long)
            subset_ids = self._loaded_subset_ids_tensor
            subset_weights = self._motion_weights[subset_ids]

            # Apply difficulty if provided
            if motion_difficulty is not None:
                subset_difficulty = motion_difficulty[subset_ids]
                if use_error_aware_sampling and max_key_body_error is not None:
                    # Apply error aware sampling formula
                    error_aware_prob = torch.ones_like(subset_difficulty)
                    difficulty_one_mask = (subset_difficulty == 1.0)
                    if difficulty_one_mask.any():
                        subset_error = max_key_body_error[subset_ids]
                        normalized_error = torch.clamp(subset_error / error_sampling_threshold, max=1.0)
                        error_prob = normalized_error ** error_sampling_power
                        error_aware_prob[difficulty_one_mask] = error_prob[difficulty_one_mask]
                    difficulty_gt_one_mask = (subset_difficulty > 1.0)
                    error_aware_prob[difficulty_gt_one_mask] = subset_difficulty[difficulty_gt_one_mask]
                    subset_prob = subset_weights * error_aware_prob
                else:
                    subset_prob = subset_weights * subset_difficulty
            else:
                subset_prob = subset_weights

            # Sample from subset
            subset_indices = torch.multinomial(subset_prob, num_samples=n, replacement=True)
            motion_ids = subset_ids[subset_indices]
        else:
            # Original sampling logic (non-resample mode)
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
            # NOTE: Skip DDP sharding if skip_ddp_sharding=True (resample mode) to allow sampling from full dataset.
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    world_size = dist.get_world_size()
                    if world_size > 1 and not self._skip_ddp_sharding:
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
                    elif world_size > 1 and self._skip_ddp_sharding:
                        # In resample mode, skip DDP sharding so each rank can sample from full dataset
                        # Each rank will load all metadata and sample independently during resample
                        if dist.get_rank() == 0:
                            print(f"[MotionLib] Skipping DDP sharding: each rank loads full dataset for resampling")
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
        # In resample mode, use resample-specific data
        if self._resample_mode:
            # CRITICAL: motion_ids are original IDs (0-562), but _motion_num_frames_resample
            # is indexed by subset position (0-49). Need to map via _motion_id_to_idx_tensor.
            subset_indices = self._motion_id_to_idx_tensor[motion_ids]
            num_frames = self._motion_num_frames_resample[subset_indices]
            # motion_lengths is still from full dataset (indexed by original motion_ids)
            motion_lengths = self._motion_lengths
        else:
            num_frames = self._motion_num_frames[motion_ids]
            motion_lengths = self._motion_lengths

        phase = times / motion_lengths[motion_ids]
        phase = torch.clip(phase, 0.0, 1.0)

        frame_idx0_local = (phase * (num_frames - 1)).long()
        frame_idx1_local = torch.min(frame_idx0_local + 1, num_frames - 1)
        blend = phase * (num_frames - 1) - frame_idx0_local.float()

        # In resample mode, use _motion_start_idx_by_id (maps motion_id -> start_frame)
        if self._resample_mode:
            frame_start_idx = self._motion_start_idx_by_id[motion_ids]
        else:
            frame_start_idx = self._motion_start_idx[motion_ids]

        frame_idx0 = frame_idx0_local + frame_start_idx
        frame_idx1 = frame_idx1_local + frame_start_idx

        return frame_idx0, frame_idx1, frame_idx0_local, frame_idx1_local, blend
        

    
    def get_key_body_idx(self, key_body_names):
        key_body_idx = []
        for key_body_name in key_body_names:
            key_body_idx.append(self._body_link_list.index(key_body_name))
        return key_body_idx # list
    
    def get_motion_names(self):
        return self._motion_names

    def save_difficulty_to_csv(self, log_dir, iteration, motion_difficulty, rank=0):
        """
        Save motion difficulty values to a CSV file.

        Args:
            log_dir: Directory where the difficulty folder will be created
            iteration: Current training iteration (used in filename)
            motion_difficulty: Tensor of difficulty values (1-10 scale, or 100.0 for initial)
            rank: Process rank for multi-GPU training (used in filename). None = no suffix.
        """
        import csv

        # Create difficulty directory
        difficulty_dir = os.path.join(log_dir, "difficulty")
        os.makedirs(difficulty_dir, exist_ok=True)

        # Prepare CSV file path (include rank in filename only if specified)
        if rank is not None:
            csv_file = os.path.join(difficulty_dir, f"difficulty_iter_{iteration:07d}_rank{rank}.csv")
        else:
            csv_file = os.path.join(difficulty_dir, f"difficulty_iter_{iteration:07d}.csv")

        # Get motion file paths and difficulties (use paths instead of names for easier lookup)
        motion_files = self._motion_files  # List of file paths
        difficulties_cpu = motion_difficulty.cpu().numpy()

        # Map to 0-100 scale
        # Note: initial value is 100.0, after updates it's clamped to 1-10 range
        # For initial 100.0, treat as 0 (not yet trained)
        # For 1-10 range, map to 0-100: (x - 1) / 9 * 100
        difficulties_0_100 = []
        for d in difficulties_cpu:
            if d >= 50.0:  # Initial value (100.0) or similar
                difficulties_0_100.append(0.0)
            else:
                difficulties_0_100.append((d - 1.0) / 9.0 * 100.0)
        difficulties_0_100 = np.array(difficulties_0_100)

        # Write to CSV with file paths instead of motion names
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['motion_idx', 'motion_path', 'difficulty_0_100', 'difficulty_raw'])
            for idx, (path, diff_0_100, diff_raw) in enumerate(zip(motion_files, difficulties_0_100, difficulties_cpu)):
                writer.writerow([idx, str(path), f"{diff_0_100:.2f}", f"{diff_raw:.4f}"])

        print(f"[MotionLib] Saved motion difficulty to {csv_file}")

    def load_difficulty_from_csvs(self, csv_files: List[str]) -> "torch.Tensor":
        """Load and merge difficulty from multiple CSV files, match by current motion paths.

        This method uses the current MotionLib instance's _motion_files (which has already
        been processed by shuffle, motion_ids filter, max_motions, DDP sharding, etc.)
        to ensure the returned difficulties are in the correct order.

        Args:
            csv_files: List of CSV file paths (from multiple ranks)

        Returns:
            Tensor of difficulty values matching current motion order (1-10 scale)

        Raises:
            FileNotFoundError: If any current motion not found in CSVs
        """
        import csv

        # Step 1: Read all CSV files and build motion_path -> difficulties mapping
        motion_path_to_diffs: Dict[str, List[float]] = {}

        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                print(f"[MotionLib] Warning: CSV file not found: {csv_file}")
                continue

            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    motion_path = row['motion_path']
                    diff_0_100 = float(row['difficulty_0_100'])

                    if motion_path not in motion_path_to_diffs:
                        motion_path_to_diffs[motion_path] = []
                    motion_path_to_diffs[motion_path].append(diff_0_100)

        if not motion_path_to_diffs:
            raise FileNotFoundError(f"No valid difficulty data found in CSV files: {csv_files}")

        # Step 2: Match current motion paths to CSV entries
        # Use self._motion_files which is the FINAL list after all preprocessing
        # (shuffle, motion_ids filter, max_motions, DDP sharding, etc.)
        difficulties = []
        missing_motions = []

        for curr_path in self._motion_files:
            if curr_path in motion_path_to_diffs:
                # Take MIN across all ranks (consistent with training sync logic)
                min_diff = min(motion_path_to_diffs[curr_path])
                difficulties.append(min_diff)
            else:
                missing_motions.append(curr_path)

        if missing_motions:
            raise FileNotFoundError(
                f"Found {len(missing_motions)} current motions that are not in CSV difficulty files.\n"
                f"First few missing: {missing_motions[:5]}\n"
                f"CSV files: {csv_files}"
            )

        # Step 3: Convert 0-100 scale back to internal 1-10 scale
        # Formula reverse: internal = 0_100 / 100 * 9 + 1
        difficulties_tensor = torch.tensor(difficulties, dtype=torch.float32, device=self._device)
        # Convert 0-100 to 1-10
        difficulties_internal = difficulties_tensor / 100.0 * 9.0 + 1.0

        return difficulties_internal


    # ============================================================
    # Async Resample: Prepare next subset in background on CPU
    # ============================================================
    
    def enable_async_resample(self, interval: int):
        """Enable async resample mode.

        In this mode, a background thread prepares the next subset on CPU while
        training continues. When it's time to resample, we simply switch to the
        pre-loaded data (fast CPU->GPU copy).

        Args:
            interval: Resample interval (in iterations)
        """
        import threading
        import time

        if self._async_resample_thread is not None:
            self.disable_async_resample()

        self._async_resample_enabled = True
        self._async_resample_interval = interval
        self._async_resample_stop_event = threading.Event()
        self._async_resample_ready_event = threading.Event()
        self._async_resample_lock = threading.Lock()

        # Start background thread
        self._async_resample_thread = threading.Thread(
            target=self._async_resample_worker,
            daemon=True,
            name="AsyncResampleWorker"
        )
        self._async_resample_thread.start()

        print(f"[MotionLib] ========== ASYNC RESAMPLE ENABLED: interval={interval} iterations ==========", flush=True)

        # IMPORTANT: Trigger the first preparation immediately after thread starts
        # Set _async_resample_last_iteration to trigger preparation at 10% of first interval
        # This ensures the first resample will have pre-loaded data ready
        time.sleep(0.2)  # Give thread a moment to start
        with self._async_resample_lock:
            # Set to (10% of interval) so that worker will trigger at 10% point
            # This gives 90% of interval time for first preparation
            prepare_offset = self._async_resample_interval // 10
            print(f"[AsyncResample] ENABLE: setting last={prepare_offset} (interval={self._async_resample_interval})", flush=True)
            self._async_resample_last_iteration = prepare_offset  # Set to prepare point, so worker triggers immediately
            # The worker thread will pick this up and trigger preparation
    
    def disable_async_resample(self):
        """Disable async resample and clean up resources."""
        import gc

        self._async_resample_enabled = False

        if self._async_resample_stop_event is not None:
            self._async_resample_stop_event.set()

        if self._async_resample_thread is not None:
            self._async_resample_thread.join(timeout=5.0)
            if self._async_resample_thread.is_alive():
                print("[MotionLib] Warning: Async resample thread did not stop gracefully")
            self._async_resample_thread = None

        # Clean up large data structures explicitly
        if self._async_resample_next_data is not None:
            self._async_resample_next_data.clear()
        self._async_resample_next_data = None
        self._async_resample_next_ids = None

        self._async_resample_stop_event = None
        self._async_resample_ready_event = None

        # Force garbage collection to free memory
        gc.collect()

        print("[MotionLib] Async resample disabled and resources cleaned up")
    
    def _async_resample_worker(self):
        """Background worker thread that prepares the next subset on CPU."""
        import time

        try:
            # Use stored resample config from MotionLib initialization
            num_motions = getattr(self, "_resample_num_motions", 15000)
            gpu_memory_budget_gb = getattr(self, "_resample_gpu_memory_budget_gb", None)

            budget_str = f"{gpu_memory_budget_gb}GB" if gpu_memory_budget_gb else f"{num_motions} motions"
            print(f"[AsyncResample] Worker STARTED: interval={self._async_resample_interval}, budget={budget_str}", flush=True)

            while not self._async_resample_stop_event.is_set():
                time.sleep(0.1)  # Check every 100ms

                # Check if we should prepare next subset
                with self._async_resample_lock:
                    current_iteration = self._async_resample_last_iteration

                    # Check if we should prepare next subset for upcoming resample
                    # Strategy: Prepare at 10% of each interval cycle (fixed trigger point)
                    # Example (interval=200): Prepare at 20, 220, 420... for use at 200, 400, 600...
                    # This gives 90% of the interval time for preparation
                    prepare_offset = self._async_resample_interval // 10  # At least 1, or 10% of interval
                    at_prepare_point = (current_iteration % self._async_resample_interval == prepare_offset)

                    should_prepare = at_prepare_point

                    # Also IMMEDIATELY trigger if ready_event is cleared (sync resample happened)
                    # This ensures async mode recovers quickly after a sync fallback
                    if not should_prepare and not self._async_resample_ready_event.is_set():
                        should_prepare = True
                        print(f"[AsyncResample] Ready event cleared (sync resample happened), immediately preparing next subset...", flush=True)

                    # Even if ready_event is_set, we still need to prepare at trigger point
                    if at_prepare_point and (self._async_resample_next_data is None or not self._async_resample_ready_event.is_set()):
                        should_prepare = True

                    if should_prepare:
                        # Check if data is already being prepared or is ready
                        if self._async_resample_next_data is None or not self._async_resample_ready_event.is_set():
                            # Calculate the next resample iteration (round up to next interval boundary)
                            next_resample = ((current_iteration // self._async_resample_interval) + 1) * self._async_resample_interval
                            print(f"[AsyncResample] TRIGGER: last={current_iteration}, interval={self._async_resample_interval}, next_resample={next_resample})", flush=True)
                            # Clear old data and prepare new subset
                            if self._async_resample_next_data is not None:
                                self._async_resample_next_data.clear()
                            self._async_resample_next_data = None
                            self._prepare_next_subset(num_motions, gpu_memory_budget_gb)

                            if self._async_resample_next_data is not None:
                                self._async_resample_ready_event.set()
                                print(f"[AsyncResample] Next subset READY: {len(self._async_resample_next_ids)} motions - waiting for resample trigger", flush=True)
        except Exception as e:
            print(f"[AsyncResample] Worker error: {e}")
            import traceback
            traceback.print_exc()
    
    def _prepare_next_subset(self, num_motions: int, gpu_memory_budget_gb: float = None):
        """Prepare the next subset on CPU in background."""
        import time

        t0 = time.time()
        print(f"[AsyncResample] _prepare_next_subset STARTED", flush=True)

        try:
            # Sample next subset (same logic as resample_subset)
            seed = int(time.time() * 1000000)  # Random seed
            if seed is not None:
                torch.manual_seed(seed)

            num_total = len(self._full_motion_ids)

            # Convert gpu_memory_budget_gb to float if it's a string
            if gpu_memory_budget_gb is not None:
                try:
                    gpu_memory_budget_gb = float(gpu_memory_budget_gb)
                except (ValueError, TypeError):
                    gpu_memory_budget_gb = None

            # Use cumulative sampling if budget specified
            if gpu_memory_budget_gb is not None:
                print(f"[AsyncResample] Sampling with GPU budget {gpu_memory_budget_gb}GB...", flush=True)
                sampled_ids = self._cumulative_sample_by_budget(
                    gpu_memory_budget_gb, None, num_total
                )
            else:
                print(f"[AsyncResample] Sampling {num_motions} motions...", flush=True)
                # Fixed number sampling
                sampled_ids = self._sample_fixed_num_motions(
                    num_motions, None, num_total
                )
            print(f"[AsyncResample] Sampled {len(sampled_ids)} motions, loading data to CPU...", flush=True)

            # Prepare data on CPU (load to CPU tensors, not GPU)
            next_data = {}
            total_frames = 0
            load_start = time.time()
            for i, motion_id in enumerate(sampled_ids):
                data = self._load_motion_on_demand(motion_id)
                # IMPORTANT: Clone tensors to avoid holding references to cached data
                # This prevents the CPU cache from being locked by async pre-loading
                next_data[motion_id] = {
                    'root_pos': data["root_pos"].clone(),  # Clone to break reference to cache
                    'root_rot': data["root_rot"].clone(),
                    'root_vel': data["root_vel"].clone(),
                    'root_ang_vel': data["root_ang_vel"].clone(),
                    'dof_pos': data["dof_pos"].clone(),
                    'dof_vel': data["dof_vel"].clone(),
                    'local_body_pos': data["local_body_pos"].clone(),
                    'root_pos_delta_local': data["root_pos_delta_local"].clone(),
                    'root_rot_delta_local': data["root_rot_delta_local"].clone(),
                    'num_frames': data["root_pos"].shape[0],
                }
                # Get root_pos_delta separately
                root_pos_delta = data.get("root_pos_delta")
                if root_pos_delta is not None:
                    next_data[motion_id]['root_pos_delta'] = root_pos_delta.clone()

                total_frames += data["root_pos"].shape[0]
                # Progress update every 1000 motions
                if (i + 1) % 1000 == 0:
                    print(f"[AsyncResample] Loaded {i+1}/{len(sampled_ids)} motions...", flush=True)

            self._async_resample_next_ids = sampled_ids
            self._async_resample_next_data = next_data

            t1 = time.time()
            load_time = load_start - t0
            copy_time = t1 - load_start
            print(f"[AsyncResample] Prepared {len(sampled_ids)} motions ({total_frames} frames) on CPU in {t1-t0:.1f}s (sample+load: {load_time:.1f}s, copy: {copy_time:.1f}s)", flush=True)

        except Exception as e:
            print(f"[AsyncResample] FAILED to prepare next subset: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self._async_resample_next_data = None
            self._async_resample_next_ids = None
    
    def _switch_to_next_subset_async(self):
        """Fast switch to the pre-loaded next subset.

        This is called when it's time to resample. It loads the pre-loaded
        CPU data to GPU, which is much faster than sampling and loading.
        """
        import time
        import gc

        if self._async_resample_next_data is None:
            print("[AsyncResample] WARNING: No pre-loaded data available, falling back to synchronous resample", flush=True)
            return False

        t0 = time.time()

        try:
            # _async_resample_next_ids is a list of motion IDs
            motion_ids = list(self._async_resample_next_ids) if self._async_resample_next_ids else []

            print(f"[AsyncResample] ========== SWITCHING to pre-loaded subset ({len(motion_ids)} motions) ==========", flush=True)

            # Load to GPU using the same logic as enable_resample_mode
            self._load_subset_to_gpu(motion_ids, self._async_resample_next_data)

            # Clear pre-loaded data and explicitly free memory
            self._async_resample_next_data.clear()
            self._async_resample_next_data = None
            self._async_resample_next_ids = None
            self._async_resample_ready_event.clear()

            # Force garbage collection to free CPU memory
            gc.collect()

            t1 = time.time()
            print(f"[AsyncResample] ========== SWITCH COMPLETED in {t1-t0:.2f}s ==========", flush=True)

            return True

        except Exception as e:
            print(f"[AsyncResample] FAILED to switch to next subset: {e}", flush=True)
            import traceback
            traceback.print_exc()
            # Clear data on error
            if hasattr(self, '_async_resample_next_data') and self._async_resample_next_data is not None:
                self._async_resample_next_data.clear()
            self._async_resample_next_data = None
            self._async_resample_next_ids = None
            self._async_resample_ready_event.clear()
            gc.collect()
            return False
    
    def check_and_resample_async(self, current_iteration: int):
        """Check if we should resample and do async switch if ready.

        This should be called from the training loop instead of _maybe_resample_motions.

        Args:
            current_iteration: Current training iteration

        Returns:
            True if resample happened, False otherwise
        """
        # IMPORTANT: Don't update _async_resample_last_iteration at the start!
        # This allows background thread to correctly trigger when:
        # (current_iteration + 1) % interval == 0
        # The iteration will be updated AFTER successful switch or at the end.

        # Check if it's time to resample
        if current_iteration > 0 and current_iteration % self._async_resample_interval == 0:
            print(f"[AsyncResample] Iteration {current_iteration}: Time to resample!", flush=True)

            # Try to use pre-loaded data first
            if self._async_resample_ready_event.is_set():
                print(f"[AsyncResample] Pre-loaded data is READY, switching...", flush=True)
                with self._async_resample_lock:
                    success = self._switch_to_next_subset_async()
                    if success:
                        # IMPORTANT: Update last to current prepare point in this cycle
                        # This ensures worker will find next prepare point correctly
                        # Formula: round down to nearest prepare_offset point
                        prepare_offset = self._async_resample_interval // 10
                        self._async_resample_last_iteration = (current_iteration // self._async_resample_interval) * self._async_resample_interval + prepare_offset
                        return True
                    else:
                        # Fall through to synchronous resample
                        pass
            else:
                print(f"[AsyncResample] Pre-loaded data NOT ready yet, waiting up to 5s...", flush=True)
                # Wait a bit for the async thread to finish (with timeout)
                self._async_resample_ready_event.wait(timeout=5.0)

                if self._async_resample_ready_event.is_set():
                    print(f"[AsyncResample] Data is now ready, switching...", flush=True)
                    with self._async_resample_lock:
                        success = self._switch_to_next_subset_async()
                        if success:
                            # IMPORTANT: Update last to current prepare point in this cycle
                            # Same logic as async success path above
                            prepare_offset = self._async_resample_interval // 10
                            self._async_resample_last_iteration = (current_iteration // self._async_resample_interval) * self._async_resample_interval + prepare_offset
                            return True
                else:
                    print(f"[AsyncResample] Data still NOT ready after 5s, falling back to synchronous resample", flush=True)
                    # Fall through to synchronous resample
                    pass

        # IMPORTANT: Only update last at prepare points (iteration % interval == prepare_offset)
        # This matches the worker trigger condition for async preparation
        # If async succeeded above, last was already updated. Otherwise, only update at prepare points.
        # Example (interval=200): only update when current is 20, 220, 420...
        prepare_offset = self._async_resample_interval // 10
        at_prepare_point = (current_iteration % self._async_resample_interval == prepare_offset)

        if at_prepare_point:
            self._async_resample_last_iteration = current_iteration
        # else: don't update, so worker can still find the prepare point

        return False
    
    def _load_subset_to_gpu(self, motion_ids: list, motion_data: dict):
        """Load pre-loaded motion data from CPU dict to GPU merged tensors.

        This is similar to enable_resample_mode but uses pre-loaded data.

        Args:
            motion_ids: List of motion IDs
            motion_data: Dict of motion_id -> data dict (CPU tensors)
        """
        import time
        import gc

        if not motion_ids:
            raise ValueError("motion_ids is empty in _load_subset_to_gpu")

        t0 = time.time()
        D = motion_data[motion_ids[0]]["dof_pos"].shape[-1]
        B = motion_data[motion_ids[0]]["local_body_pos"].shape[1]

        # Calculate total frames
        total_frames = sum(motion_data[mid]["num_frames"] for mid in motion_ids)

        print(f"[AsyncResample] Allocating GPU tensors for {total_frames} frames...", flush=True)

        # CRITICAL: Delete old GPU tensors FIRST to avoid double memory usage
        # This prevents GPU OOM when replacing large datasets
        if hasattr(self, '_gpu_root_pos'):
            del self._gpu_root_pos
        if hasattr(self, '_gpu_root_rot'):
            del self._gpu_root_rot
        if hasattr(self, '_gpu_root_vel'):
            del self._gpu_root_vel
        if hasattr(self, '_gpu_root_ang_vel'):
            del self._gpu_root_ang_vel
        if hasattr(self, '_gpu_dof_pos'):
            del self._gpu_dof_pos
        if hasattr(self, '_gpu_dof_vel'):
            del self._gpu_dof_vel
        if hasattr(self, '_gpu_local_body_pos'):
            del self._gpu_local_body_pos
        if hasattr(self, '_gpu_root_pos_delta_local'):
            del self._gpu_root_pos_delta_local
        if hasattr(self, '_gpu_root_rot_delta_local'):
            del self._gpu_root_rot_delta_local
        if hasattr(self, '_gpu_root_pos_delta'):
            del self._gpu_root_pos_delta

        # Force GPU cache flush before allocating new tensors
        if self._device.type == 'cuda':
            torch.cuda.empty_cache()

        # Allocate GPU tensors
        self._gpu_root_pos = torch.empty((total_frames, 3), device=self._device, dtype=torch.float32)
        self._gpu_root_rot = torch.empty((total_frames, 4), device=self._device, dtype=torch.float32)
        self._gpu_root_vel = torch.empty((total_frames, 3), device=self._device, dtype=torch.float32)
        self._gpu_root_ang_vel = torch.empty((total_frames, 3), device=self._device, dtype=torch.float32)
        self._gpu_dof_pos = torch.empty((total_frames, D), device=self._device, dtype=torch.float32)
        self._gpu_dof_vel = torch.empty((total_frames, D), device=self._device, dtype=torch.float32)
        self._gpu_local_body_pos = torch.empty((total_frames, B, 3), device=self._device, dtype=torch.float32)
        self._gpu_root_pos_delta_local = torch.empty((total_frames, 3), device=self._device, dtype=torch.float32)
        self._gpu_root_rot_delta_local = torch.empty((total_frames, 3), device=self._device, dtype=torch.float32)

        # Build motion-id mapping tables (must match synchronous resample path)
        self._motion_id_to_frame = {}
        self._motion_id_to_idx = {mid: i for i, mid in enumerate(motion_ids)}

        # Per-motion root_pos_delta is indexed by subset index (not global motion_id)
        self._gpu_root_pos_delta = torch.empty((len(motion_ids), 3), device=self._device, dtype=torch.float32)

        print(f"[AsyncResample] Copying data from CPU to GPU...", flush=True)

        # Copy data to GPU
        start = 0
        copy_start = time.time()
        for i, motion_id in enumerate(motion_ids):
            data = motion_data[motion_id]
            num_frames = data["num_frames"]
            end = start + num_frames

            self._gpu_root_pos[start:end] = data["root_pos"].to(self._device, non_blocking=True)
            self._gpu_root_rot[start:end] = data["root_rot"].to(self._device, non_blocking=True)
            self._gpu_root_vel[start:end] = data["root_vel"].to(self._device, non_blocking=True)
            self._gpu_root_ang_vel[start:end] = data["root_ang_vel"].to(self._device, non_blocking=True)
            self._gpu_dof_pos[start:end] = data["dof_pos"].to(self._device, non_blocking=True)
            self._gpu_dof_vel[start:end] = data["dof_vel"].to(self._device, non_blocking=True)
            self._gpu_local_body_pos[start:end] = data["local_body_pos"].to(self._device, non_blocking=True)
            self._gpu_root_pos_delta_local[start:end] = data["root_pos_delta_local"].to(self._device, non_blocking=True)
            self._gpu_root_rot_delta_local[start:end] = data["root_rot_delta_local"].to(self._device, non_blocking=True)

            # Per-motion root_pos_delta
            root_pos_delta = motion_data[motion_id].get("root_pos_delta")
            if root_pos_delta is not None:
                self._gpu_root_pos_delta[self._motion_id_to_idx[motion_id]] = root_pos_delta.to(self._device, non_blocking=True)
            else:
                self._gpu_root_pos_delta[self._motion_id_to_idx[motion_id]] = 0.0

            self._motion_id_to_frame[motion_id] = (start, num_frames)

            start = end

        copy_time = time.time() - copy_start
        print(f"[AsyncResample] CPU->GPU copy completed in {copy_time:.2f}s", flush=True)

        # Create fast lookup tables: motion_id -> frame start / subset index
        max_id = max(motion_ids)
        self._motion_start_frame = torch.zeros(max_id + 1, device=self._device, dtype=torch.long)
        for mid, (start_frame, _) in self._motion_id_to_frame.items():
            self._motion_start_frame[mid] = start_frame

        self._motion_id_to_idx_tensor = torch.full((max_id + 1,), -1, device=self._device, dtype=torch.long)
        for mid, idx in self._motion_id_to_idx.items():
            self._motion_id_to_idx_tensor[mid] = idx

        # Update resample-specific frame/length metadata (used by _calc_frame_blend)
        subset_num_frames = torch.tensor(
            [motion_data[mid]["num_frames"] for mid in motion_ids],
            device=self._device,
            dtype=torch.long,
        )
        subset_lengths = subset_num_frames.float() / self._motion_fps[torch.tensor(motion_ids, device=self._device, dtype=torch.long)]
        self._motion_num_frames_resample = subset_num_frames
        self._motion_lengths_resample = subset_lengths

        lengths_shifted = torch.roll(subset_num_frames, 1)
        lengths_shifted[0] = 0
        start_indices = lengths_shifted.cumsum(0)
        self._motion_start_idx = start_indices

        self._motion_start_idx_by_id = torch.full((max_id + 1,), -1, device=self._device, dtype=torch.long)
        for i, mid in enumerate(motion_ids):
            self._motion_start_idx_by_id[mid] = start_indices[i]

        # Cache dimensions/stats for debug and downstream assumptions
        self._resample_D = D
        self._resample_B = B
        self._resample_total_motions = len(motion_ids)
        self._resample_total_frames = total_frames

        # Update cached IDs
        self._loaded_subset_ids = set(motion_ids)
        self._loaded_subset_ids_tensor = torch.tensor(sorted(motion_ids), device=self._device, dtype=torch.long)
        
        print(f"[AsyncResample] Loaded {len(motion_ids)} motions ({total_frames} frames) to GPU")
