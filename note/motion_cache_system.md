# Motion Data Caching System 技术文档

## 概述

TWIST2 的运动数据缓存系统是一个**三级缓存架构**，旨在优化大规模运动数据集在强化学习训练中的访问性能。该系统通过磁盘、CPU内存和GPU显存的智能协作，支持超大规模数据集（百万级帧）的实时访问。

---

## 1. 系统架构

### 1.1 三级缓存层次结构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          训练环境 (Isaac Gym)                        │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     GPU Cache (最快)                         │   │
│  │  • 容量: gpu_cache_gib (默认 4GB)                           │   │
│  │  • 存储格式: float32                                         │   │
│  │  • LRU驱逐策略                                               │   │
│  │  • 仅缓存活跃帧                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                      │
│                              │ 非阻塞拷贝                            │
│                              │ (non_blocking=True)                  │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    CPU Cache / Storage                       │   │
│  │  • 容量: cpu_cache_gib (默认 50GB) 或 全量                  │   │
│  │  • 存储格式: float32 或 float16 (storage_dtype)            │   │
│  │  • LRU驱逐策略 (lazy_load模式)                               │   │
│  │  • 存储完整运动序列                                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ▲                                      │
│                              │ 按需加载                              │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      Disk (磁盘)                              │   │
│  │  • 存储格式: .pkl / .npz 文件                               │   │
│  │  • 存储完整运动数据                                          │   │
│  │  • 延迟最高，但容量无限                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `lazy_load` | `False` | 是否启用延迟加载（仅加载元数据，按需加载完整数据） |
| `gpu_cache_gib` | `4.0` | GPU缓存大小（GiB），设为0禁用 |
| `cpu_cache_gib` | `50.0` | CPU缓存大小（GiB），仅在lazy_load模式有效 |
| `storage_dtype` | `"float32"` | CPU存储精度，`"float16"` 可节省约50%内存 |
| `store_on_cpu` | `True` | 是否将数据存储在CPU而非直接存储在GPU |

---

## 2. 数据加载模式

### 2.1 标准模式 (lazy_load=False)

```
启动阶段:
┌─────────┐    完整加载    ┌─────────┐    存储到    ┌──────────────────┐
│  Disk   │ ────────────> │   CPU   │ ──────────> │ 全量运动数据     │
│ (.pkl)  │              └─────────┥            │ (float32/16)     │
└─────────┘                        │            └──────────────────┘
                                   │
                                   ▼
                            ┌──────────────────┐
                            │ _motion_root_pos │ (Concatenated)
                            │ _motion_root_rot │
                            │ _motion_dof_pos  │
                            │ ...              │
                            └──────────────────┘

训练阶段:
┌──────────────────┐    按需拷贝    ┌─────────┐
│   GPU Cache      │ <───────────── │   CPU   │
│ (活跃帧缓存)      │                │ Storage │
└──────────────────┘                └─────────┘
```

**特点:**
- 启动时加载所有运动数据到CPU内存
- 适合中小型数据集（<100GB）
- 访问速度最快（数据已在内存）

### 2.2 延迟加载模式 (lazy_load=True)

```
启动阶段:
┌─────────┐    仅元数据    ┌──────────────────┐
│  Disk   │ ────────────> │ _motion_files    │ (文件路径列表)
│ (.pkl)  │              │ _motion_fps      │ (帧率)
└─────────┘               │ _motion_num_frames│ (帧数)
                          │ _motion_lengths  │ (时长)
                          │ _cpu_motion_cache │ (空LRU缓存)
                          └──────────────────┘

训练阶段 (首次访问某motion):
┌─────────┐    完整加载    ┌─────────────────────────────┐
│  Disk   │ ────────────> │ _cpu_motion_cache[motion_id] │
│ (.pkl)  │              │ • root_pos, root_rot         │
└─────────┘               │ • dof_pos, dof_vel           │
                          │ • local_body_pos             │
                          │ (float32/16)                │
                          └─────────────────────────────┘
                                      ▲
                                      │ LRU驱逐
                                      │ (超出cpu_cache_gib时)
                                      ▼
                              (已加载的motion被逐出)

训练阶段 (GPU访问):
┌──────────────────┐    按需拷贝    ┌─────────────────────────────┐
│   GPU Cache      │ <─────────────> │ _cpu_motion_cache          │
│ (活跃帧缓存)      │                │ (LRU缓存中的motion数据)     │
└──────────────────┘                └─────────────────────────────┘
```

**特点:**
- 启动时仅加载元数据（文件路径、帧数、时长等）
- 首次访问某motion时才完整加载
- 适合超大型数据集（>200GB）
- 启动速度快，内存占用可控

---

## 3. GPU Cache 详细设计

### 3.1 初始化 (_init_gpu_cache)

```python
def _init_gpu_cache(self) -> None:
    # GPU Cache 启用条件:
    # 1. 设备类型为 CUDA
    # 2. store_on_cpu = True
    # 3. gpu_cache_gib > 0
    self._gpu_cache_enabled = (
        self._device.type == "cuda"
        and self._store_on_cpu
        and self._gpu_cache_gib > 0.0
    )

    # 计算可缓存帧数
    max_bytes = int(self._gpu_cache_gib * (1024 ** 3))
    bytes_per_frame = self._cache_bytes_per_frame()
    self._cache_max_frames = max(1, max_bytes // bytes_per_frame)

    # 预分配GPU内存
    self._cache_root_pos = torch.empty(...)    # (max_frames, 3)
    self._cache_root_rot = torch.empty(...)    # (max_frames, 4)
    self._cache_dof_pos = torch.empty(...)     # (max_frames, dof_dim)
    self._cache_local_body_pos = torch.empty(...)
    # ... 其他张量
```

### 3.2 缓存管理数据结构

```python
# 帧级偏移映射
# _cache_offset[motion_id] = 该motion在GPU cache中的起始帧偏移
# 若为-1，表示该motion未被缓存
self._cache_offset = torch.full((num_motions,), -1, dtype=torch.int64)

# 缓存长度映射
self._cache_len = torch.zeros(num_motions, dtype=torch.int64)

# 自由空间管理 (链表式空闲段管理)
self._cache_free_segments = []  # [(offset, length), ...]
```

### 3.3 缓存填充 (_populate_motion_cache)

```
调用流程:
calc_motion_frame()
    │
    ▼
检查 GPU cache 命中
    │
    ├─ 命中 ──> 直接从 cache 读取 (最快)
    │
    └─ 未命中 ──> prefetch(motion_ids)
                       │
                       ▼
              _populate_motion_cache(motion_id)
                       │
                       ├─ 分配 GPU 空间
                       │   │
                       │   ├─ 足够空间 ──> 直接分配
                       │   │
                       │   └─ 空间不足 ──> LRU驱逐旧motion
                       │
                       └─ CPU数据 → GPU (非阻塞拷贝)
                           non_blocking=True
```

---

## 4. CPU Cache (LRU) 详细设计

### 4.1 数据结构

```python
from collections import OrderedDict

# LRU缓存: OrderedDict保持插入顺序
# 最久未使用的item在头部，最新使用的在尾部
self._cpu_motion_cache: OrderedDict[int, dict] = OrderedDict()

# 当前使用量（字节）
self._cpu_cache_bytes_used = 0

# 最大容量（字节）
self._cpu_cache_max_bytes = int(cpu_cache_gib * (1024 ** 3))
```

### 4.2 缓存操作

#### 加载 (_ensure_motion_loaded)

```python
def _ensure_motion_loaded(self, motion_id: int):
    # 1. 检查是否已缓存
    if motion_id in self._cpu_motion_cache:
        # 移到末尾（标记为最近使用）
        self._cpu_motion_cache.move_to_end(motion_id, last=True)
        return

    # 2. 从磁盘加载
    data = self._load_motion_on_demand(motion_id)

    # 3. 计算大小
    size_bytes = sum(t.element_size() * t.numel()
                     for t in data.values() if isinstance(t, torch.Tensor))

    # 4. 驱逐直到有足够空间 (LRU策略)
    while (self._cpu_cache_bytes_used + size_bytes > self._cpu_cache_max_bytes
           and self._cpu_motion_cache):
        # popitem(last=False) 从头部移除最久未使用的item
        evict_id, evict_data = self._cpu_motion_cache.popitem(last=False)
        evict_size = ...
        self._cpu_cache_bytes_used -= evict_size

    # 5. 添加到缓存
    self._cpu_motion_cache[motion_id] = data
    self._cpu_cache_bytes_used += size_bytes
```

---

## 5. 完整数据流图

### 5.1 训练时的数据访问流程

```
训练请求: calc_motion_frame(motion_ids, motion_times)
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │  1. Lazy Load 预处理           │
                    │  for unique motion_id:        │
                    │    _ensure_motion_loaded(id)  │
                    └───────────────────────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
        ┌───────────────┐                     ┌───────────────┐
        │   CPU Cache   │                     │   CPU Cache   │
        │   已缓存      │                     │   未缓存      │
        │   直接使用    │                     │   从磁盘加载   │
        └───────────────┘                     └───────────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  2. GPU Cache 检查            │
                    │  cache_offset = _cache_offset │
                    │  cached_mask = (offset >= 0)  │
                    └───────────────────────────────┘
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
        ┌───────────────┐                     ┌───────────────┐
        │ GPU Cache命中 │                     │ GPU Cache未命中│
        │ 直接读取      │                     │ prefetch()    │
        │ (最快)        │                     │ CPU→GPU拷贝   │
        └───────────────┘                     └───────────────┘
                │                                       │
                └───────────────────┬───────────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  3. 帧插值                     │
                    │  frame_idx = floor(time * fps) │
                    │  blend = time * fps - idx      │
                    │  result = lerp(frame0, frame1,  │
                    │                  blend)        │
                    └───────────────────────────────┘
                                    │
                                    ▼
                            ┌───────────────┐
                            │  返回运动帧    │
                            │  (root_pos,   │
                            │   root_rot,   │
                            │   dof_pos,    │
                            │   ...)        │
                            └───────────────┘
```

### 5.2 内存使用估算

#### 单帧大小计算 (_cache_bytes_per_frame)

```python
def _cache_bytes_per_frame(self) -> int:
    bytes_per_frame = 0
    # root_pos: float32 * 3 = 12 bytes
    bytes_per_frame += 3 * 4
    # root_rot: float32 * 4 = 16 bytes
    bytes_per_frame += 4 * 4
    # root_vel: float32 * 3 = 12 bytes
    bytes_per_frame += 3 * 4
    # root_ang_vel: float32 * 3 = 12 bytes
    bytes_per_frame += 3 * 4
    # dof_pos: float32 * dof_dim (e.g., 19) = 76 bytes
    bytes_per_frame += dof_dim * 4
    # dof_vel: float32 * dof_dim = 76 bytes
    bytes_per_frame += dof_dim * 4
    # local_body_pos: float32 * num_bodies * 3 (e.g., 38 * 3 * 4 = 456 bytes)
    bytes_per_frame += num_bodies * 3 * 4
    # root_pos_delta_local: float32 * 3 = 12 bytes
    bytes_per_frame += 3 * 4
    # root_rot_delta_local: float32 * 3 = 12 bytes
    bytes_per_frame += 3 * 4

    # 总计: ~700-800 bytes/frame
    return bytes_per_frame
```

#### 容量规划示例

| GPU Cache | 可缓存帧数 | 假设120fps | 假设30fps |
|-----------|-----------|------------|-----------|
| 4 GB      | ~5,000,000 | ~11.6小时  | ~46小时   |
| 8 GB      | ~10,000,000| ~23小时    | ~92小时   |

---

## 6. 使用场景与配置建议

### 6.1 小型数据集 (<50GB)

```yaml
# 推荐配置
lazy_load: False
gpu_cache_gib: 4.0
storage_dtype: "float32"
```

**原因**: 标准模式将所有数据加载到内存，访问速度最快。

### 6.2 中型数据集 (50-200GB)

```yaml
# 推荐配置
lazy_load: False
gpu_cache_gib: 8.0
storage_dtype: "float16"  # 节省50%内存
```

**原因**: 使用float16存储可显著减少内存占用。

### 6.3 超大型数据集 (>200GB)

```yaml
# 推荐配置
lazy_load: True
gpu_cache_gib: 8.0
cpu_cache_gib: 100.0
storage_dtype: "float16"
```

**原因**: 延迟加载避免一次性加载全部数据，CPU缓存优化重复访问。

### 6.4 多GPU DDP训练

```yaml
# 在每个rank上均匀分配motion files
# torchrun会自动处理motion list的分片
```

**注意**: DDP模式下，每个rank只加载分配给它的motion子集。

---

## 7. 性能优化要点

### 7.1 非阻塞内存传输

```python
# CPU → GPU 拷贝使用 non_blocking=True
loaded.to(self._device, non_blocking=True)
```

### 7.2 批处理

```python
# prefetch() 批量预取多个motion
def prefetch(self, motion_ids: torch.Tensor) -> None:
    uniq_ids = torch.unique(motion_ids)
    for mid in uniq_ids:
        self._populate_motion_cache(mid)
```

### 7.3 精度优化

| 精度 | 显存占用 | 精度损失 | 适用场景 |
|------|---------|---------|---------|
| float32 | 1x | 无 | 默认 |
| float16 | 0.5x | 极小 | 推荐用于存储 |
| bfloat16 | 0.5x | 极小 | Ampere+ GPU推荐 |

---

## 8. 故障排查

### 8.1 OOM (Out of Memory)

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
1. 减少 `gpu_cache_gib`
2. 减少 `num_envs`
3. 启用 `storage_dtype: "float16"`
4. 启用 `lazy_load: True`

### 8.2 加载速度慢

**症状**: 启动时间过长

**解决方案**:
1. 启用 `lazy_load: True`
2. 检查磁盘IO性能
3. 减小数据集规模

### 8.3 训练卡顿

**症状**: FPS不稳定

**解决方案**:
1. 增加 `gpu_cache_gib`
2. 增加 `cpu_cache_gib` (lazy_load模式)
3. 检查是否有频繁的cache驱逐

---

## 9. 代码文件位置

- **核心实现**: `pose/pose/utils/motion_lib_pkl.py`
- **配置文件**: `legged_gym/envs/*/g1_*_config.py`
- **数据配置**: `legged_gym/motion_data_configs/*.yaml`

---

## 10. 总结

TWIST2的运动缓存系统通过三级架构实现了:
1. **无限的数据集规模支持** (lazy_load + 磁盘)
2. **极快的训练时访问** (GPU cache)
3. **灵活的内存-性能权衡** (可配置的cache大小)

理解这个系统对于优化大规模数据集的训练性能至关重要。
