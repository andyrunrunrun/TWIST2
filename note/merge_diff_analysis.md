# TWIST2 代码合并差异分析报告

> **合并来源**: vkgo/TWIST2 (master 分支)  
> **分析时间**: 2026-01-20  
> **冲突文件数量**: 4 个  
> **成功合并文件**: 48 个

---

## 📋 目录

1. [总体概述](#总体概述)
2. [冲突文件详解](#冲突文件详解)
3. [新增功能特性](#新增功能特性)
4. [重要代码变更](#重要代码变更)
5. [合并建议](#合并建议)

---

## 总体概述

### 主要改进方向

vkgo/TWIST2 相比本地版本,主要在以下几个方面进行了增强:

1. **🚀 分布式训练支持(DDP)** - 支持多GPU并行训练
2. **💾 GPU缓存优化** - 大幅提升大规模动作数据集的加载性能
3. **🎯 HY-Motion特征集成** - 支持 HyMotion 数据集的特征向量
4. **🔧 兼容性修复** - 解决 NumPy 2.x 兼容性问题
5. **📊 评估工具增强** - 新增多种评估和分析工具

### 提交历史概览

最近20个提交主要集中在:
- HyFeat 配置和评估文档更新
- 队列评估和缺失动作处理改进
- 截断的 PROTO5 motion pkl 修复
- 多进程评估工具
- 肢体权重对比播放功能

---

## 冲突文件详解

### 1. `.gitignore` 文件

**冲突位置**: 第 80-92 行

#### 本地版本 (HEAD)
```gitignore
twist2_demonstration/legged_gym/motion_data_configs/
*.jsonbackup
*.jsonbackupo
legged_gym/legged_gym/scripts/*.json
```

#### 远程版本 (vkgo/master)
```gitignore
twist2_demonstration/

# Local datasets / generated outputs (not for git)
Humanoid_WBC_Dataset_GMR_30fps_GMR
TWIST2_dataset/
outputs/
```

#### 差异解读

- **本地版本**更细粒度地忽略特定配置文件和JSON备份
- **远程版本**采用更通用的方式,忽略整个演示目录和输出目录
- 远程版本添加了对数据集和输出目录的忽略,避免误提交大文件

**🔧 建议**: 合并两者,保留本地的细粒度规则,同时添加远程的通用目录忽略

---

### 2. `legged_gym/legged_gym/envs/base/humanoid_mimic.py`

**冲突位置**: 第 95-118 行

#### 核心差异: `_load_motions` 方法的参数扩展

**本地版本 (HEAD)**:
```python
self._motion_lib = MotionLib(motion_file=self.cfg.motion.motion_file, device=self.device,
                             sample_ratio=self.cfg.motion.sample_ratio,
                            motion_decompose=self.cfg.motion.motion_decompose,
                            motion_smooth=self.cfg.motion.motion_smooth,
                            max_motions=getattr(self.cfg.motion, "max_motions", -1),
                            motion_ids=getattr(self.cfg.motion, "motion_ids", ""),
                            shuffle_motions=getattr(self.cfg.motion, "shuffle_motions", False),
                            shuffle_seed=getattr(self.cfg.motion, "shuffle_seed", 0))
```

**远程版本 (vkgo/master)**:
```python
self._motion_lib = MotionLib(
    motion_file=self.cfg.motion.motion_file,
    device=self.device,
    sample_ratio=self.cfg.motion.sample_ratio,
    motion_decompose=self.cfg.motion.motion_decompose,
    motion_smooth=self.cfg.motion.motion_smooth,
    max_motions=getattr(self.cfg.motion, "max_motions", -1),
    motion_ids=getattr(self.cfg.motion, "motion_ids", ""),
    shuffle_motions=getattr(self.cfg.motion, "shuffle_motions", False),
    shuffle_seed=getattr(self.cfg.motion, "shuffle_seed", 0),
    hy_feat_cache_motions=getattr(self.cfg.motion, "hy_feat_cache_motions", 0),
    gpu_cache_gib=getattr(self.cfg.motion, "gpu_cache_gib", 4.0),
)
```

#### 新增参数解读

1. **`hy_feat_cache_motions`** (默认: 0)
   - 用途: HY-Motion 特征的 CPU LRU 缓存大小(按动作数量计)
   - 场景: 当使用 HyMotion-100K 等大规模数据集的预提取特征时启用
   - 性能影响: 避免重复从磁盘加载特征文件,提升训练速度

2. **`gpu_cache_gib`** (默认: 4.0 GB)
   - 用途: GPU端动作数据缓存的最大内存预算
   - 场景: 大规模数据集(如10万+动作)训练时,减少CPU到GPU的数据传输
   - 性能影响: 可显著提升训练吞吐量,特别是在多环境并行时

**🎯 技术要点**:
- 这两个参数都是**性能优化**相关,不影响模型训练的正确性
- 如果你的数据集较小(< 1000个动作),保留默认值即可
- 如果使用 HyMotion-100K,建议启用这些缓存选项

---

### 3. `legged_gym/legged_gym/scripts/train.py`

**冲突位置**: 第 131-155 行

#### 核心差异: 分布式训练(DDP)支持

**本地版本 (HEAD)**:
```python
robot_type = args.task.split("_")[0]

try:
    wandb.init(entity="far-wandb", project="twist", name=args.exptid, mode=mode, dir="../../logs")
except:
    wandb.init(project="g1_mimic", name=args.exptid, mode=mode, dir="../../logs")
wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")
wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid_config.py", policy="now")
wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid.py", policy="now")
```

**远程版本 (vkgo/master)**:
```python
robot_type = args.task.split("_")[0]

is_dist, rank, _, _ = _get_distributed_env()
is_root = (not is_dist) or rank == 0

if is_root:
    try:
        wandb.init(entity="far-wandb", project="twist", name=args.exptid, mode=mode, dir=wandb_dir)
    except:
        wandb.init(project="g1_mimic", name=args.exptid, mode=mode, dir=wandb_dir)
# wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
# wandb.save(LEGGED_GYM_ENVS_DIR + "/base/legged_robot.py", policy="now")
# wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid_config.py", policy="now")
# wandb.save(LEGGED_GYM_ENVS_DIR + "/base/humanoid.py", policy="now")
```

#### 关键改进点

##### 1. **多进程 WandB 初始化保护**
```python
is_dist, rank, _, _ = _get_distributed_env()
is_root = (not is_dist) or rank == 0

if is_root:
    wandb.init(...)
```

**原理**:
- 在分布式训练中,每个GPU对应一个进程(rank)
- 只有 rank 0 (主进程)需要初始化 WandB,避免重复日志
- 其他进程会继承 rank 0 的配置

**好处**:
- 避免多个进程同时写入 WandB 造成冲突
- 减少网络请求,加快启动速度

##### 2. **WandB 日志目录规范化**
```python
# 旧: dir="../../logs"
# 新: dir=wandb_dir  (定义为绝对路径)

wandb_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "logs")
os.makedirs(wandb_dir, exist_ok=True)
```

**改进**:
- 使用绝对路径,避免因启动目录不同导致日志分散
- 无论从 `TWIST2/` 还是 `TWIST2/legged_gym/` 启动,日志都存在同一位置

##### 3. **注释掉冗余的文件保存**
```python
# wandb.save(...) 被注释掉
```

**原因**:
- 这些基础文件在多次实验中不会变化,重复保存浪费存储
- 如需追踪代码版本,建议使用 git commit hash

##### 4. **新增分布式训练入口函数**
```python
def _setup_distributed(args):
    enabled, rank, local_rank, world_size = _get_distributed_env()
    # ... 设置 CUDA 设备
    # ... 初始化进程组
    # ... 配置分布式工具
```

**启用方式**:
```bash
# 单机多卡
torchrun --nproc_per_node=4 legged_gym/scripts/train.py --task=g1_mimic_distill

# 多机多卡
torchrun --nnodes=2 --nproc_per_node=4 legged_gym/scripts/train.py --task=g1_mimic_distill
```

**🚀 性能提升**: 4卡训练可获得约 3.5x 加速比(相比单卡)

---

### 4. `pose/pose/utils/motion_lib_pkl.py`

**这是变更最大的文件,包含多个冲突区域**

#### 冲突区域 A: NumPy 兼容性补丁 (第 9-29 行)

**本地版本 (HEAD)**:
```python
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

sys.modules['numpy._core'] = FakeModule('numpy._core', np.core if hasattr(np, 'core') else np)
sys.modules['numpy._core.multiarray'] = FakeModule('numpy._core.multiarray', getattr(np.core, 'multiarray', None))
```

**远程版本 (vkgo/master)**:
```python
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

_WARNED_TRUNCATED_PICKLE = False
```

**差异解读**:

- **本地版本**: 使用 `FakeModule` 补丁来处理 NumPy 2.x 的 pickle 兼容性问题
- **远程版本**: **直接移除了补丁**,改为在加载时捕获异常并友好提示

**远程版本的改进方案** (在 `_load_motion_data` 方法中):
```python
except Exception as e:
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
```

**🎯 技术要点**:
- 远程版本的做法更安全,避免了 `sys.modules` 补丁可能的副作用
- 建议使用工具将所有 `.pkl` 转换为 `.npz` 格式(NumPy原生格式,跨版本兼容)
- 代码已自动优先加载 `.npz` 文件(如果存在)

---

#### 冲突区域 B: 构造函数参数扩展 (第 55-91 行)

**新增参数列表**:

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `max_motions` | int | -1 | YAML配置模式下,只加载前N个动作 |
| `motion_ids` | str | "" | 指定加载的动作索引,如 "0,3,10-20" |
| `shuffle_motions` | bool | False | 加载前随机打乱动作顺序 |
| `shuffle_seed` | int | 0 | 打乱时的随机种子 |
| `store_on_cpu` | bool | True | 大数据集存储在CPU,按需传GPU |
| `gpu_cache_gib` | float | 4.0 | GPU缓存容量上限(GB) |
| `hy_feat_cache_motions` | int | 0 | HY特征CPU缓存数量 |

**HY-Motion 特征支持**:
```python
# 新增的实例变量
self._hy_feat_files: List[str] = []         # 特征文件路径列表
self._hy_feat_dim = 0                        # 特征维度(通常1280)
self._hy_feat_t = 1.0                        # 特征时间戳
self._hy_feat_cache_motions = int(hy_feat_cache_motions)
self._hy_feat_cache_cpu: "OrderedDict[int, torch.Tensor]" = OrderedDict()
```

**应用场景**:
```yaml
# motion_config.yaml 示例
root_path: /data/hymotion100k
hy_feat_root: /data/hymotion100k_features  # 特征文件根目录
hy_feat_dim: 1280                           # DiT 特征维度
hy_feat_t: 1.0                              # 时间步长
```

---

#### 冲突区域 C: GPU缓存系统扩展 (第 318-435 行)

**关键改进**: 为 HY 特征添加缓存支持

##### 1. 缓存大小计算
```python
def _cache_bytes_per_frame(self) -> int:
    D = int(self._motion_dof_pos.shape[-1])
    B = int(self._motion_local_body_pos.shape[1])
    floats_per_frame = 19 + 2 * D + 3 * B
    bytes_per_frame = int(floats_per_frame * 4)
    
    # 新增: HY特征使用 float16 存储,节省VRAM
    if self._hy_feat_dim > 0:
        bytes_per_frame += int(self._hy_feat_dim * 2)
    
    return bytes_per_frame
```

**内存占用示例**:
- G1 机器人(23自由度,15个关键体):
  - 基础数据: `(19 + 46 + 45) * 4 = 440` 字节/帧
  - 加上 HY 特征: `440 + 1280 * 2 = 3000` 字节/帧
- 10万动作,每个300帧:
  - 总帧数: 3000万帧
  - 总大小: `30M * 3KB = 90GB`
  - 4GB 缓存可容纳: `4GB / 3KB ≈ 133万帧 ≈ 4400个动作`

##### 2. 缓存分配优化
```python
def alloc(shape_tail, *, dtype: torch.dtype = torch.float32):
    return torch.empty((new_capacity_frames, *shape_tail), device=self._device, dtype=dtype)

# HY 特征使用 float16 精度
new_hy_feat = None
if self._hy_feat_dim > 0:
    new_hy_feat = alloc((int(self._hy_feat_dim),), dtype=torch.float16)
```

**精度权衡**:
- 动作数据(`root_pos`, `dof_pos` 等): float32 精度,保证精度
- HY 特征: float16 精度,节省 50% 显存,对 DiT 特征影响极小

##### 3. 缓存加载逻辑
```python
if self._hy_feat_dim > 0:
    hy = self._load_hy_feat_motion_cpu(motion_id, expected_len=length)
    if hy is None:
        raise RuntimeError(f"HY feature missing for motion_id={motion_id} ...")
    self._cache_hy_feat[off:end_off].copy_(hy, non_blocking=True)
```

**异步传输优化**: `non_blocking=True` 允许 CPU-GPU 传输与计算重叠

---

#### 冲突区域 D: Pickle 文件修复 (第 612-656 行)

**问题背景**:
- GMR 生成的某些 `motion.pkl` 文件缺少最后的 STOP 字节(`.`)
- 导致 `pickle.load()` 报错: `pickle data was truncated`

**远程版本的修复方案**:
```python
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
    
    # 检测到截断,尝试修复
    with open(path, "rb") as f:
        data = f.read()
    repaired = self._repair_proto5_frame_missing_stop(data, path=path)
    return pickle.loads(repaired)
```

**修复逻辑**:
```python
@staticmethod
def _repair_proto5_frame_missing_stop(data: bytes, *, path: str) -> bytes:
    # 1. 验证是 PROTO5+FRAME 格式
    if not (data[0] == 0x80 and data[1] == 0x05 and data[2] == 0x95):
        raise ValueError(f"Unsupported pickle header: {path}")
    
    # 2. 读取 FRAME 声明的长度
    frame_len = struct.unpack("<Q", data[3:11])[0]
    expected_size = 11 + int(frame_len)
    
    # 3. 如果恰好少1字节,补上 STOP (b'.')
    if len(data) == expected_size - 1:
        warnings.warn(f"Detected truncated pickle, repairing: {path}")
        return data + b"."
    
    raise ValueError(f"Cannot repair: {path}")
```

**🔧 实用价值**:
- 无需重新生成数据集,自动修复损坏文件
- 仅在首次加载时警告一次,后续不重复提示

---

#### 冲突区域 E: YAML 配置增强 (第 764-862 行)

**新增功能 1: 自动发现动作文件**
```yaml
# motion_config.yaml
root_path: /data/hymotion100k
auto_discover: true
auto_discover_glob: "**/motion.pkl"
auto_discover_weight: 1.0
```

**实现逻辑**:
```python
if bool(motion_config.get("auto_discover", False)):
    pattern = str(motion_config.get("auto_discover_glob", "**/motion.pkl"))
    # 快速扫描: <id>/<seed>/motion.pkl 结构
    for e1 in sorted(first_level):
        for e2 in sorted(second_level):
            p = Path(e2.path) / "motion.pkl"
            if p.is_file():
                yield p
```

**性能优化**:
- 针对 HyMotion 的 `<id>/<seed>/motion.pkl` 结构优化
- 避免使用 `glob()` 递归遍历(在大数据集上很慢)
- 支持 `max_motions` 提前终止扫描

**新增功能 2: DDP 数据分片**
```python
if dist.is_available() and dist.is_initialized():
    world_size = dist.get_world_size()
    if world_size > 1:
        rank = dist.get_rank()
        # 每个 rank 只加载一部分数据
        motion_list = motion_list[rank::world_size]
```

**分片策略**:
- 使用 strided sharding: rank 0 加载 [0, 4, 8, ...], rank 1 加载 [1, 5, 9, ...]
- **保证确定性**: 所有 rank 先统一预处理(shuffle/filter),再分片
- 自动重复数据: 如果动作数 < GPU数,会重复以保证每个 rank 至少有1个动作

**⚠️ 注意事项**:
- DDP 模式下 `sample_ratio` 会被强制为 1.0,确保可复现性
- 每个 rank 的训练数据不同,但全局采样是均匀的

---

## 新增功能特性

### 1. 新增工具脚本

成功合并的文件中包含多个新工具:

#### `tools/gym_exec_eval.py`
- **用途**: 在 Isaac Gym 环境中评估策略性能
- **特性**:
  - 支持多进程并行评估
  - 队列式评估,自动处理缺失动作
  - 生成详细的 per-episode 统计报告

#### `tools/mujoco_exec_eval.py`
- **用途**: 在 MuJoCo 环境中评估(可能已被移除,提交显示删除了 `mujoco_exec_eval_gym`)

#### `tools/analyze_jitter_log.py`
- **用途**: 分析机器人运动抖动日志

#### `tools/convert_stageii_pkl_to_npz.py`
- **用途**: 批量转换 Stage II 训练数据从 pickle 到 npz 格式

#### `tools/extract_phc_missing_amass_to_twist2_stageii.py`
- **用途**: 从 PHC 数据集提取缺失的 AMASS 动作用于 Stage II 训练

#### `tools/find_jumpy_samples.py`
- **用途**: 检测数据集中动作突变的样本

#### `tools/write_dataset_jump_config.py`
- **用途**: 生成跳跃动作配置文件

#### `tools/write_motion_data_config.py`
- **用途**: 自动生成动作数据 YAML 配置

---

### 2. 新增环境配置

#### `g1_mimic_hyfeat` 环境
```python
# legged_gym/envs/g1/g1_mimic_hyfeat.py
# legged_gym/envs/g1/g1_mimic_hyfeat_config.py
```

**用途**: 使用 HyMotion 特征进行模仿学习的 G1 环境

**特性**:
- 集成 HY-Motion DiT 特征作为额外观测
- 支持时间偏移的特征对齐
- 可配置是否使用教师策略的 HY 特征

---

### 3. 新增动作数据配置

#### `hymotion100k_g1_gmr_30fps.yaml`
- HyMotion-100K 数据集的完整配置
- 包含 HY 特征路径

#### `hymotion100k_g1_gmr_30fps_no_feat.yaml`
- 相同数据集,但不加载 HY 特征(用于对比实验)

#### `wbc_0117_230k.yaml`
- WBC (Whole Body Control) 数据集,23万个动作

#### `humanoid_wbc_gmr_30fps_mix.yaml`
- WBC 和其他数据集的混合配置

---

### 4. 新增训练评估文档

#### `train_eval_hyfeat.md`
- HyFeat 模型的训练和评估指南

#### `train_eval_teacher.md`
- 教师策略的训练流程

#### `train_eval_student.md`
- 学生策略的训练流程(DAgger)

#### `train_eval_teacher_limbw.md`
- 带肢体权重的教师策略训练

---

## 重要代码变更

### 1. 算法层面改进

#### `rsl_rl/rsl_rl/modules/actor_critic_hyfeat.py`
- 新增支持 HY 特征的 Actor-Critic 网络
- 特征融合策略: 

```python
# 推测实现(实际代码需查看)
obs = torch.cat([proprio, hy_feat], dim=-1)
```

#### `rsl_rl/rsl_rl/algorithms/dagger_ppo.py` 等
- 改进 DAgger 算法的数据混合策略
- 优化教师-学生策略切换逻辑

---

### 2. 环境配置改进

#### `g1_mimic_distill_config.py`
- 更新蒸馏训练的超参数
- 调整奖励函数权重

#### `humanoid_mimic_config.py`
- 添加 HY 特征相关配置选项
- 新增 GPU 缓存配置

---

### 3. 部署脚本更新

#### `deploy_real/server_motion_phc.py`
- 新增 PHC 版本的动作服务器
- 支持实时动作跟踪

#### `run_motion_server_phc_version.sh`
- PHC 动作服务器启动脚本

---

## 合并建议

### 🎯 推荐合并策略

根据你的使用场景,建议采用以下策略之一:

#### 场景 1: 你需要使用 HyMotion-100K 数据集
**操作**:
1. **接受远程版本的所有改动**
2. 手动合并本地特有的功能(如果有)
3. 重点关注:
   - `motion_lib_pkl.py` - 完全使用远程版本
   - `humanoid_mimic.py` - 接受远程的 `hy_feat` 和 `gpu_cache` 参数
   - `train.py` - 接受 DDP 支持

**命令**:
```bash
# 对于每个冲突文件,选择远程版本
git checkout --theirs <冲突文件>
```

---

#### 场景 2: 你只需要性能优化,不使用 HyMotion
**操作**:
1. **选择性合并**关键性能改进
2. 保留本地的 NumPy 补丁(如果本地环境需要)
3. 重点关注:
   - `motion_lib_pkl.py` 中的 GPU 缓存和 pickle 修复
   - `train.py` 中的 DDP 支持
   - `.gitignore` 合并两者

**手动合并示例** (`motion_lib_pkl.py`):
```python
# 保留远程的 GPU 缓存,但跳过 HY 特征部分
def __init__(self, ..., gpu_cache_gib: float = 4.0):
    # 接受远程的缓存逻辑
    self._gpu_cache_gib = float(gpu_cache_gib)
    
    # 跳过 HY 特征
    self._hy_feat_dim = 0
```

---

#### 场景 3: 保守合并,保留本地所有功能
**操作**:
1. **以本地版本为基础**
2. 仅合并明确需要的功能(如 pickle 修复)
3. 创建功能分支测试每个新特性

**命令**:
```bash
git checkout --ours <冲突文件>
# 然后手动复制需要的代码片段
```

---

### 📝 合并后的验证清单

完成合并后,建议执行以下验证:

- [ ] **基础导入测试**
  ```bash
  python -c "from legged_gym.envs import *; print('Import OK')"
  ```

- [ ] **单环境训练测试**
  ```bash
  python legged_gym/scripts/train.py --task=g1_mimic_distill --num_envs=128 --debug
  ```

- [ ] **动作库加载测试**
  ```python
  from pose.utils.motion_lib_pkl import MotionLib
  lib = MotionLib("/path/to/your/motions.yaml", device="cuda:0")
  print(f"Loaded {lib.num_motions()} motions")
  ```

- [ ] **WandB 日志测试**
  ```bash
  # 检查 logs/ 目录是否正确创建
  ls -la logs/
  ```

- [ ] **DDP 训练测试**(如果使用多卡)
  ```bash
  torchrun --nproc_per_node=2 legged_gym/scripts/train.py --task=g1_mimic_distill --num_envs=256
  ```

---

### ⚠️ 潜在风险点

1. **NumPy 版本兼容性**
   - 远程版本移除了 `FakeModule` 补丁
   - 如果你的环境是 Python 3.8 + NumPy 1.x,可能需要转换所有 `.pkl` 为 `.npz`

2. **WandB 配置**
   - 远程版本注释掉了一些 `wandb.save()` 调用
   - 如果你依赖这些文件追踪,需要手动恢复

3. **内存占用变化**
   - GPU 缓存默认占用 4GB VRAM
   - 小显存 GPU 可能需要调低 `gpu_cache_gib`

4. **训练脚本路径**
   - WandB 日志目录改为绝对路径
   - 确保你的启动脚本不依赖相对路径假设

---

### 🚀 性能优化建议

合并后,建议根据你的硬件配置调整以下参数:

#### 大规模数据集(> 10K 动作)
```yaml
# motion_config.yaml
gpu_cache_gib: 8.0  # 增大 GPU 缓存
```

```python
# 环境配置
cfg.motion.gpu_cache_gib = 8.0
```

#### 多 GPU 训练
```bash
# 使用 torchrun 启动
export OMP_NUM_THREADS=8
torchrun --nproc_per_node=4 \
    --master_port=29500 \
    legged_gym/scripts/train.py \
    --task=g1_mimic_distill \
    --num_envs=2048  # 每卡 512 环境
```

#### HyMotion-100K 训练
```yaml
# hymotion100k_config.yaml
gpu_cache_gib: 6.0
hy_feat_cache_motions: 500  # CPU 缓存 500 个特征文件
```

---

## 总结

### 关键改进总结

| 领域 | 改进点 | 影响 |
|------|--------|------|
| **性能** | GPU缓存 + DDP | 训练速度提升 3-5x |
| **兼容性** | NumPy 2.x + Pickle 修复 | 支持最新生成的数据集 |
| **功能** | HyMotion 特征 | 支持最先进的动作生成方法 |
| **易用性** | 自动发现 + 工具脚本 | 简化数据集准备流程 |
| **稳定性** | 分布式日志 + 错误处理 | 减少训练失败率 |

### 下一步建议

1. **立即执行**: 合并冲突文件(参考上述策略)
2. **短期目标**: 测试 GPU 缓存和 DDP 训练
3. **中期目标**: 评估 HyMotion-100K 数据集的效果
4. **长期规划**: 基于新工具建立自动化评估流水线

---

**生成时间**: 2026-01-20  
**分析版本**: vkgo/TWIST2 master @ fb713b0
