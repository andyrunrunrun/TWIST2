# PyTorch 分布式训练(DDP)完整教程

> **基于 TWIST2 项目的实战指南**  
> **难度**: 中级 → 高级  
> **前置知识**: PyTorch 基础、多进程概念

---

## 📚 目录

1. [DDP 核心概念](#ddp-核心概念)
2. [TWIST2 中的 DDP 实现解析](#twist2-中的-ddp-实现解析)
3. [从零开始写 DDP 训练](#从零开始写-ddp-训练)
4. [常见问题与调试](#常见问题与调试)
5. [性能优化技巧](#性能优化技巧)

---

## DDP 核心概念

### 什么是 DDP?

**DistributedDataParallel (DDP)** 是 PyTorch 的官方分布式训练方案,核心思想:

```
单机单卡:  [GPU 0] ← 所有数据 ← 模型
           ↓
          慢!

多机多卡:  [GPU 0]  [GPU 1]  [GPU 2]  [GPU 3]
           ↓        ↓        ↓        ↓
          数据A    数据B    数据C    数据D
           ↓        ↓        ↓        ↓
          模型副本0 模型副本1 模型副本2 模型副本3
           └────────┴────────┴────────┘
                    ↓
              同步梯度,更新参数
                    ↓
                   快 3-4x!
```

### 关键术语

| 术语 | 含义 | 示例 |
|------|------|------|
| **World Size** | 总进程数(通常=GPU数) | 4卡训练 → world_size=4 |
| **Rank** | 进程的全局ID | Rank 0, 1, 2, 3 |
| **Local Rank** | 单机内的进程ID | 单机4卡: local_rank = 0,1,2,3 |
| **Master** | Rank 0 进程,负责日志等 | 只有 rank 0 打印日志 |
| **Process Group** | 通信组,进程间同步梯度 | NCCL backend for GPU |

### DDP 执行流程

```python
# 1. 每个进程启动,绑定到一个GPU
# Rank 0 → GPU 0
# Rank 1 → GPU 1
# Rank 2 → GPU 2
# Rank 3 → GPU 3

# 2. 每个进程加载模型
model = MyModel().to(device)

# 3. 包装成 DDP
model = DDP(model, device_ids=[local_rank])

# 4. 数据并行
# Rank 0: batch[0:32]
# Rank 1: batch[32:64]
# Rank 2: batch[64:96]
# Rank 3: batch[96:128]

# 5. 前向传播(独立)
loss = model(data)

# 6. 反向传播
loss.backward()  # DDP自动同步梯度!

# 7. 优化器更新(每个进程独立,但参数相同)
optimizer.step()
```

---

## TWIST2 中的 DDP 实现解析

### 架构总览

TWIST2 的 DDP 实现分为 4 个层次:

```
legged_gym/scripts/train.py
  ├─ _get_distributed_env()      # 读取环境变量
  ├─ _setup_distributed()        # 初始化进程组
  └─ train()                      # 主训练逻辑
       ↓
legged_gym/gym_utils/task_registry.py
  └─ make_env()                   # 调整每个rank的seed
       ↓
rsl_rl/utils/utils.py
  ├─ maybe_wrap_ddp()             # 包装模型为DDP
  ├─ reduce_sum/mean/...          # 梯度/统计量同步
  └─ ForwardingDistributedDataParallel  # 自定义DDP包装
       ↓
pose/utils/motion_lib_pkl.py
  └─ _fetch_motion_files()        # 数据分片给每个rank
```

---

### 层次 1: 环境变量检测

**文件**: `legged_gym/scripts/train.py`

```python
def _get_distributed_env():
    """从 torchrun 设置的环境变量中读取分布式信息"""
    
    def _get_int(name: str, default: int) -> int:
        val = os.environ.get(name, None)
        if val is None:
            return default
        try:
            return int(val)
        except ValueError:
            return default
    
    # torchrun 会自动设置这些环境变量
    world_size = _get_int("WORLD_SIZE", 1)  # 总进程数
    rank = _get_int("RANK", 0)              # 全局进程ID
    local_rank = _get_int("LOCAL_RANK", 0)  # 本地进程ID
    
    # 如果 world_size > 1,说明是分布式训练
    enabled = world_size > 1
    
    return enabled, rank, local_rank, world_size
```

**工作原理**:

```bash
# 当你运行:
torchrun --nproc_per_node=4 train.py

# torchrun 会启动 4 个进程,每个进程的环境变量:
# 进程 0: RANK=0, LOCAL_RANK=0, WORLD_SIZE=4
# 进程 1: RANK=1, LOCAL_RANK=1, WORLD_SIZE=4
# 进程 2: RANK=2, LOCAL_RANK=2, WORLD_SIZE=4
# 进程 3: RANK=3, LOCAL_RANK=3, WORLD_SIZE=4
```

---

### 层次 2: 初始化进程组

**文件**: `legged_gym/scripts/train.py`

```python
def _setup_distributed(args):
    """初始化分布式训练环境"""
    
    # 1. 检测是否启用分布式
    enabled, rank, local_rank, world_size = _get_distributed_env()
    if not enabled:
        return False, 0, 0, 1  # 单卡模式
    
    # 2. 检查依赖
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed 不可用!")
    
    if not torch.cuda.is_available():
        raise RuntimeError("DDP需要CUDA支持!")
    
    # 3. 绑定GPU
    torch.cuda.set_device(local_rank)
    
    # 4. 统一设备名称
    device = f"cuda:{local_rank}"
    args.device = device
    args.sim_device = device
    args.rl_device = device
    
    # 5. 初始化进程组(最关键!)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",          # GPU通信后端
            init_method="env://",    # 从环境变量读取配置
            rank=rank,
            world_size=world_size
        )
    
    # 6. 配置辅助工具
    try:
        from rsl_rl.utils import utils as rsl_dist_utils
        # 设置标量reduce的设备
        rsl_dist_utils.global_mp_device = device
    except Exception:
        pass
    
    # 7. 减少非主进程的输出噪音
    if rank != 0:
        os.environ.setdefault("WANDB_SILENT", "true")
    
    return True, rank, local_rank, world_size
```

**关键点解析**:

1. **`backend="nccl"`**: GPU间通信协议,比 gloo 快 10x
2. **`init_method="env://"`**: 告诉 PyTorch 从环境变量读取 MASTER_ADDR, MASTER_PORT 等
3. **`torch.cuda.set_device(local_rank)`**: 确保每个进程只操作自己的GPU

---

### 层次 3: 模型包装

**文件**: `rsl_rl/utils/utils.py`

```python
class ForwardingDistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    """自定义 DDP 包装器,转发未定义的属性到内部模型"""
    
    def __getattr__(self, name):
        try:
            # 先尝试从 DDP 自身获取属性
            return super().__getattr__(name)
        except AttributeError:
            # 如果没有,转发到包装的模型
            return getattr(self.module, name)

def maybe_wrap_ddp(model: torch.nn.Module, device: str, 
                   find_unused_parameters: bool = True):
    """根据运行模式决定是否包装模型为 DDP"""
    
    # 1. 检查是否已初始化分布式
    if not (torch.distributed.is_available() and 
            torch.distributed.is_initialized()):
        return model  # 单卡模式,直接返回
    
    # 2. 检查是否多进程
    if torch.distributed.get_world_size() <= 1:
        return model
    
    # 3. GPU 模式包装
    if isinstance(device, str) and device.startswith("cuda"):
        try:
            local_rank = int(device.split(":")[1]) if ":" in device else 0
        except Exception:
            local_rank = 0
        
        return ForwardingDistributedDataParallel(
            model,
            device_ids=[local_rank],         # 绑定到哪个GPU
            output_device=local_rank,        # 输出在哪个GPU
            find_unused_parameters=find_unused_parameters,  # 是否检查未使用参数
        )
    
    # 4. CPU 模式包装(罕见)
    return ForwardingDistributedDataParallel(
        model,
        find_unused_parameters=find_unused_parameters,
    )
```

**为什么需要自定义包装器?**

```python
# 标准 DDP 问题:
model = torch.ne.parallel.DistributedDataParallel(my_model)
model.my_custom_method()  # AttributeError!

# 自定义 ForwardingDDP 解决:
model = ForwardingDistributedDataParallel(my_model)
model.my_custom_method()  # 成功! 转发到 my_model.my_custom_method()
```

---

### 层次 4: 数据并行策略

#### 4.1 随机种子偏移

**文件**: `legged_gym/gym_utils/task_registry.py`

```python
# 在 make_env() 中
try:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        # 每个 rank 使用不同的种子
        env_cfg.seed = int(env_cfg.seed) + int(dist.get_rank())
except Exception:
    pass
```

**原理**:
- Rank 0: seed = 42 + 0 = 42
- Rank 1: seed = 42 + 1 = 43
- Rank 2: seed = 42 + 2 = 44
- ...

**作用**: 确保每个 rank 生成不同的环境随机性,提升数据多样性

---

#### 4.2 动作数据分片

**文件**: `pose/utils/motion_lib_pkl.py`

```python
# 在 _fetch_motion_files() 中
try:
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        if world_size > 1:
            rank = dist.get_rank()
            
            # 强制 sample_ratio=1 保证确定性
            if self._sample_ratio != 1.0:
                if rank == 0:
                    print(f"[Warning] DDP模式下 sample_ratio={self._sample_ratio}, "
                          "强制设为1.0以保证可复现")
                self._sample_ratio = 1.0
            
            # 确保每个rank至少有1个动作
            if len(motion_list) < world_size:
                repeats = (world_size + len(motion_list) - 1) // len(motion_list)
                motion_list = (motion_list * repeats)[:world_size]
            
            # Strided sharding: rank, rank+W, rank+2W, ...
            motion_list = motion_list[rank::world_size]
except Exception:
    pass
```

**分片策略示例**:

```python
# 假设有 10 个动作, 4 个 rank
motion_list = [m0, m1, m2, m3, m4, m5, m6, m7, m8, m9]

# Strided sharding:
# Rank 0: [m0, m4, m8]
# Rank 1: [m1, m5, m9]
# Rank 2: [m2, m6]
# Rank 3: [m3, m7]

# 好处: 如果动作按难度排序,每个rank都能获得各种难度的动作
```

---

### 层次 5: 梯度同步工具

**文件**: `rsl_rl/utils/utils.py`

```python
def reduce_sum(x):
    """所有 rank 的 x 求和"""
    return reduce_all(x, torch.distributed.ReduceOp.SUM)

def reduce_mean(x):
    """所有 rank 的 x 求均值"""
    n = get_num_procs()
    sum_x = reduce_sum(x)
    return sum_x / n

def reduce_all(x, op):
    """通用 reduce 操作"""
    if not enable_mp():  # 单进程模式
        return x
    
    is_tensor = torch.is_tensor(x)
    if is_tensor:
        buffer = x.clone()
    else:
        buffer = torch.tensor(x, device=get_device())
    
    # 关键: all_reduce 会同步所有进程
    torch.distributed.all_reduce(buffer, op=op)
    
    if not is_tensor:
        buffer = buffer.item()
    
    return buffer
```

**使用场景**:

```python
# 场景1: 计算全局平均奖励
local_reward = torch.tensor(123.45, device="cuda:0")
global_avg_reward = reduce_mean(local_reward)
# Rank 0: local=120, global=(120+130+125+135)/4=127.5
# Rank 1: local=130, global=127.5
# Rank 2: local=125, global=127.5
# Rank 3: local=135, global=127.5

# 场景2: 只有主进程打印
if is_root_proc():
    print(f"Global average reward: {global_avg_reward}")
```

---

## 从零开始写 DDP 训练

### 最小示例: 单文件 DDP

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset

# ==================== 步骤 1: 环境变量读取 ====================
def get_dist_info():
    """从环境变量获取分布式信息"""
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return world_size, rank, local_rank

# ==================== 步骤 2: 初始化进程组 ====================
def setup_ddp(backend="nccl"):
    """初始化 DDP"""
    world_size, rank, local_rank = get_dist_info()
    
    if world_size == 1:
        return False, rank, local_rank  # 单卡模式
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(local_rank)
    
    # 初始化进程组
    dist.init_process_group(
        backend=backend,
        init_method="env://",  # 从环境变量读取 MASTER_ADDR 等
        world_size=world_size,
        rank=rank
    )
    
    return True, rank, local_rank

# ==================== 步骤 3: 清理资源 ====================
def cleanup_ddp():
    """训练结束后清理"""
    if dist.is_initialized():
        dist.destroy_process_group()

# ==================== 步骤 4: 定义模型和数据 ====================
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class SimpleDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 100)
        self.labels = torch.randint(0, 10, (size,))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# ==================== 步骤 5: 主训练函数 ====================
def train():
    # 5.1 初始化 DDP
    is_ddp, rank, local_rank = setup_ddp()
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    
    # 5.2 创建模型
    model = SimpleModel().to(device)
    if is_ddp:
        model = DDP(model, device_ids=[local_rank])
    
    # 5.3 创建数据加载器
    dataset = SimpleDataset(size=10000)
    sampler = DistributedSampler(dataset) if is_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler,
        shuffle=(sampler is None)  # 如果用了sampler就不能shuffle
    )
    
    # 5.4 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 5.5 训练循环
    num_epochs = 10
    for epoch in range(num_epochs):
        # ⚠️ 重要: DDP 需要在每个 epoch 开始时设置 epoch
        if is_ddp:
            sampler.set_epoch(epoch)
        
        model.train()
        epoch_loss = 0.0
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()  # DDP 自动同步梯度!
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 5.6 计算全局平均损失
        avg_loss = epoch_loss / len(dataloader)
        if is_ddp:
            loss_tensor = torch.tensor(avg_loss, device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg_loss = loss_tensor.item() / dist.get_world_size()
        
        # 5.7 只有主进程打印
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # 5.8 清理
    cleanup_ddp()

# ==================== 运行 ====================
if __name__ == "__main__":
    train()
```

### 启动方式

```bash
# 单卡训练(调试)
python train_ddp.py

# 单机 4 卡
torchrun --nproc_per_node=4 train_ddp.py

# 多机训练(2台机器,每台4卡)
# 机器 0 (主节点):
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=0 \
         --master_addr=192.168.1.100 \
         --master_port=29500 \
         train_ddp.py

# 机器 1:
torchrun --nnodes=2 --nproc_per_node=4 \
         --node_rank=1 \
         --master_addr=192.168.1.100 \
         --master_port=29500 \
         train_ddp.py
```

---

## 常见问题与调试

### 问题 1: 进程卡住不动

**现象**:
```
启动后所有进程都在等待,没有任何输出
```

**原因**: 某个进程中有集体通信操作(all_reduce, broadcast等),但不是所有进程都执行了

**错误示例**:
```python
if rank == 0:
    loss = compute_loss()
    dist.all_reduce(loss)  # ❌ 只有 rank 0 执行,其他 rank 在等待
```

**正确做法**:
```python
loss = compute_loss()
dist.all_reduce(loss)  # ✅ 所有 rank 都执行
if rank == 0:
    print(f"Global loss: {loss}")
```

---

### 问题 2: 梯度不同步

**现象**:
```
各个 rank 的模型参数逐渐发散
```

**原因**: 忘记使用 `DistributedSampler` 或忘记调用 `set_epoch()`

**错误示例**:
```python
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # ❌
```

**正确做法**:
```python
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # ✅ 关键!
    ...
```

---

### 问题 3: OOM (Out of Memory)

**现象**:
```
CUDA out of memory error
```

**原因**: 每个 GPU 都加载了完整数据集或完整模型

**解决方案**:

1. **减小单卡 batch size**:
```python
# 调整前: 单卡 batch=128
# 调整后: 单卡 batch=32, 4卡总 batch=128
```

2. **模型分片(ZeRO)**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model)  # 参数分片到各GPU
```

---

### 问题 4: 速度没有提升

**现象**:
```
4卡训练速度不到单卡的2倍
```

**可能原因**:

1. **数据加载瓶颈**:
```python
# 增加 num_workers
dataloader = DataLoader(dataset, batch_size=32, num_workers=8)
```

2. **小模型通信开销大**:
```python
# 使用梯度累积,减少通信频率
for i, (data, target) in enumerate(dataloader):
    loss = model(data, target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()  # 每N步才同步
        optimizer.zero_grad()
```

3. **`find_unused_parameters=True` 拖慢速度**:
```python
# 如果确定所有参数都被使用,设为 False
model = DDP(model, find_unused_parameters=False)
```

---

###问题 5: WandB 日志重复

**现象**:
```
每个 metric 都被记录了 4 次(4卡训练)
```

**原因**: 所有 rank 都在记录日志

**解决方案**:
```python
is_dist, rank, _, _ = _get_distributed_env()
is_root = (not is_dist) or rank == 0

if is_root:
    wandb.init(project="my_project", name="run_1")

# 训练循环中
if is_root:
    wandb.log({"loss": avg_loss})
```

---

## 性能优化技巧

### 技巧 1: 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for data, target in dataloader:
    optimizer.zero_grad()
    
    # 前向传播用 float16
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    
    # 反向传播用缩放防止下溢
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**性能提升**: 约 2-3x,显存减半

---

### 技巧 2: 梯度累积

```python
accumulation_steps = 4  # 实际 batch_size = 32 * 4 = 128

for i, (data, target) in enumerate(dataloader):
    # 缩放损失
    loss = criterion(model(data), target) / accumulation_steps
    loss.backward()
    
    # 每 N 步才更新
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

**好处**: 模拟更大 batch size,减少通信频率

---

### 技巧 3: NCCL 环境变量调优

```bash
# 单机多卡优化
export NCCL_P2P_DISABLE=0            # 启用GPU直连
export NCCL_IB_DISABLE=1             # 禁用InfiniBand(单机不需要)
export NCCL_SOCKET_IFNAME=lo         # 使用本地回环

# 多机训练优化
export NCCL_IB_DISABLE=0             # 启用InfiniBand
export NCCL_SOCKET_IFNAME=eth0       # 指定网络接口
export NCCL_DEBUG=INFO               # 调试通信问题
```

---

### 技巧 4: GPU 亲和性绑定

```python
# 避免跨 NUMA node 访问内存
import subprocess

def set_affinity(local_rank):
    # 查询 GPU 的 NUMA node
    cmd = f"nvidia-smi topo -m | grep GPU{local_rank}"
    # 设置 CPU 亲和性...
    pass

# 在初始化后调用
set_affinity(local_rank)
```

---

## TWIST2 实战清单

### 启动命令模板

```bash
# 1. 调试模式(单卡,小环境)
python legged_gym/scripts/train.py --task=g1_mimic_distill --debug

# 2. 单机 2 卡
torchrun --nproc_per_node=2 \
    legged_gym/scripts/train.py \
    --task=g1_mimic_distill \
    --num_envs=2048 \
    --exptid=ddp_test

# 3. 单机 4 卡 + 自定义端口
torchrun --nproc_per_node=4 \
    --master_port=29501 \
    legged_gym/scripts/train.py \
    --task=g1_mimic_distill \
    --num_envs=4096

# 4. 多机 2x4 卡
# 主节点 (192.168.1.100):
torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    legged_gym/scripts/train.py \
    --task=g1_mimic_distill \
    --num_envs=8192

# 从节点 (192.168.1.101):
torchrun --nnodes=2 --nproc_per_node=4 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    legged_gym/scripts/train.py \
    --task=g1_mimic_distill \
    --num_envs=8192
```

### 验证 DDP 是否生效

```bash
# 查看 GPU 利用率
watch -n 1 nvidia-smi

# 期望输出(4卡训练):
# GPU 0: 95% memory, 90% utilization
# GPU 1: 95% memory, 90% utilization
# GPU 2: 95% memory, 90% utilization
# GPU 3: 95% memory, 90% utilization

# 如果只有 GPU 0 在工作 → DDP 未启用!
```

### 性能基准测试

```python
# 添加到训练脚本测试吞吐量
import time

start_time = time.time()
num_steps = 1000

for i in range(num_steps):
    # 训练一步
    ...

elapsed = time.time() - start_time
steps_per_sec = num_steps / elapsed

if rank == 0:
    print(f"吞吐量: {steps_per_sec:.2f} steps/sec")
    print(f"加速比: {steps_per_sec / single_gpu_baseline:.2f}x")
```

---

## 总结与下一步

### DDP 核心要点

1. ✅ **torchrun 自动设置环境变量** → 读取 WORLD_SIZE, RANK
2. ✅ **init_process_group 初始化通信** → 使用 NCCL backend
3. ✅ **每个进程绑定一个 GPU** → `torch.cuda.set_device(local_rank)`
4. ✅ **模型包装为 DDP** → 自动同步梯度
5. ✅ **数据分片** → DistributedSampler + set_epoch()
6. ✅ **只有主进程记录日志** → `if rank == 0`

### 进阶主题

- **DeepSpeed**: 超大模型训练(ZeRO优化器)
- **FSDP**: PyTorch 原生模型并行
- **Pipeline Parallelism**: 层间流水线
- **Tensor Parallelism**: 张量切分

### 参考资源

- [PyTorch DDP 官方教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL 性能调优](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [torchrun 文档](https://pytorch.org/docs/stable/elastic/run.html)

---

**最后建议**: 从最小示例开始,逐步添加功能。每次只改一个地方,这样容易定位问题!

🚀 Happy Distributed Training!
