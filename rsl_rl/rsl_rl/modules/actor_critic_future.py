# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np                              # 导入NumPy库，用于数值计算
import torch                                    # 导入PyTorch深度学习框架
import torch.nn as nn                           # 导入PyTorch神经网络模块
import torch.nn.functional as F                 # 导入PyTorch函数式API
from torch.distributions import Normal          # 从torch.distributions导入正态分布类
from torch.nn.modules import rnn                # 导入RNN模块（当前未使用）
from torch.nn.modules.activation import ReLU    # 导入ReLU激活函数（当前未直接使用）


def get_activation(act_name):
    """
    根据名称获取对应的激活函数。
    
    Args:
        act_name: 激活函数名称字符串
        
    Returns:
        nn.Module: 对应的PyTorch激活函数模块
    """
    if act_name == "elu":                           # 如果名称为"elu"
        return nn.ELU()                             # 返回ELU激活函数
    elif act_name == "selu":                        # 如果名称为"selu"
        return nn.SELU()                            # 返回SELU激活函数
    elif act_name == "relu":                        # 如果名称为"relu"
        return nn.ReLU()                            # 返回ReLU激活函数
    elif act_name == "crelu":                       # 如果名称为"crelu"
        return nn.ReLU()                            # 返回ReLU（crelu暂未单独实现）
    elif act_name == "lrelu":                       # 如果名称为"lrelu"
        return nn.LeakyReLU()                       # 返回LeakyReLU激活函数
    elif act_name == "tanh":                        # 如果名称为"tanh"
        return nn.Tanh()                            # 返回Tanh激活函数
    elif act_name == "sigmoid":                     # 如果名称为"sigmoid"
        return nn.Sigmoid()                         # 返回Sigmoid激活函数
    elif act_name == "silu":                        # 如果名称为"silu"
        return nn.SiLU()                            # 返回SiLU（Swish）激活函数
    else:                                           # 其他不支持的名称
        print("invalid activation function!")       # 打印错误信息
        return None                                 # 返回None


class MotionEncoder(nn.Module):
    """
    动作编码器 - 使用CNN对时序动作观测进行编码。
    
    将时序的动作观测数据通过线性投影和1D卷积层编码成固定维度的特征向量。
    支持不同的时间步长（1, 10, 20, 50）。
    
    Attributes:
        activation_fn: 激活函数
        tsteps: 时间步长
        encoder: 线性投影层，将输入特征投影到更高维度
        conv_layers: 1D卷积层序列，用于提取时序特征
        linear_output: 输出线性层，将卷积结果映射到目标维度
    """
    
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        """
        初始化动作编码器。
        
        Args:
            activation_fn: 激活函数
            input_size: 单步观测的输入维度
            tsteps: 时间步长（支持1, 10, 20, 50）
            output_size: 输出特征维度
            tanh_encoder_output: 是否在输出上使用tanh（当前未使用）
        """
        super().__init__()                          # 调用父类nn.Module的初始化方法
        self.activation_fn = activation_fn          # 存储激活函数
        self.tsteps = tsteps                        # 存储时间步长

        channel_size = 20                           # 设置卷积通道数基数为20

        # 线性投影层：将输入特征投影到3倍通道大小（60维）
        self.encoder = nn.Sequential(               # 创建顺序容器
                nn.Linear(input_size, 3 * channel_size),  # 线性层：input_size -> 60
                self.activation_fn,                 # 添加激活函数
                )                                   # 结束encoder的定义

        # 根据时间步长选择合适的卷积结构
        if tsteps == 50:                            # 如果时间步长为50
            # 50步：3层卷积，逐步降采样，从50步压缩到特征向量
            self.conv_layers = nn.Sequential(       # 创建卷积层序列
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4),  # 第1层卷积：60->40通道，50->11序列长度
                    self.activation_fn,             # 添加激活函数
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1),      # 第2层卷积：40->20通道，11->7序列长度
                    self.activation_fn,             # 添加激活函数
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1),          # 第3层卷积：20->20通道，7->3序列长度
                    self.activation_fn,             # 添加激活函数
                    nn.Flatten())                   # 展平层：20*3=60维
        elif tsteps == 10:                          # 如果时间步长为10
            # 10步：2层卷积
            self.conv_layers = nn.Sequential(       # 创建卷积层序列
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2),      # 第1层卷积：60->40通道，10->4序列长度
                self.activation_fn,                 # 添加激活函数
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1),          # 第2层卷积：40->20通道，4->3序列长度
                self.activation_fn,                 # 添加激活函数
                nn.Flatten())                       # 展平层：20*3=60维
        elif tsteps == 20:                          # 如果时间步长为20
            # 20步：2层卷积，不同的核大小和步长
            self.conv_layers = nn.Sequential(       # 创建卷积层序列
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2),      # 第1层卷积：60->40通道，20->8序列长度
                self.activation_fn,                 # 添加激活函数
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2),          # 第2层卷积：40->20通道，8->3序列长度
                self.activation_fn,                 # 添加激活函数
                nn.Flatten())                       # 展平层：20*3=60维
        elif tsteps == 1:                           # 如果只有1个时间步
            # 单步：直接展平，无需卷积
            self.conv_layers = nn.Flatten()         # 仅使用展平层
        else:                                       # 不支持的时间步长
            raise(ValueError("tsteps must be 1, 10, 20 or 50"))  # 抛出数值错误异常
        
        # 输出线性层：将卷积输出的60维映射到目标输出维度
        self.linear_output = nn.Linear(channel_size * 3, output_size)  # 线性层：60 -> output_size

    def forward(self, obs):
        """
        前向传播。
        
        Args:
            obs: 输入观测，形状为 (batch_size, tsteps, input_size) 或 (batch_size, tsteps * input_size)
            
        Returns:
            output: 编码后的特征，形状为 (batch_size, output_size)
        """
        nd = obs.shape[0]                           # 获取batch大小（样本数量）
        T = self.tsteps                             # 获取时间步长
        
        # 将观测 reshape 为 (batch * tsteps, input_size) 并通过编码器
        # 展平时序维度，使每个时间步独立通过线性层
        projection = self.encoder(obs.reshape([nd * T, -1]))  # 投影：n_proprio -> 3*channel_size (60)
        
        # reshape回 (batch, tsteps, 3*channel_size)，转置为 (batch, channels, tsteps) 以适应Conv1d
        # Conv1d期望输入形状为 (batch, channels, seq_len)
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))  # 卷积处理
        output = self.linear_output(output)         # 通过输出线性层映射到目标维度
        return output                               # 返回编码后的特征


class HistoryEncoder(nn.Module):
    """
    历史编码器 - 与MotionEncoder结构相同，用于编码历史观测。
    
    用于处理机器人的历史观测信息，提取时序特征。
    结构与MotionEncoder完全一致，但语义上用于处理不同的输入。
    
    Attributes:
        activation_fn: 激活函数
        tsteps: 历史时间步长
        encoder: 线性投影层
        conv_layers: 1D卷积层序列
        linear_output: 输出线性层
    """
    
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        """
        初始化历史编码器。
        
        Args:
            activation_fn: 激活函数
            input_size: 单步观测的输入维度
            tsteps: 时间步长（支持1, 10, 20, 50）
            output_size: 输出特征维度
            tanh_encoder_output: 是否在输出上使用tanh
        """
        super().__init__()                          # 调用父类初始化
        self.activation_fn = activation_fn          # 存储激活函数
        self.tsteps = tsteps                        # 存储时间步长

        channel_size = 20                           # 设置卷积通道数基数为20

        # 线性投影层
        self.encoder = nn.Sequential(               # 创建编码器序列
                nn.Linear(input_size, 3 * channel_size),  # 线性层：input_size -> 60
                self.activation_fn,                 # 添加激活函数
                )                                   # 结束encoder定义

        # 根据时间步长选择卷积结构，与MotionEncoder完全一致
        if tsteps == 50:                            # 50步配置
            self.conv_layers = nn.Sequential(       # 3层卷积结构
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4),
                    self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1),
                    self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1),
                    self.activation_fn, nn.Flatten())
        elif tsteps == 10:                          # 10步配置
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2),
                self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1),
                self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:                          # 20步配置
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2),
                self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2),
                self.activation_fn,
                nn.Flatten())
        elif tsteps == 1:                           # 单步配置
            self.conv_layers = nn.Flatten()         # 直接展平
        else:                                       # 不支持的时间步长
            assert False, f"tsteps must be 1, 10, 20 or 50, but got {tsteps}"  # 断言失败
        self.linear_output = nn.Linear(channel_size * 3, output_size)  # 输出线性层

    def forward(self, obs):
        """
        前向传播。
        
        Args:
            obs: 历史观测，形状为 (batch_size, tsteps, input_size)
            
        Returns:
            output: 编码后的历史特征，形状为 (batch_size, output_size)
        """
        nd = obs.shape[0]                           # 获取batch大小
        T = self.tsteps                             # 获取时间步长
        # 与MotionEncoder相同的前向传播逻辑
        projection = self.encoder(obs.reshape([nd * T, -1]))  # 投影：n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))  # 卷积处理
        output = self.linear_output(output)         # 输出映射
        return output                               # 返回编码结果
 

class FutureMotionEncoder(nn.Module):
    """
    未来动作编码器 - 简化的未来观测编码器，不使用注意力机制。
    
    将未来时刻的动作观测展平后通过MLP编码成固定维度的特征向量。
    输入包含未来观测和一个mask指示器（表示该时刻是否有效）。
    
    Attributes:
        activation_fn: 激活函数
        tsteps: 未来时间步数
        input_size: 单步观测维度
        output_size: 输出特征维度
        encoder: MLP编码器
    """
    
    def __init__(self, activation_fn, input_size, tsteps, output_size, 
                 attention_heads=4, dropout=0.1, temporal_embedding_dim=64):
        """
        初始化未来动作编码器。
        
        Args:
            activation_fn: 激活函数
            input_size: 单步观测的输入维度
            tsteps: 未来时间步数
            output_size: 输出特征维度
            attention_heads: 注意力头数（当前未使用，保留用于兼容）
            dropout: Dropout概率
            temporal_embedding_dim: 时间嵌入维度（当前未使用）
        """
        super().__init__()                          # 调用父类初始化
        self.activation_fn = activation_fn          # 存储激活函数
        self.tsteps = tsteps                        # 存储时间步长
        self.input_size = input_size                # 存储输入维度
        self.output_size = output_size              # 存储输出维度
        
        # 简单方法：展平所有未来观测并使用MLP编码
        total_input_size = input_size * tsteps      # 计算展平后的总维度
        
        # MLP编码器：3层全连接网络
        self.encoder = nn.Sequential(
            nn.Linear(total_input_size, 256),       # 第1层：展平输入 -> 256维
            activation_fn,                          # 激活函数
            nn.Dropout(dropout),                    # Dropout正则化
            nn.Linear(256, 128),                    # 第2层：256 -> 128维
            activation_fn,                          # 激活函数
            nn.Dropout(dropout),                    # Dropout正则化
            nn.Linear(128, output_size)             # 第3层：128 -> output_size维
        )
        
        # 使用稳定的权重初始化（Xavier均匀初始化）
        for layer in self.encoder:                  # 遍历编码器的所有层
            if isinstance(layer, nn.Linear):        # 如果是线性层
                nn.init.xavier_uniform_(layer.weight, gain=0.5)  # Xavier初始化，增益0.5
                nn.init.zeros_(layer.bias)          # 偏置初始化为0
        
    def forward(self, obs):
        """
        前向传播。
        
        Args:
            obs: 输入观测，形状为 (batch_size, tsteps, input_size + 1)
                 最后一维是mask指示器（1表示有效，0表示无效）
                 
        Returns:
            output: 编码后的未来特征，形状为 (batch_size, output_size)
        """
        batch_size = obs.shape[0]                   # 获取batch大小
        
        # 分离mask指示器和观测值
        future_obs = obs[:, :, :-1]                 # 提取未来观测：(batch_size, tsteps, input_size)
        mask_indicator = obs[:, :, -1]              # 提取mask指示器：(batch_size, tsteps)
        
        # 简单方法：展平并编码
        # 暂时忽略mask，使用所有未来观测
        flattened = future_obs.reshape(batch_size, -1)  # 展平：(batch_size, tsteps * input_size)
        
        # 通过MLP编码器
        output = self.encoder(flattened)            # (batch_size, output_size)
        
        return output                               # 返回编码后的未来特征


class ActorFuture(nn.Module):
    """
    Actor网络 - 支持未来动作观测的标准MLP策略网络。
    
    输入观测结构：[当前动作观测 | 本体感知观测 | 历史观测 | 未来观测]
    - 当前动作观测：通过MotionEncoder编码
    - 本体感知观测：直接使用
    - 历史观测：通过HistoryEncoder编码
    - 未来观测：通过FutureMotionEncoder编码
    
    Attributes:
        motion_encoder: 动作编码器
        history_encoder: 历史编码器
        future_encoder: 未来动作编码器
        actor_backbone: 主MLP网络，输出动作
    """
    
    def __init__(self, num_observations,
                 num_motion_observations,
                 num_priop_observations,
                 num_motion_steps,
                 num_future_observations,
                 num_future_steps,
                 motion_latent_dim,
                 future_latent_dim,
                 num_actions,
                 actor_hidden_dims, 
                 activation, 
                 history_latent_dim,
                 num_history_steps,
                 layer_norm=False,
                 future_encoder_dims=[256, 256, 128],
                 future_attention_heads=4,
                 future_dropout=0.1,
                 temporal_embedding_dim=64,
                 use_history_encoder=True,
                 use_motion_encoder=True,
                 tanh_encoder_output=False, **kwargs):
        """
        初始化Actor网络。
        
        观测结构：motion_obs, priop_obs, history_obs, future_obs
        
        Args:
            num_observations: 总观测维度
            num_motion_observations: 动作观测总维度（包含时序）
            num_priop_observations: 本体感知观测维度
            num_motion_steps: 动作观测时间步数
            num_future_observations: 未来观测总维度
            num_future_steps: 未来观测时间步数
            motion_latent_dim: 动作编码后的维度
            future_latent_dim: 未来观测编码后的维度
            num_actions: 动作输出维度
            actor_hidden_dims: Actor隐藏层维度列表
            activation: 激活函数
            history_latent_dim: 历史编码后的维度
            num_history_steps: 历史时间步数
            layer_norm: 是否使用LayerNorm
            future_encoder_dims: 未来编码器维度
            future_attention_heads: 注意力头数
            future_dropout: Dropout概率
            temporal_embedding_dim: 时间嵌入维度
            use_history_encoder: 是否使用历史编码器
            use_motion_encoder: 是否使用动作编码器
            tanh_encoder_output: 输出是否使用tanh
        """
        super().__init__()                          # 调用父类初始化
        self.num_observations = num_observations    # 存储总观测维度
        self.num_actions = num_actions              # 存储动作维度
        self.num_motion_observations = num_motion_observations  # 存储动作观测维度
        self.num_priop_observations = num_priop_observations    # 存储本体感知维度
        self.num_motion_steps = num_motion_steps    # 存储动作时间步数
        self.num_history_steps = num_history_steps  # 存储历史时间步数
        self.num_future_observations = num_future_observations  # 存储未来观测维度
        self.num_future_steps = num_future_steps    # 存储未来时间步数
        self.use_history_encoder = use_history_encoder  # 是否使用历史编码器
        self.use_motion_encoder = use_motion_encoder    # 是否使用动作编码器
        
        # 计算单步观测维度
        self.num_single_motion_observations = int(num_motion_observations / num_motion_steps)  # 单步动作观测维度
        self.num_single_priop_observations = num_priop_observations  # 单步本体感知维度（直接使用）
        
        # 计算历史观测总维度
        # 历史观测是单步观测重复num_history_steps次
        num_single_history_observations = num_motion_observations + num_priop_observations  # 单步历史观测维度
        history_size = num_single_history_observations * num_history_steps  # 历史观测总维度
        
        # 计算未来单步观测维度
        self.num_single_future_observations = int(num_future_observations / num_future_steps) if num_future_observations > 0 else 0  # 单步未来观测维度
        self.future_latent_dim = future_latent_dim  # 存储未来编码维度
        
        # 动作编码器（与tracking相同）
        if self.use_motion_encoder:                 # 如果使用动作编码器
            self.motion_encoder = MotionEncoder(activation, self.num_single_motion_observations, self.num_motion_steps, motion_latent_dim)  # 创建MotionEncoder
        else:                                       # 如果不使用编码器
            self.motion_encoder = nn.Identity()     # 使用恒等映射
            motion_latent_dim = self.num_single_motion_observations  # 编码后维度等于输入维度
        
        # 历史编码器（与tracking相同）
        if self.use_history_encoder:                # 如果使用历史编码器
            self.history_encoder = HistoryEncoder(activation, num_single_history_observations, self.num_history_steps, history_latent_dim)  # 创建HistoryEncoder
        else:                                       # 如果不使用编码器
            self.history_encoder = nn.Identity()    # 使用恒等映射
            history_latent_dim = history_size       # 编码后维度等于输入维度
            
        # 未来动作编码器（新增）
        if self.num_single_future_observations > 0:  # 如果有未来观测
            self.future_encoder = FutureMotionEncoder(
                activation,                         # 激活函数
                self.num_single_future_observations - 1,  # -1 因为mask指示器是分开的
                self.num_future_steps,              # 未来时间步数
                future_latent_dim,                  # 输出维度
                attention_heads=future_attention_heads,  # 注意力头数（兼容参数）
                dropout=future_dropout,             # Dropout概率
                temporal_embedding_dim=temporal_embedding_dim  # 时间嵌入维度（兼容参数）
            )
        else:                                       # 如果没有未来观测
            self.future_encoder = None              # 未来编码器为None
        
        # 主Actor网络
        # 输入维度 = 动作编码 + 单步动作观测 + 本体感知 + 历史编码 + 未来编码
        input_dim = (motion_latent_dim + self.num_single_motion_observations + 
                    self.num_single_priop_observations + history_latent_dim + future_latent_dim)
        
        actor_layers = []                           # 初始化Actor层列表
        first_layer = nn.Linear(input_dim, actor_hidden_dims[0])  # 第一层线性层
        # 第一层使用较小的权重初始化，增加稳定性
        nn.init.xavier_uniform_(first_layer.weight, gain=0.5)  # Xavier初始化，增益0.5
        nn.init.zeros_(first_layer.bias)            # 偏置初始化为0
        actor_layers.append(first_layer)            # 添加第一层到列表
        actor_layers.append(activation)             # 添加激活函数
        
        # 构建隐藏层
        for l in range(len(actor_hidden_dims)):     # 遍历隐藏层维度列表
            if l == len(actor_hidden_dims) - 1:     # 如果是最后一层
                # 最后一层：输出动作
                final_layer = nn.Linear(actor_hidden_dims[l], num_actions)  # 输出层
                # 输出层使用非常小的权重
                nn.init.xavier_uniform_(final_layer.weight, gain=0.1)  # Xavier初始化，增益0.1（更小的初始输出）
                nn.init.zeros_(final_layer.bias)    # 偏置初始化为0
                actor_layers.append(final_layer)    # 添加输出层
            else:                                   # 如果不是最后一层
                layer = nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])  # 创建线性层
                nn.init.xavier_uniform_(layer.weight)  # 标准Xavier初始化
                nn.init.zeros_(layer.bias)          # 偏置初始化为0
                actor_layers.append(layer)          # 添加线性层
                # 倒数第二层可以使用LayerNorm
                if layer_norm and l == len(actor_hidden_dims) - 2:  # 如果是倒数第二层且启用layer_norm
                    actor_layers.append(nn.LayerNorm(actor_hidden_dims[l + 1]))  # 添加LayerNorm
                actor_layers.append(activation)     # 添加激活函数
                
        if tanh_encoder_output:                     # 如果启用tanh输出
            actor_layers.append(nn.Tanh())          # 添加Tanh激活（限制输出范围）
            
        self.actor_backbone = nn.Sequential(*actor_layers)  # 使用Sequential包装所有层

    def forward(self, obs, hist_encoding: bool = False):
        """
        前向传播。
        
        解析观测结构：当前观测 + 历史观测 + 未来观测
        当前观测 = 动作观测 + 本体感知观测
        
        Args:
            obs: 完整观测向量
            hist_encoding: 是否只返回历史编码（当前未使用）
            
        Returns:
            backbone_output: 动作输出
        """
        # 解析观测结构：当前 + 历史 + 未来
        # 当前 = 动作 + 本体感知
        current_size = self.num_motion_observations + self.num_priop_observations  # 当前观测总维度
        
        # 提取当前观测
        motion_obs = obs[:, :self.num_motion_observations]  # 提取动作观测（前num_motion_observations维）
        single_motion_obs = obs[:, :self.num_single_motion_observations]  # 提取单步动作观测（前num_single_motion_observations维）
        priop_obs = obs[:, self.num_motion_observations:current_size]  # 提取本体感知观测
        
        # 提取历史观测
        history_start = current_size                # 历史观测起始位置
        history_size = self.num_history_steps * current_size  # 历史观测总大小
        history_end = history_start + history_size  # 历史观测结束位置
        history_obs = obs[:, history_start:history_end]  # 提取历史观测
        
        # 提取未来观测（排除其他组件如masked_priv_info）
        future_start = history_end                  # 未来观测起始位置
        future_end = future_start + self.num_future_observations  # 未来观测结束位置
        future_obs = obs[:, future_start:future_end]  # 提取未来观测
        
        # 编码所有组件
        motion_latent = self.motion_encoder(motion_obs)  # 通过动作编码器
        history_latent = self.history_encoder(history_obs)  # 通过历史编码器
        
        # 编码未来动作 - 简化方法
        if self.future_encoder is not None and self.num_future_observations > 0 and future_obs.shape[1] > 0:  # 如果有未来编码器和未来观测
            # 将未来观测reshape为 (batch_size, num_future_steps, obs_per_step)
            future_obs_reshaped = future_obs.reshape(-1, self.num_future_steps, self.num_single_future_observations)
            future_latent = self.future_encoder(future_obs_reshaped)  # 通过未来编码器
        else:                                       # 如果没有未来观测
            # 如果没有未来观测，创建虚拟的未来特征（全零）
            future_latent = torch.zeros(obs.shape[0], self.future_latent_dim, device=obs.device)
        
        # 组合所有特征
        backbone_input = torch.cat([
            single_motion_obs,                      # 单步动作观测
            priop_obs,                              # 本体感知观测
            motion_latent,                          # 动作编码特征
            history_latent,                         # 历史编码特征
            future_latent                           # 未来编码特征
        ], dim=1)                                   # 在特征维度上拼接
        
        backbone_output = self.actor_backbone(backbone_input)  # 通过主网络
        return backbone_output                      # 返回动作输出


class ActorCriticFuture(nn.Module):
    """
    Actor-Critic网络 - 支持未来动作观测的策略-价值网络。
    
    支持两种Actor结构：
    1. 标准MLP（ActorFuture）
    2. 混合专家模型MoE（ActorFutureMoE）
    
    通过policy配置中的use_moe参数切换。
    
    Attributes:
        actor: Actor网络（生成动作）
        critic: Critic网络（评估价值）
        std: 动作噪声标准差
        distribution: 动作分布（高斯分布）
    """
    
    is_recurrent = False                            # 非循环网络标记
    
    def __init__(self,  
                num_observations,
                num_critic_observations,
                num_motion_observations,
                num_motion_steps,
                num_priop_observations,
                num_history_steps,
                num_actions,
                actor_hidden_dims=[256, 256, 256],
                critic_hidden_dims=[256, 256, 256],
                motion_latent_dim=128,
                history_latent_dim=128,
                future_latent_dim=128,
                activation='silu',
                init_noise_std=1.0,
                fix_action_std=False,
                action_std=None,
                layer_norm=False,
                # 未来动作特定参数
                future_encoder_dims=[256, 256, 128],
                future_attention_heads=4,
                future_dropout=0.1,
                temporal_embedding_dim=64,
                # MoE特定参数
                use_moe=False,
                num_experts=4,
                expert_hidden_dims=[512, 384, 192],
                gating_hidden_dim=128,
                moe_topk=2,
                moe_temperature=1.0,
                # Transformer specific parameters
                use_transformer=False,
                d_model=256,
                nhead=8,
                num_transformer_layers=2,
                transformer_dropout=0.1,
                **kwargs):
        """
        初始化Actor-Critic网络。
        
        Args:
            num_observations: Actor输入的总观测维度
            num_critic_observations: Critic输入的观测维度（通常包含特权信息）
            num_motion_observations: 动作观测总维度
            num_motion_steps: 动作观测时间步数
            num_priop_observations: 本体感知观测维度
            num_history_steps: 历史时间步数
            num_actions: 动作维度
            actor_hidden_dims: Actor隐藏层维度
            critic_hidden_dims: Critic隐藏层维度
            motion_latent_dim: 动作编码维度
            history_latent_dim: 历史编码维度
            future_latent_dim: 未来编码维度
            activation: 激活函数名称
            init_noise_std: 初始动作噪声标准差
            fix_action_std: 是否固定动作标准差
            action_std: 固定的动作标准差值
            layer_norm: 是否使用LayerNorm
            future_encoder_dims: 未来编码器维度
            future_attention_heads: 注意力头数
            future_dropout: Dropout概率
            temporal_embedding_dim: 时间嵌入维度
            use_moe: 是否使用MoE
            num_experts: 专家数量
            expert_hidden_dims: 专家网络隐藏层维度
            gating_hidden_dim: 门控网络隐藏层维度
            moe_topk: 激活的专家数
            moe_temperature: 门控softmax温度
            use_transformer: 是否使用Transformer
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_transformer_layers: Transformer层数
            transformer_dropout: Transformer dropout概率
        """
        if kwargs:                                  # 如果有额外的关键字参数
            # 过滤已知kwargs避免打印警告
            known_kwargs = ['tanh_encoder_output', 'num_future_observations', 'num_future_steps']
            unknown_kwargs = {k: v for k, v in kwargs.items() if k not in known_kwargs}  # 筛选未知参数
            if unknown_kwargs:                        # 如果有未知参数
                print("ActorCriticFuture.__init__ got unexpected arguments, which will be ignored: " + str(list(unknown_kwargs.keys())))  # 打印警告
        super().__init__()                          # 调用父类初始化

        self.fix_action_std = fix_action_std        # 存储是否固定标准差
        self.use_moe = use_moe                      # 存储是否使用MoE
        self.use_transformer = use_transformer      # 存储是否使用Transformer
        self.kwargs = kwargs                        # 存储额外参数
        activation_fn = get_activation(activation)  # 获取激活函数实例
        
        # 根据环境配置计算未来动作维度
        single_obs_size = num_motion_observations + num_priop_observations  # 单步观测总大小
        expected_history_size = num_history_steps * single_obs_size  # 期望的历史观测大小
        expected_current_size = single_obs_size     # 期望的当前观测大小
        
        # 使用显式的num_future_observations（如果提供），否则计算
        if 'num_future_observations' in kwargs:     # 如果提供了明确的未来观测维度
            num_future_observations = kwargs['num_future_observations']  # 使用提供的值
            print(f"ActorCriticFuture: Using explicit num_future_observations = {num_future_observations}")  # 打印信息
        else:                                       # 如果没有提供
            num_future_observations = max(0, num_observations - expected_current_size - expected_history_size)  # 计算未来观测维度
            print(f"ActorCriticFuture: Calculated num_future_observations = {num_future_observations}")  # 打印计算结果
        
        # 从kwargs获取未来步数，否则使用默认值
        num_future_steps = kwargs.get('num_future_steps', 10)  # 获取未来步数，默认10
        
        # 打印网络配置信息
        print(f"ActorCriticFuture: obs={num_observations}, motion={num_motion_observations}, priop={num_priop_observations}")
        print(f"ActorCriticFuture: current={expected_current_size}, history={expected_history_size}, future={num_future_observations}")
        print(f"ActorCriticFuture: future_steps={num_future_steps}")
        
        # 根据标志选择Actor类（优先级: Transformer > MoE > MLP）
        if use_transformer:                         # 如果使用Transformer
            print(f"ActorCriticFuture: Using Transformer actor (d_model={d_model}, nhead={nhead}, layers={num_transformer_layers})")
            actor_class = ActorFutureTransformer    # 使用Transformer Actor类
            actor_kwargs = {                        # Transformer特定参数
                'd_model': d_model,
                'nhead': nhead,
                'num_transformer_layers': num_transformer_layers,
                'transformer_dropout': transformer_dropout,
            }
        elif use_moe:                               # 如果使用MoE
            print(f"ActorCriticFuture: Using MoE actor with {num_experts} experts, top-{moe_topk}")
            actor_class = ActorFutureMoE            # 使用MoE Actor类
            actor_kwargs = {                        # MoE特定参数
                'num_experts': num_experts,
                'expert_hidden_dims': expert_hidden_dims,
                'gating_hidden_dim': gating_hidden_dim,
                'moe_topk': moe_topk,
                'moe_temperature': moe_temperature,
            }
        else:                                       # 默认使用标准MLP
            print(f"ActorCriticFuture: Using standard MLP actor")
            actor_class = ActorFuture               # 使用标准Actor类
            actor_kwargs = {}                       # 无额外参数
        
        # 创建Actor网络
        self.actor = actor_class(
            num_observations=num_observations,      # 总观测维度
            num_actions=num_actions,                # 动作维度
            num_motion_observations=num_motion_observations,  # 动作观测维度
            num_priop_observations=num_priop_observations,    # 本体感知维度
            num_motion_steps=num_motion_steps,      # 动作时间步数
            num_future_observations=num_future_observations,  # 未来观测维度
            num_future_steps=num_future_steps,      # 未来时间步数
            num_history_steps=num_history_steps,    # 历史时间步数
            motion_latent_dim=motion_latent_dim,    # 动作编码维度
            future_latent_dim=future_latent_dim,    # 未来编码维度
            history_latent_dim=history_latent_dim,  # 历史编码维度
            actor_hidden_dims=actor_hidden_dims,    # Actor隐藏层维度
            activation=activation_fn,               # 激活函数
            layer_norm=layer_norm,                  # 是否使用LayerNorm
            future_encoder_dims=future_encoder_dims,  # 未来编码器维度
            future_attention_heads=future_attention_heads,  # 注意力头数
            future_dropout=future_dropout,          # Dropout概率
            temporal_embedding_dim=temporal_embedding_dim,  # 时间嵌入维度
            use_history_encoder=True,               # 使用历史编码器
            use_motion_encoder=True,                # 使用动作编码器
            tanh_encoder_output=kwargs.get('tanh_encoder_output', False),  # 是否使用tanh输出
            **actor_kwargs                          # 展开额外参数
        )
        # 统计Actor网络参数量
        print(f"ActorCriticFuture: actor network has {sum(p.numel() for p in self.actor.parameters()) / 1e6:.2f}M parameters")
        
        # 存储Critic的维度
        self.num_motion_observations = num_motion_observations  # 存储动作观测维度
        self.num_single_motion_obs = int(num_motion_observations / num_motion_steps)  # 单步动作观测维度
        
        # Critic网络（使用特权观测，不包含未来动作）
        self.critic_motion_encoder = MotionEncoder(activation_fn, self.num_single_motion_obs, num_motion_steps, motion_latent_dim)  # Critic动作编码器
        
        # Critic输入维度计算
        # Critic观测 = 特权观测（无未来动作），需要重新计算输入维度
        critic_input_dim = num_critic_observations - num_motion_observations + motion_latent_dim + self.num_single_motion_obs
        
        critic_layers = []                          # 初始化Critic层列表
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))  # 第一层线性层
        critic_layers.append(activation_fn)         # 添加激活函数
        
        # 构建Critic隐藏层
        for l in range(len(critic_hidden_dims)):    # 遍历隐藏层维度
            if l == len(critic_hidden_dims) - 1:    # 如果是最后一层
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))  # 输出层：输出价值（单值）
            else:                                   # 如果不是最后一层
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))  # 线性层
                if layer_norm and l == len(critic_hidden_dims) - 2:  # 如果是倒数第二层且启用layer_norm
                    critic_layers.append(nn.LayerNorm(critic_hidden_dims[l + 1]))  # 添加LayerNorm
                critic_layers.append(activation_fn) # 添加激活函数
                
        self.critic = nn.Sequential(*critic_layers)  # 使用Sequential包装所有层

        # 动作噪声（用于探索）
        if self.fix_action_std:                     # 如果固定标准差
            self.init_action_std_tensor = torch.tensor(action_std)  # 创建标准差张量
            self.std = nn.Parameter(self.init_action_std_tensor, requires_grad=False)  # 不可学习的参数
        else:                                       # 如果可学习标准差
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))  # 可学习的标准差参数
            
        self.distribution = None                    # 初始化动作分布为None
        # 禁用参数验证以加速
        Normal.set_default_validate_args = False
        
    @staticmethod
    def init_weights(sequential, scales):
        """
        初始化序列模块的权重。
        
        Args:
            sequential: 序列模块
            scales: 每层的增益系数列表
        """
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        """
        重置网络状态（用于循环网络，此处为空实现）。
        
        Args:
            dones: 完成标志
        """
        pass                                        # 非循环网络无需重置

    def forward(self, observations, critic_observations=None, actions=None, **kwargs):
        """
        DDP友好的前向传播（参考rsl_rl.modules.actor_critic.ActorCritic.forward）。
        
        Args:
            observations: Actor观测
            critic_observations: Critic观测（可选）
            actions: 动作（用于计算log_prob）
            
        Returns:
            actions: 采样的动作
            actions_log_prob: 动作的log概率
            value: 状态价值
            mu: 动作均值
            sigma: 动作标准差
            entropy: 动作分布熵
        """
        self.update_distribution(observations)      # 更新动作分布

        if actions is None:                         # 如果没有提供动作
            actions = self.distribution.sample()    # 从分布中采样动作

        actions_log_prob = self.get_actions_log_prob(actions)  # 计算动作的对数概率
        entropy = self.entropy                      # 获取分布熵
        mu = self.action_mean                       # 获取动作均值
        sigma = self.action_std                     # 获取动作标准差

        value = None                                # 初始化价值为None
        if critic_observations is not None:         # 如果提供了Critic观测
            value = self.evaluate(critic_observations, **kwargs)  # 评估状态价值
        
        return actions, actions_log_prob, value, mu, sigma, entropy  # 返回所有结果
    
    @property
    def action_mean(self):
        """动作分布的均值。"""
        return self.distribution.mean               # 返回分布的均值

    @property
    def action_std(self):
        """动作分布的标准差。"""
        return self.distribution.stddev             # 返回分布的标准差
    
    @property
    def entropy(self):
        """动作分布的熵。"""
        return self.distribution.entropy().sum(dim=-1)  # 返回分布的熵（对动作维度求和）

    def update_distribution(self, observations):
        """
        更新动作分布。
        
        Args:
            observations: 观测输入
        """
        mean = self.actor(observations)             # 通过Actor网络计算动作均值
        self.distribution = Normal(mean, mean*0. + self.std)  # 创建正态分布（均值为网络输出，标准差为可学习参数）

    def act(self, observations, **kwargs):
        """
        根据观测采样动作。
        
        Args:
            observations: 观测输入
            
        Returns:
            采样的动作
        """
        self.update_distribution(observations)      # 更新动作分布
        return self.distribution.sample()           # 从分布中采样
    
    def get_actions_log_prob(self, actions):
        """
        获取动作的log概率。
        
        Args:
            actions: 动作
            
        Returns:
            log概率
        """
        return self.distribution.log_prob(actions).sum(dim=-1)  # 计算对数概率并对动作维度求和

    def act_inference(self, observations, eval=False, **kwargs):
        """
        推理模式（确定性动作）。
        
        Args:
            observations: 观测输入
            eval: 是否评估模式
            
        Returns:
            动作均值
        """
        actions_mean = self.actor(observations)     # 直接计算动作均值（确定性输出）
        return actions_mean                         # 返回动作均值

    def evaluate(self, critic_observations, **kwargs):
        """
        Critic评估状态价值。
        
        Critic使用特权观测（不含未来动作）。
        
        Args:
            critic_observations: Critic观测（特权信息）
            
        Returns:
            状态价值
        """
        # Critic使用特权观测（无未来动作）
        motion_obs = critic_observations[:, :self.num_motion_observations]  # 提取动作观测部分
        motion_single_obs = critic_observations[:, :self.num_single_motion_obs]  # 提取单步动作观测
        motion_latent = self.critic_motion_encoder(motion_obs)  # 通过动作编码器
        
        # 组合特征：特权观测（去掉动作部分）+ 单步动作 + 动作编码
        backbone_input = torch.cat([
            critic_observations[:, self.num_motion_observations:],  # 特权信息（去掉动作观测）
            motion_single_obs,                      # 单步动作观测
            motion_latent                           # 动作编码特征
        ], dim=1)                                   # 在特征维度上拼接
        
        value = self.critic(backbone_input)         # 通过Critic网络
        return value                                # 返回状态价值
    
    def reset_std(self, std, num_actions, device):
        """
        重置动作标准差。
        
        Args:
            std: 新的标准差
            num_actions: 动作维度
            device: 设备
        """
        new_std = std * torch.ones(num_actions, device=device)  # 创建新的标准差张量
        self.std.data = new_std.data                # 更新标准差参数
        
    def if_fix_std(self):
        """是否固定动作标准差。"""
        return self.fix_action_std                  # 返回是否固定标准差的标志
    
    def get_moe_aux_loss(self):
        """
        获取MoE负载均衡辅助损失（如果使用MoE Actor）。
        
        Returns:
            辅助损失张量
        """
        if self.use_moe and hasattr(self.actor, 'get_moe_aux_loss'):  # 如果使用MoE且Actor有该方法
            return self.actor.get_moe_aux_loss()    # 返回MoE辅助损失
        return torch.tensor(0.0, device=next(self.parameters()).device)  # 否则返回0
    
    def get_gating_weights(self):
        """
        获取门控权重用于分析/可视化。
        
        Returns:
            门控权重张量或None
        """
        if self.use_moe and hasattr(self.actor, 'get_gating_weights'):  # 如果使用MoE且Actor有该方法
            return self.actor.get_gating_weights()  # 返回门控权重
        return None                                 # 否则返回None


# =============================================================================
# Mixture of Experts (MoE) Implementation
# 混合专家模型实现
# =============================================================================

class MoELayer(nn.Module):
    """
    混合专家层 - 使用门控机制动态选择专家网络。
    
    设计原则：门控 + 1个专家 ≈ 原始MLP参数量
    
    工作原理：
    1. 门控网络根据输入决定各专家的权重
    2. 选择top-k个专家进行计算
    3. 根据门控权重融合各专家的输出
    4. 计算负载均衡损失，鼓励均匀使用各专家
    
    Attributes:
        gating: 门控网络，输出各专家的权重
        experts: 专家网络列表
        top_k: 激活的专家数
        temperature: 门控softmax温度
    """
    
    def __init__(self, 
                 input_dim,
                 output_dim,
                 num_experts=4,
                 expert_hidden_dims=[512, 384, 192],
                 gating_hidden_dim=128,
                 top_k=2,
                 temperature=1.0,
                 activation=nn.SiLU(),
                 layer_norm=False):
        """
        初始化MoE层。
        
        Args:
            input_dim: 输入特征维度
            output_dim: 输出维度（动作维度）
            num_experts: 专家网络数量
            expert_hidden_dims: 每个专家的隐藏层维度
            gating_hidden_dim: 门控网络隐藏层维度
            top_k: 激活的专家数（None表示使用所有）
            temperature: 门控softmax温度
            activation: 激活函数
            layer_norm: 是否使用LayerNorm
        """
        super().__init__()                          # 调用父类初始化
        
        self.input_dim = input_dim                  # 存储输入维度
        self.output_dim = output_dim                # 存储输出维度
        self.num_experts = num_experts              # 存储专家数量
        self.top_k = top_k if top_k is not None else num_experts  # 存储top_k值
        self.temperature = temperature              # 存储温度参数
        
        # 门控网络：轻量级MLP，输入->隐藏->专家数
        self.gating = nn.Sequential(
            nn.Linear(input_dim, gating_hidden_dim),  # 输入层
            activation,                             # 激活函数
            nn.Linear(gating_hidden_dim, num_experts)  # 输出层（输出各专家的权重）
        )
        
        # 门控网络使用较小权重初始化，增加稳定性
        for layer in self.gating:                   # 遍历门控网络的层
            if isinstance(layer, nn.Linear):        # 如果是线性层
                nn.init.xavier_uniform_(layer.weight, gain=0.5)  # Xavier初始化，增益0.5
                nn.init.zeros_(layer.bias)          # 偏置初始化为0
        
        # 专家网络：每个专家是一个独立的MLP
        self.experts = nn.ModuleList()              # 创建ModuleList存储专家
        for _ in range(num_experts):                # 遍历创建每个专家
            expert_layers = []                      # 当前专家的层列表
            dims = [input_dim] + expert_hidden_dims + [output_dim]  # 维度列表
            
            for i in range(len(dims) - 1):          # 遍历每一层
                expert_layers.append(nn.Linear(dims[i], dims[i + 1]))  # 添加线性层
                if i < len(dims) - 2:               # 如果不是最后一层
                    if layer_norm and i == len(dims) - 3:  # 如果是倒数第二层且启用layer_norm
                        expert_layers.append(nn.LayerNorm(dims[i + 1]))  # 添加LayerNorm
                    expert_layers.append(activation)  # 添加激活函数
            
            self.experts.append(nn.Sequential(*expert_layers))  # 将专家添加到列表
        
        # 存储辅助损失和门控权重（用于后续访问）
        self._moe_aux_loss = None                   # 初始化辅助损失为None
        self._gating_weights = None                 # 初始化门控权重为None
        
    def forward(self, x):
        """
        前向传播。
        
        Args:
            x: 输入张量，形状为 (batch_size, input_dim)
        
        Returns:
            output: 输出张量，形状为 (batch_size, output_dim)
        """
        batch_size = x.shape[0]                     # 获取batch大小
        
        # 计算门控权重
        logits = self.gating(x) / self.temperature  # (batch, num_experts)，应用温度缩放
        
        # Top-k门控
        if self.top_k < self.num_experts:           # 如果top_k小于专家总数
            # 选择top-k个专家
            top_k_logits, top_k_indices = torch.topk(logits, self.top_k, dim=-1)  # 获取top-k的logits和索引
            gates = torch.zeros_like(logits)        # 创建全零门控权重
            gates.scatter_(-1, top_k_indices, F.softmax(top_k_logits, dim=-1))  # 对top-k应用softmax
        else:                                       # 如果使用所有专家
            gates = F.softmax(logits, dim=-1)       # 对所有专家应用softmax
        
        # 存储门控权重用于分析
        self._gating_weights = gates.detach()       # 存储（ detach避免梯度）
        
        # 计算负载均衡损失（鼓励均匀使用各专家）
        # loss = num_experts * sum_j (fraction_j * fraction_j)
        # 其中fraction_j是专家j的平均门控权重
        fraction = gates.mean(dim=0)                # 计算每个专家的平均门控权重 (num_experts,)
        load_balancing_loss = self.num_experts * (fraction ** 2).sum()  # 负载均衡损失
        self._moe_aux_loss = load_balancing_loss    # 存储辅助损失
        
        # 计算各专家的输出
        expert_outputs = []                         # 初始化专家输出列表
        for expert in self.experts:                 # 遍历每个专家
            expert_outputs.append(expert(x))        # 计算当前专家的输出
        expert_outputs = torch.stack(expert_outputs, dim=1)  # 堆叠成张量 (batch, num_experts, output_dim)
        
        # 加权组合各专家输出
        gates_expanded = gates.unsqueeze(-1)        # 扩展门控权重维度 (batch, num_experts, 1)
        output = (gates_expanded * expert_outputs).sum(dim=1)  # 加权求和 (batch, output_dim)
        
        return output                               # 返回最终输出
    
    def get_aux_loss(self):
        """
        获取负载均衡辅助损失。
        
        Returns:
            辅助损失张量
        """
        return self._moe_aux_loss if self._moe_aux_loss is not None else torch.tensor(0.0)  # 返回辅助损失或0
    
    def get_gating_weights(self):
        """
        获取门控权重用于可视化/分析。
        
        Returns:
            门控权重张量
        """
        return self._gating_weights                 # 返回门控权重


class ActorFutureMoE(nn.Module):
    """
    基于MoE的Actor网络 - 使用混合专家模型替代标准MLP。
    
    所有编码器（动作、历史、未来）与ActorFuture相同，
    仅将主网络替换为MoE层。
    
    Attributes:
        motion_encoder: 动作编码器
        history_encoder: 历史编码器
        future_encoder: 未来动作编码器
        actor_backbone: MoE主干网络
    """
    
    def __init__(self, num_observations,
                 num_motion_observations,
                 num_priop_observations,
                 num_motion_steps,
                 num_future_observations,
                 num_future_steps,
                 motion_latent_dim,
                 future_latent_dim,
                 num_actions,
                 actor_hidden_dims,  # 未使用，保留API兼容
                 activation, 
                 history_latent_dim,
                 num_history_steps,
                 layer_norm=False,
                 future_encoder_dims=[256, 256, 128],
                 future_attention_heads=4,
                 future_dropout=0.1,
                 temporal_embedding_dim=64,
                 use_history_encoder=True,
                 use_motion_encoder=True,
                 tanh_encoder_output=False,
                 # MoE特定参数
                 use_moe=True,
                 num_experts=4,
                 expert_hidden_dims=[512, 384, 192],
                 gating_hidden_dim=128,
                 moe_topk=2,
                 moe_temperature=1.0,
                 **kwargs):
        """
        初始化MoE Actor网络。
        
        观测结构：motion_obs, priop_obs, history_obs, future_obs
        
        Args:
            num_observations: 总观测维度
            num_motion_observations: 动作观测总维度
            num_priop_observations: 本体感知观测维度
            num_motion_steps: 动作观测时间步数
            num_future_observations: 未来观测总维度
            num_future_steps: 未来观测时间步数
            motion_latent_dim: 动作编码维度
            future_latent_dim: 未来编码维度
            num_actions: 动作维度
            actor_hidden_dims: 未使用（API兼容）
            activation: 激活函数
            history_latent_dim: 历史编码维度
            num_history_steps: 历史时间步数
            layer_norm: 是否使用LayerNorm
            future_encoder_dims: 未来编码器维度
            future_attention_heads: 注意力头数
            future_dropout: Dropout概率
            temporal_embedding_dim: 时间嵌入维度
            use_history_encoder: 是否使用历史编码器
            use_motion_encoder: 是否使用动作编码器
            tanh_encoder_output: 输出是否使用tanh
            use_moe: 是否使用MoE（固定为True）
            num_experts: 专家数量
            expert_hidden_dims: 专家隐藏层维度
            gating_hidden_dim: 门控隐藏层维度
            moe_topk: 激活专家数
            moe_temperature: 门控温度
        """
        super().__init__()                          # 调用父类初始化
        self.num_observations = num_observations    # 存储总观测维度
        self.num_actions = num_actions              # 存储动作维度
        self.num_motion_observations = num_motion_observations  # 存储动作观测维度
        self.num_priop_observations = num_priop_observations    # 存储本体感知维度
        self.num_motion_steps = num_motion_steps    # 存储动作时间步数
        self.num_history_steps = num_history_steps  # 存储历史时间步数
        self.num_future_observations = num_future_observations  # 存储未来观测维度
        self.num_future_steps = num_future_steps    # 存储未来时间步数
        self.use_history_encoder = use_history_encoder  # 是否使用历史编码器
        self.use_motion_encoder = use_motion_encoder    # 是否使用动作编码器
        
        # 计算单步观测维度
        self.num_single_motion_observations = int(num_motion_observations / num_motion_steps)  # 单步动作观测维度
        self.num_single_priop_observations = num_priop_observations  # 单步本体感知维度
        
        # 计算历史观测总维度
        num_single_history_observations = num_motion_observations + num_priop_observations  # 单步历史观测维度
        history_size = num_single_history_observations * num_history_steps  # 历史观测总维度
        
        # 计算未来单步观测维度
        self.num_single_future_observations = int(num_future_observations / num_future_steps) if num_future_observations > 0 else 0  # 单步未来观测维度
        self.future_latent_dim = future_latent_dim  # 存储未来编码维度
        
        # 动作编码器（与ActorFuture相同）
        if self.use_motion_encoder:                 # 如果使用动作编码器
            self.motion_encoder = MotionEncoder(activation, self.num_single_motion_observations, self.num_motion_steps, motion_latent_dim)  # 创建MotionEncoder
        else:                                       # 如果不使用
            self.motion_encoder = nn.Identity()     # 使用恒等映射
            motion_latent_dim = self.num_single_motion_observations  # 编码后维度等于输入维度
        
        # 历史编码器（与ActorFuture相同）
        if self.use_history_encoder:                # 如果使用历史编码器
            self.history_encoder = HistoryEncoder(activation, num_single_history_observations, self.num_history_steps, history_latent_dim)  # 创建HistoryEncoder
        else:                                       # 如果不使用
            self.history_encoder = nn.Identity()    # 使用恒等映射
            history_latent_dim = history_size       # 编码后维度等于输入维度
            
        # 未来动作编码器（与ActorFuture相同）
        if self.num_single_future_observations > 0:  # 如果有未来观测
            self.future_encoder = FutureMotionEncoder(
                activation,                         # 激活函数
                self.num_single_future_observations - 1,  # 单步未来观测维度（-1为mask指示器）
                self.num_future_steps,              # 未来时间步数
                future_latent_dim,                  # 输出维度
                attention_heads=future_attention_heads,  # 注意力头数（兼容）
                dropout=future_dropout,             # Dropout概率
                temporal_embedding_dim=temporal_embedding_dim  # 时间嵌入维度（兼容）
            )
        else:                                       # 如果没有未来观测
            self.future_encoder = None              # 未来编码器为None
        
        # 主Actor网络 - MoE主干
        # 计算输入维度（与ActorFuture相同）
        input_dim = (motion_latent_dim + self.num_single_motion_observations + 
                    self.num_single_priop_observations + history_latent_dim + future_latent_dim)
        
        # 创建MoE层替代标准MLP
        self.actor_backbone = MoELayer(
            input_dim=input_dim,                    # 输入维度
            output_dim=num_actions,                 # 输出维度（动作维度）
            num_experts=num_experts,                # 专家数量
            expert_hidden_dims=expert_hidden_dims,  # 专家隐藏层维度
            gating_hidden_dim=gating_hidden_dim,    # 门控隐藏层维度
            top_k=moe_topk,                         # top-k选择
            temperature=moe_temperature,            # 门控温度
            activation=activation,                  # 激活函数
            layer_norm=layer_norm                   # 是否使用LayerNorm
        )
        
        # 存储辅助损失
        self._moe_aux_loss = None                   # 初始化辅助损失为None

    def forward(self, obs, hist_encoding: bool = False):
        """
        前向传播（与ActorFuture相同）。
        
        Args:
            obs: 完整观测向量
            hist_encoding: 是否只返回历史编码
            
        Returns:
            backbone_output: 动作输出
        """
        # 解析观测（与ActorFuture相同）
        current_size = self.num_motion_observations + self.num_priop_observations  # 当前观测总维度
        
        motion_obs = obs[:, :self.num_motion_observations]  # 提取动作观测
        single_motion_obs = obs[:, :self.num_single_motion_observations]  # 提取单步动作观测
        priop_obs = obs[:, self.num_motion_observations:current_size]  # 提取本体感知观测
        
        history_start = current_size                # 历史观测起始位置
        history_size = self.num_history_steps * current_size  # 历史观测大小
        history_end = history_start + history_size  # 历史观测结束位置
        history_obs = obs[:, history_start:history_end]  # 提取历史观测
        
        future_start = history_end                  # 未来观测起始位置
        future_end = future_start + self.num_future_observations  # 未来观测结束位置
        future_obs = obs[:, future_start:future_end]  # 提取未来观测
        
        # 编码所有组件
        motion_latent = self.motion_encoder(motion_obs)  # 动作编码
        history_latent = self.history_encoder(history_obs)  # 历史编码
        
        if self.future_encoder is not None and self.num_future_observations > 0 and future_obs.shape[1] > 0:  # 如果有未来观测
            future_obs_reshaped = future_obs.reshape(-1, self.num_future_steps, self.num_single_future_observations)  # reshape
            future_latent = self.future_encoder(future_obs_reshaped)  # 未来编码
        else:                                       # 如果没有未来观测
            future_latent = torch.zeros(obs.shape[0], self.future_latent_dim, device=obs.device)  # 创建零向量
        
        # 组合所有特征
        backbone_input = torch.cat([
            single_motion_obs,                      # 单步动作观测
            priop_obs,                              # 本体感知观测
            motion_latent,                          # 动作编码特征
            history_latent,                         # 历史编码特征
            future_latent                           # 未来编码特征
        ], dim=1)                                   # 特征维度拼接
        
        # MoE主干
        backbone_output = self.actor_backbone(backbone_input)  # 通过MoE网络
        
        # 存储辅助损失供外部访问
        self._moe_aux_loss = self.actor_backbone.get_aux_loss()  # 获取并存储MoE辅助损失
        
        return backbone_output                      # 返回动作输出
    
    def get_moe_aux_loss(self):
        """
        获取MoE负载均衡损失。
        
        Returns:
            辅助损失张量
        """
        return self._moe_aux_loss if self._moe_aux_loss is not None else torch.tensor(0.0, device=next(self.parameters()).device)  # 返回辅助损失或0
    
    def get_gating_weights(self):
        """
        获取门控权重用于分析。
        
        Returns:
            门控权重张量
        """
        return self.actor_backbone.get_gating_weights()  # 返回门控权重


# =============================================================================
# Transformer Backbone Implementation
# Transformer主干网络实现
# =============================================================================

class TransformerBackbone(nn.Module):
    """Transformer-based backbone for policy network.
    
    Replaces the standard MLP backbone with a Transformer encoder.
    The input is treated as a sequence of tokens (patching approach).
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output dimension (num_actions)
        d_model: Transformer model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network (default: 4*d_model)
        dropout: Dropout rate
        activation: Activation function
    """
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 d_model=256,
                 nhead=8,
                 num_layers=2,
                 dim_feedforward=None,
                 dropout=0.1,
                 activation=nn.SiLU(),
                 layer_norm=False):
        super().__init__()                          # 调用父类初始化
        
        self.input_dim = input_dim                  # 存储输入维度
        self.output_dim = output_dim                # 存储输出维度
        self.d_model = d_model                      # 存储Transformer模型维度
        
        if dim_feedforward is None:                 # 如果未指定前馈维度
            dim_feedforward = 4 * d_model           # 默认使用4倍模型维度
        
        # Input embedding: project input to d_model
        self.input_embedding = nn.Linear(input_dim, d_model)  # 输入嵌入层：将输入投影到d_model维度
        
        # Positional encoding (learnable)
        # We treat the input as a single token, but use Transformer for self-attention
        # Actually, let's use a different approach: patch the input into tokens
        # For simplicity, we use the full input as one token and add transformer layers
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)  # 可学习位置编码，随机初始化
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,                        # 模型维度
            nhead=nhead,                            # 注意力头数
            dim_feedforward=dim_feedforward,        # 前馈网络维度
            dropout=dropout,                        # Dropout概率
            activation='gelu',                      # GELU激活（Transformer标准）
            batch_first=True,                       # batch维度在前
            norm_first=False                        # Post-norm（标准配置）
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  # 创建Transformer编码器
        
        # Output head
        output_layers = []                          # 输出层列表
        output_layers.append(nn.Linear(d_model, d_model // 2))  # 第1层：d_model -> d_model/2
        if layer_norm:                              # 如果启用LayerNorm
            output_layers.append(nn.LayerNorm(d_model // 2))  # 添加LayerNorm
        output_layers.append(activation)            # 添加激活函数
        output_layers.append(nn.Linear(d_model // 2, output_dim))  # 第2层：d_model/2 -> output_dim
        
        self.output_head = nn.Sequential(*output_layers)  # 使用Sequential包装输出层
        
        # Initialize weights
        self._init_weights()                        # 调用权重初始化方法
        
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():               # 遍历所有模块
            if isinstance(module, nn.Linear):       # 如果是线性层
                nn.init.xavier_uniform_(module.weight)  # Xavier均匀初始化
                if module.bias is not None:         # 如果有偏置
                    nn.init.zeros_(module.bias)     # 偏置初始化为0
            elif isinstance(module, nn.LayerNorm):  # 如果是LayerNorm
                nn.init.ones_(module.weight)        # 权重初始化为1
                nn.init.zeros_(module.bias)         # 偏置初始化为0
    
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, input_dim)
        
        Returns:
            output: (batch_size, output_dim)
        """
        # Embed input: (batch, input_dim) -> (batch, d_model)
        x = self.input_embedding(x)                 # 输入嵌入
        
        # Add positional encoding and reshape for transformer
        # Treat each sample as a sequence of 1 token: (batch, 1, d_model)
        x = x.unsqueeze(1) + self.pos_encoding      # 添加位置编码并增加序列维度
        
        # Transformer encoding
        x = self.transformer(x)                     # 通过Transformer编码器
        
        # Remove sequence dimension: (batch, 1, d_model) -> (batch, d_model)
        x = x.squeeze(1)                            # 移除序列维度
        
        # Output head
        output = self.output_head(x)                # 通过输出头
        
        return output                               # 返回输出


class ActorFutureTransformer(nn.Module):
    """Actor network with Transformer backbone.
    
    All encoders (Motion, History, Future) are identical to ActorFuture.
    Only the backbone is replaced with Transformer.
    """
    
    def __init__(self, num_observations,
                 num_motion_observations,
                 num_priop_observations,
                 num_motion_steps,
                 num_future_observations,
                 num_future_steps,
                 motion_latent_dim,
                 future_latent_dim,
                 num_actions,
                 actor_hidden_dims,  # Not used for Transformer, kept for API compatibility
                 activation, 
                 history_latent_dim,
                 num_history_steps,
                 layer_norm=False,
                 future_encoder_dims=[256, 256, 128],
                 future_attention_heads=4,
                 future_dropout=0.1,
                 temporal_embedding_dim=64,
                 use_history_encoder=True,
                 use_motion_encoder=True,
                 tanh_encoder_output=False,
                 # Transformer specific parameters
                 use_transformer=True,
                 d_model=256,
                 nhead=8,
                 num_transformer_layers=2,
                 transformer_dropout=0.1,
                 **kwargs):
        """
        observation structure: motion_obs, priop_obs, history_obs, future_obs
        """
        super().__init__()                          # 调用父类初始化
        self.num_observations = num_observations    # 存储总观测维度
        self.num_actions = num_actions              # 存储动作维度
        self.num_motion_observations = num_motion_observations  # 存储动作观测维度
        self.num_priop_observations = num_priop_observations    # 存储本体感知维度
        self.num_motion_steps = num_motion_steps    # 存储动作时间步数
        self.num_history_steps = num_history_steps  # 存储历史时间步数
        self.num_future_observations = num_future_observations  # 存储未来观测维度
        self.num_future_steps = num_future_steps    # 存储未来时间步数
        self.use_history_encoder = use_history_encoder  # 是否使用历史编码器
        self.use_motion_encoder = use_motion_encoder    # 是否使用动作编码器
        
        # Calculate single step sizes
        self.num_single_motion_observations = int(num_motion_observations / num_motion_steps)  # 单步动作观测维度
        self.num_single_priop_observations = num_priop_observations  # 单步本体感知维度
        
        # Calculate history size
        num_single_history_observations = num_motion_observations + num_priop_observations  # 单步历史观测维度
        history_size = num_single_history_observations * num_history_steps  # 历史观测总维度
        
        # Calculate future single step size
        self.num_single_future_observations = int(num_future_observations / num_future_steps) if num_future_observations > 0 else 0  # 单步未来观测维度
        self.future_latent_dim = future_latent_dim  # 存储未来编码维度
        
        # Motion encoder (same as ActorFuture)
        if self.use_motion_encoder:                 # 如果使用动作编码器
            self.motion_encoder = MotionEncoder(activation, self.num_single_motion_observations, self.num_motion_steps, motion_latent_dim)  # 创建MotionEncoder
        else:                                       # 如果不使用
            self.motion_encoder = nn.Identity()     # 使用恒等映射
            motion_latent_dim = self.num_single_motion_observations  # 编码后维度等于输入维度
        
        # History encoder (same as ActorFuture)
        if self.use_history_encoder:                # 如果使用历史编码器
            self.history_encoder = HistoryEncoder(activation, num_single_history_observations, self.num_history_steps, history_latent_dim)  # 创建HistoryEncoder
        else:                                       # 如果不使用
            self.history_encoder = nn.Identity()    # 使用恒等映射
            history_latent_dim = history_size       # 编码后维度等于输入维度
            
        # Future motion encoder (same as ActorFuture)
        if self.num_single_future_observations > 0:  # 如果有未来观测
            self.future_encoder = FutureMotionEncoder(
                activation,                         # 激活函数
                self.num_single_future_observations - 1,  # 单步未来观测维度（-1为mask指示器）
                self.num_future_steps,              # 未来时间步数
                future_latent_dim,                  # 输出维度
                attention_heads=future_attention_heads,  # 注意力头数（兼容）
                dropout=future_dropout,             # Dropout概率
                temporal_embedding_dim=temporal_embedding_dim  # 时间嵌入维度（兼容）
            )
        else:                                       # 如果没有未来观测
            self.future_encoder = None              # 未来编码器为None
        
        # Main actor network - Transformer backbone
        # 计算输入维度（与ActorFuture相同）
        input_dim = (motion_latent_dim + self.num_single_motion_observations + 
                    self.num_single_priop_observations + history_latent_dim + future_latent_dim)
        
        # 创建TransformerBackbone替代标准MLP
        self.actor_backbone = TransformerBackbone(
            input_dim=input_dim,                    # 输入维度
            output_dim=num_actions,                 # 输出维度（动作维度）
            d_model=d_model,                        # Transformer模型维度
            nhead=nhead,                            # 注意力头数
            num_layers=num_transformer_layers,      # Transformer层数
            dropout=transformer_dropout,            # Dropout概率
            activation=activation,                  # 激活函数
            layer_norm=layer_norm                   # 是否使用LayerNorm
        )

    def forward(self, obs, hist_encoding: bool = False):
        # Parse observations (identical to ActorFuture)
        current_size = self.num_motion_observations + self.num_priop_observations  # 当前观测总维度
        
        motion_obs = obs[:, :self.num_motion_observations]  # 提取动作观测
        single_motion_obs = obs[:, :self.num_single_motion_observations]  # 提取单步动作观测
        priop_obs = obs[:, self.num_motion_observations:current_size]  # 提取本体感知观测
        
        history_start = current_size                # 历史观测起始位置
        history_size = self.num_history_steps * current_size  # 历史观测大小
        history_end = history_start + history_size  # 历史观测结束位置
        history_obs = obs[:, history_start:history_end]  # 提取历史观测
        
        future_start = history_end                  # 未来观测起始位置
        future_end = future_start + self.num_future_observations  # 未来观测结束位置
        future_obs = obs[:, future_start:future_end]  # 提取未来观测
        
        # Encode all components
        motion_latent = self.motion_encoder(motion_obs)  # 动作编码
        history_latent = self.history_encoder(history_obs)  # 历史编码
        
        if self.future_encoder is not None and self.num_future_observations > 0 and future_obs.shape[1] > 0:  # 如果有未来观测
            future_obs_reshaped = future_obs.reshape(-1, self.num_future_steps, self.num_single_future_observations)  # reshape
            future_latent = self.future_encoder(future_obs_reshaped)  # 未来编码
        else:                                       # 如果没有未来观测
            future_latent = torch.zeros(obs.shape[0], self.future_latent_dim, device=obs.device)  # 创建零向量
        
        # Combine all features
        backbone_input = torch.cat([
            single_motion_obs,                      # 单步动作观测
            priop_obs,                              # 本体感知观测
            motion_latent,                          # 动作编码特征
            history_latent,                         # 历史编码特征
            future_latent                           # 未来编码特征
        ], dim=1)                                   # 特征维度拼接
        
        # Transformer backbone
        backbone_output = self.actor_backbone(backbone_input)  # 通过Transformer网络
        
        return backbone_output                      # 返回动作输出
