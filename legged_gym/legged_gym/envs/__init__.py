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

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .base.legged_robot import LeggedRobot

from .base.humanoid import Humanoid
from .base.humanoid_mimic import HumanoidMimic

from .g1.g1_mimic_config import G1MimicCfg, G1MimicCfgPPO
from .g1.g1_mimic import G1Mimic

# DeepMimic (for teleoperation)
from .g1.g1_mimic_distill import G1MimicDistill
from .g1.g1_mimic_distill_config import G1MimicPrivCfg, G1MimicPrivCfgPPO, G1MimicStuCfg, G1MimicStuCfgDAgger
from .g1.g1_mimic_distill_config import G1MimicStuRLCfg, G1MimicStuRLCfgDAgger
from .g1.g1_mimic_distill_config import G1MimicPrivLimbWeightCfg, G1MimicPrivLimbWeightCfgPPO
from .g1.g1_mimic_distill_config import G1MimicStuLimbWeightCfg, G1MimicStuLimbWeightCfgDAgger

from .g1.g1_mimic_future import G1MimicFuture
from .g1.g1_mimic_future_config import G1MimicStuFutureCfg, G1MimicStuFutureCfgDAgger

# MoE (Mixture of Experts) configuration
from .g1.g1_mimic_moe_config import G1MimicStuFutureMoECfg, G1MimicStuFutureMoECfgDAgger

# Transformer configuration
from .g1.g1_mimic_transformer_config import (
    G1MimicStuFutureTrans2xCfg,
    G1MimicStuFutureTrans2xCfgDAgger,
    G1MimicStuFutureTrans4xCfg,
    G1MimicStuFutureTrans4xCfgDAgger,
)

from .g1.g1_mimic_hyfeat import G1MimicHyFeat
from .g1.g1_mimic_hyfeat_config import (
    G1HyMotion100kPrivCfg,
    G1HyMotion100kPrivCfgPPO,
    G1HyMotion100kStuHyFeatCfg,
    G1HyMotion100kStuHyFeatCfgDAgger,
)

from legged_gym.gym_utils.task_registry import task_registry


# DeepMimic (for teleoperation)
task_registry.register("g1_mimic", G1Mimic, G1MimicCfg(), G1MimicCfgPPO())
task_registry.register("g1_stu_mimic", G1MimicDistill, G1MimicStuCfg(), G1MimicStuCfgDAgger())
task_registry.register("g1_priv_mimic", G1MimicDistill, G1MimicPrivCfg(), G1MimicPrivCfgPPO())
task_registry.register("g1_stu_rl", G1MimicDistill, G1MimicStuRLCfg(), G1MimicStuRLCfgDAgger())
task_registry.register("g1_priv_mimic_limbw", G1MimicDistill, G1MimicPrivLimbWeightCfg(), G1MimicPrivLimbWeightCfgPPO())
task_registry.register("g1_stu_mimic_limbw", G1MimicDistill, G1MimicStuLimbWeightCfg(), G1MimicStuLimbWeightCfgDAgger())
task_registry.register("g1_stu_future", G1MimicFuture, G1MimicStuFutureCfg(), G1MimicStuFutureCfgDAgger())

# MoE (Mixture of Experts) - Student policy with MoE backbone
task_registry.register("g1_stu_future_moe", G1MimicFuture, G1MimicStuFutureMoECfg(), G1MimicStuFutureMoECfgDAgger())

# Transformer - Student policy with Transformer backbone (2x params)
task_registry.register("g1_stu_future_trans2x", G1MimicFuture, G1MimicStuFutureTrans2xCfg(), G1MimicStuFutureTrans2xCfgDAgger())

# Transformer - Student policy with Transformer backbone (4x params)
task_registry.register("g1_stu_future_trans4x", G1MimicFuture, G1MimicStuFutureTrans4xCfg(), G1MimicStuFutureTrans4xCfgDAgger())

# HYMotion100k (HY features + curriculum masking on mimic_obs)
task_registry.register("g1_priv_hymotion100k", G1MimicHyFeat, G1HyMotion100kPrivCfg(), G1HyMotion100kPrivCfgPPO())
task_registry.register("g1_stu_hymotion100k_hyfeat", G1MimicHyFeat, G1HyMotion100kStuHyFeatCfg(), G1HyMotion100kStuHyFeatCfgDAgger())
