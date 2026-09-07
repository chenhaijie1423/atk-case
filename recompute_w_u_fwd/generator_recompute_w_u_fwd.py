# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
import random

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig


K_INDEX = 0
V_INDEX = 1
BETA_INDEX = 2
A_INDEX = 3
G_INDEX = 4
CU_SEQLENS_INDEX = 5
CHUNK_INDICES_INDEX = 6
CHUNK_SIZE_INDEX = 7
BETA_TYPE_INDEX = 8
IS_FIX_INDEX = 9
QKV_TYPE_INDEX = 10


@GENERATOR_REGISTRY.register("generator_recompute_w_u_fwd")
class RecomputeWUFwdGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        # kdtype -> v, A; g 跟随 beta dtype（kernel 中 beta/g 共用 betaType）
        qkv_type = case_config.inputs[K_INDEX].dtype
        case_config.inputs[V_INDEX].dtype = qkv_type
        case_config.inputs[A_INDEX].dtype = qkv_type
        case_config.inputs[QKV_TYPE_INDEX].range_values = qkv_type

        # kernel bin 仅支持 beta/g 与 k/v/A 同 dtype（fp16/bf16）或 fp32，
        # 交叉组合（如 k=fp16 + beta/g=bf16）无编译产物，SelectBin 会失败
        beta_type = case_config.inputs[BETA_INDEX].dtype
        if beta_type != qkv_type and beta_type != "fp32":
            beta_type = random.choice((qkv_type, "fp32"))
        case_config.inputs[BETA_INDEX].dtype = beta_type
        case_config.inputs[G_INDEX].dtype = beta_type
        case_config.inputs[BETA_TYPE_INDEX].range_values = beta_type

        is_fix = case_config.inputs[IS_FIX_INDEX].range_values
        H = random.randint(1, 20)
        if not is_fix:
            # 变长模式：B = 1，cu_seqlens 累积和决定 T
            B = 1
            T = random.randint(1, 65536)
        else:
            B = random.randint(1, 16)
            T = random.randint(1, 4096)

        K = 128
        V = random.choice((128, 256))
        chunk_size = case_config.inputs[CHUNK_SIZE_INDEX].range_values
        if isinstance(chunk_size, list):
            chunk_size = chunk_size[0]

        # HV = n * HK, pick n from {1, 2, 3, 4, 5, 6} for dimension split testing
        gva_mode = random.choice((0, 1))
        n_ratio = 1 if gva_mode == 0 else random.choice((2, 3, 4, 5, 6))
        HK = H
        HV = HK * n_ratio

        # k: [B, HK, T, K]; v: [B, HV, T, V]
        # beta, g: [B, HV, T]; A: [B, HV, T, chunk_size]
        # outputs: w: [B, HV, T, K]; u: [B, HV, T, V]
        case_config.inputs[K_INDEX].shape = [B, HK, T, K]
        case_config.inputs[V_INDEX].shape = [B, HV, T, V]
        case_config.inputs[BETA_INDEX].shape = [B, HV, T]
        case_config.inputs[A_INDEX].shape = [B, HV, T, chunk_size]
        case_config.inputs[G_INDEX].shape = [B, HV, T]

        return case_config


# 合并模板：前 29 条为白盒用例（覆盖所有分支），后 31 条为网络用例
# 每条模板字段：qkv_type, beta_type, B, HK, HV, T, V, chunk_size, is_fix,
#               cu_seqlens_len, cu_seqlens_val（变长时各 seq 长度，定长忽略）
WHITEBOX_TEMPLATES = [
    # --- Group A: TilingKey / dtype / chunk_size baseline (noGVA, fixed) ---
    ("bf16", "fp32", 1, 4, 4, 64, 128, 64, True, 1, 1),
    ("bf16", "fp32", 1, 4, 4, 64, 256, 64, True, 1, 1),
    ("bf16", "fp32", 1, 4, 4, 128, 128, 128, True, 1, 1),
    ("fp16", "fp32", 1, 4, 4, 64, 128, 64, True, 1, 1),
    ("bf16", "bf16", 1, 4, 4, 64, 128, 64, True, 1, 1),
    ("fp16", "fp16", 1, 4, 4, 64, 128, 64, True, 1, 1),
    ("bf16", "fp32", 1, 4, 4, 128, 256, 128, True, 1, 1),
    # --- Group B: GVA ratios (fixed) ---
    ("bf16", "fp32", 1, 4, 8, 128, 128, 64, True, 1, 1),
    ("bf16", "fp32", 1, 4, 12, 256, 256, 128, True, 1, 1),
    ("bf16", "bf16", 1, 4, 16, 128, 128, 64, True, 1, 1),
    ("fp16", "fp32", 1, 2, 12, 128, 256, 64, True, 1, 1),
    # --- Group C: Partial / tail chunk (fixed) ---
    ("bf16", "fp32", 1, 4, 4, 96, 128, 64, True, 1, 1),
    ("bf16", "fp32", 1, 4, 4, 192, 256, 128, True, 1, 1),
    ("bf16", "fp32", 1, 4, 8, 100, 128, 64, True, 1, 1),
    # --- Group D: Multi-batch (fixed) ---
    ("bf16", "fp32", 4, 4, 4, 128, 128, 64, True, 1, 1),
    ("fp16", "fp16", 2, 4, 8, 256, 256, 128, True, 1, 1),
    # --- Group E: Variable-length ---
    ("bf16", "fp32", 1, 4, 4, 64, 128, 64, False, 1, 64),
    ("bf16", "fp32", 1, 4, 4, 100, 128, 64, False, 1, 100),
    ("bf16", "fp32", 1, 4, 4, 256, 256, 128, False, 2, 128),
    ("fp16", "fp32", 1, 4, 4, 192, 128, 64, False, 3, 64),
    ("bf16", "fp32", 1, 4, 4, 30, 128, 64, False, 1, 30),
    ("bf16", "fp32", 1, 4, 8, 128, 256, 64, False, 2, 64),
    # --- Group F: A5 ring-slot reuse (tasks > GM_RING_DEPTH=8) ---
    ("bf16", "fp32", 1, 2, 2, 512, 128, 64, True, 1, 1),
    ("fp16", "fp32", 1, 4, 8, 640, 256, 64, True, 1, 1),
    ("bf16", "bf16", 1, 2, 2, 320, 128, 64, False, 5, 64),
    ("bf16", "fp32", 1, 2, 2, 1024, 128, 128, True, 1, 1),
    ("bf16", "fp32", 8, 4, 4, 64, 128, 64, True, 1, 1),
    ("fp16", "fp32", 1, 4, 8, 200, 256, 64, False, 2, 100),
    # --- Group G: additional varlen + cs128 combo ---
    ("bf16", "fp32", 1, 4, 4, 200, 128, 128, False, 1, 200),
]

# 网络用例模板：覆盖真实模型常见配置（定长为主，含 GVA / 多 dtype / 多 batch）
NETWORK_TEMPLATES = [
    # (qkv_type, beta_type, B, HK, HV, T, V, chunk_size, is_fix, cu_len, cu_val)
    # B=1 大 T 模型场景
    ("bf16", "fp32", 1, 4, 4, 32768, 128, 64, True, 1, 1),
    ("bf16", "fp32", 1, 4, 4, 32768, 256, 64, True, 1, 1),
    ("bf16", "fp32", 1, 8, 8, 32768, 128, 64, True, 1, 1),
    ("bf16", "fp32", 1, 8, 8, 32768, 256, 64, True, 1, 1),
    ("bf16", "fp32", 1, 4, 8, 32768, 128, 64, True, 1, 1),
    ("bf16", "fp32", 1, 4, 8, 32768, 256, 64, True, 1, 1),
    ("bf16", "fp32", 1, 4, 4, 32768, 128, 128, True, 1, 1),
    ("bf16", "fp32", 1, 8, 8, 32768, 128, 128, True, 1, 1),
    ("fp16", "fp32", 1, 4, 4, 32768, 128, 64, True, 1, 1),
    ("fp16", "fp32", 1, 4, 4, 32768, 256, 64, True, 1, 1),
    ("fp16", "fp16", 1, 4, 4, 32768, 128, 64, True, 1, 1),
    ("fp16", "fp16", 1, 8, 8, 32768, 256, 64, True, 1, 1),
    # B=8 中 T 模型场景
    ("bf16", "fp32", 8, 4, 4, 4096, 128, 64, True, 1, 1),
    ("bf16", "fp32", 8, 4, 4, 4096, 256, 64, True, 1, 1),
    ("bf16", "fp32", 8, 8, 8, 4096, 128, 64, True, 1, 1),
    ("bf16", "fp32", 8, 8, 8, 4096, 256, 64, True, 1, 1),
    ("bf16", "fp32", 8, 4, 8, 4096, 128, 64, True, 1, 1),
    ("bf16", "fp32", 8, 4, 8, 4096, 256, 64, True, 1, 1),
    ("bf16", "fp32", 8, 4, 4, 4096, 128, 128, True, 1, 1),
    ("bf16", "fp32", 8, 8, 8, 4096, 128, 128, True, 1, 1),
    ("fp16", "fp32", 8, 4, 4, 4096, 128, 64, True, 1, 1),
    ("fp16", "fp32", 8, 4, 4, 4096, 256, 64, True, 1, 1),
    ("fp16", "fp16", 8, 4, 4, 4096, 128, 64, True, 1, 1),
    ("fp16", "fp16", 8, 8, 8, 4096, 256, 64, True, 1, 1),
    # 变长网络场景
    ("bf16", "fp32", 1, 4, 4, 32768, 128, 64, False, 4, 8192),
    ("bf16", "fp32", 1, 8, 8, 32768, 256, 64, False, 4, 8192),
    ("bf16", "fp32", 1, 4, 8, 32768, 128, 64, False, 4, 8192),
    ("fp16", "fp32", 1, 4, 4, 32768, 128, 64, False, 4, 8192),
    ("bf16", "fp32", 1, 8, 8, 32768, 128, 128, False, 2, 16384),
    ("bf16", "fp32", 1, 4, 8, 32768, 256, 64, False, 2, 16384),
    ("fp16", "fp16", 1, 8, 8, 32768, 256, 64, False, 4, 8192),
]

# 合并模板：前 29 条白盒 + 后 31 条网络，共 60 条
MERGED_TEMPLATES = WHITEBOX_TEMPLATES + NETWORK_TEMPLATES


@GENERATOR_REGISTRY.register("generator_recompute_w_u_fwd_network")
class RecomputeWUFwdNetworkGenerator(CaseGenerator):
    _case_idx = 0

    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        # 按类级计数器轮转固定模板，强制覆盖全部参数（忽略 ATK 随机值）
        template = MERGED_TEMPLATES[self._case_idx % len(MERGED_TEMPLATES)]
        RecomputeWUFwdNetworkGenerator._case_idx += 1

        (qkv_type, beta_type, B, HK, HV, T, V, chunk_size,
         is_fix, cu_len, cu_val) = template

        K = 128

        # kdtype -> v, A; g 跟随 beta dtype（kernel 中 beta/g 共用 betaType）
        case_config.inputs[K_INDEX].dtype = qkv_type
        case_config.inputs[V_INDEX].dtype = qkv_type
        case_config.inputs[A_INDEX].dtype = qkv_type
        case_config.inputs[QKV_TYPE_INDEX].range_values = qkv_type
        case_config.inputs[BETA_INDEX].dtype = beta_type
        case_config.inputs[G_INDEX].dtype = beta_type
        case_config.inputs[BETA_TYPE_INDEX].range_values = beta_type
        case_config.inputs[IS_FIX_INDEX].range_values = is_fix
        case_config.inputs[CHUNK_SIZE_INDEX].range_values = chunk_size

        # k: [B, HK, T, K]; v: [B, HV, T, V]
        # beta, g: [B, HV, T]; A: [B, HV, T, chunk_size]
        # outputs: w: [B, HV, T, K]; u: [B, HV, T, V]
        case_config.inputs[K_INDEX].shape = [B, HK, T, K]
        case_config.inputs[V_INDEX].shape = [B, HV, T, V]
        case_config.inputs[BETA_INDEX].shape = [B, HV, T]
        case_config.inputs[A_INDEX].shape = [B, HV, T, chunk_size]
        case_config.inputs[G_INDEX].shape = [B, HV, T]

        # 变长场景：cu_seqlens shape=[cu_len]，range=[cu_val, cu_val]（确定性）
        # 定长场景：cu_seqlens 为占位符，executor 会忽略
        case_config.inputs[CU_SEQLENS_INDEX].shape = [cu_len]
        case_config.inputs[CU_SEQLENS_INDEX].range_values = [cu_val, cu_val]

        return case_config
