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


@GENERATOR_REGISTRY.register("generator_recompute_w_u_fwd_network")
class RecomputeWUFwdNetworkGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        # kdtype -> v, A; g 跟随 beta dtype（kernel 中 beta/g 共用 betaType）
        qkv_type = case_config.inputs[K_INDEX].dtype
        case_config.inputs[V_INDEX].dtype = qkv_type
        case_config.inputs[A_INDEX].dtype = qkv_type
        case_config.inputs[QKV_TYPE_INDEX].range_values = qkv_type

        # kernel bin 仅支持 beta/g 与 k/v/A 同 dtype（fp16/bf16）或 fp32
        beta_type = case_config.inputs[BETA_INDEX].dtype
        if beta_type != qkv_type and beta_type != "fp32":
            beta_type = random.choice((qkv_type, "fp32"))
        case_config.inputs[BETA_INDEX].dtype = beta_type
        case_config.inputs[G_INDEX].dtype = beta_type
        case_config.inputs[BETA_TYPE_INDEX].range_values = beta_type

        # 网络用例定长场景
        case_config.inputs[IS_FIX_INDEX].range_values = True
        is_fix = True
        H = random.choice((4, 8))
        B = random.choice((1, 8))
        T = 32768 if B == 1 else 4096

        K = 128
        V = random.choice((128, 256))
        chunk_size = case_config.inputs[CHUNK_SIZE_INDEX].range_values
        if isinstance(chunk_size, list):
            chunk_size = chunk_size[0]

        # HV = n * HK, pick n from {1, 2, 3} for dimension split testing
        gva_mode = random.choice((0, 1))
        n_ratio = 1 if gva_mode == 0 else random.choice((2, 3))
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
