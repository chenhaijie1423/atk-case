# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
import torch
import sys
import os
from typing import Optional, Tuple

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from fla_npu.ops import ascendc as ascendc_ops

# 兜底：确保同目录的 CPU 参考实现可导入（ATK 加载 executor 时通常已包含本目录）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_w_u_fwd_cpu import recompute_w_u_fwd_cpu

def create_gate_g(B: int, H: int, T: int, gtype):
    lo, hi = -5e-2, -5e-5
    span = hi - lo
    margin = max(span * 1e-7, 1e-12)
    g_t = torch.linspace(
        float(hi) - margin,
        float(lo) + margin,
        T,
        dtype=torch.float64,
    )
    return g_t.unsqueeze(0).unsqueeze(0).expand(B, H, T).contiguous().to(gtype)


def cumsum_cu_seqlens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.nn.functional.pad(
        torch.cumsum(cu_seqlens, dim=0),
        (1, 0),
        value=0,
    )


def prepare_chunk_indices(
    cu_seqlens: list[int],
    chunk_size: int
) -> list[int]:
    """
    基于 cu_seqlens (list[int]) 生成 chunk 索引。

    返回扁平化 list[int] (如 [s0, c0, s1, c1, ...])，与 npu 算子的
    chunk_indices 输入（[num_chunks, 2] 展平语义）一致。

    逻辑：
    1. 计算每个序列的长度: lens[i] = cu_seqlens[i+1] - cu_seqlens[i]
    2. 计算每个序列需要的 chunk 数: ceil(lens[i] / chunk_size)
    3. 生成对应的 (sequence_id, chunk_id) 对并展开
    """
    indices = []

    for i in range(len(cu_seqlens) - 1):
        start = cu_seqlens[i]
        end = cu_seqlens[i + 1]
        length = end - start

        if length <= 0:
            continue

        num_chunks = (length + chunk_size - 1) // chunk_size

        for chunk_id in range(num_chunks):
            indices.append(i)
            indices.append(chunk_id)

    return indices


def recompute_w_u_fwd_torch(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64,
    benchmark: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """head-major 布局与 NPU 算子一致，直接调用 CPU 参考实现。"""
    cu_seqlens_tensor = torch.tensor(cu_seqlens, dtype=torch.int64) if cu_seqlens is not None else None

    w, u = recompute_w_u_fwd_cpu(
        k, v, beta, A, g, cu_seqlens_tensor, chunk_size,
        benchmark=benchmark
    )

    return w, u


@register("executor_recompute_w_u_fwd")
class FunctionApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.qkv_type = None

    def cpu(self, input_data: InputDataset, with_output: bool = False):
        k = input_data.kwargs["k"]
        v = input_data.kwargs["v"]
        beta = input_data.kwargs["beta"]
        A = input_data.kwargs["A"]
        g = input_data.kwargs["g"]
        cu_seqlens = input_data.kwargs.get("cu_seqlens", None)
        chunk_size = input_data.kwargs["chunk_size"]

        w, u = recompute_w_u_fwd_torch(
            k, v, beta, A, g, cu_seqlens, chunk_size
        )

        if self.qkv_type == "bf16":
            w = w.to(torch.bfloat16)
            u = u.to(torch.bfloat16)
        if self.qkv_type == "fp16":
            w = w.to(torch.float16)
            u = u.to(torch.float16)

        return w, u

    def cpu_benchmark(self, input_data: InputDataset, with_output: bool = False):
        k = input_data.kwargs["k"].to(torch.float64)
        v = input_data.kwargs["v"].to(torch.float64)
        beta = input_data.kwargs["beta"].to(torch.float64)
        A = input_data.kwargs["A"].to(torch.float64)
        g = input_data.kwargs["g"].to(torch.float64)
        cu_seqlens = input_data.kwargs.get("cu_seqlens", None)
        chunk_size = input_data.kwargs["chunk_size"]

        w, u = recompute_w_u_fwd_torch(
            k, v, beta, A, g, cu_seqlens, chunk_size,
            benchmark=True
        )

        return w, u

    def npu_recompute_w_u_fwd(self, input_data: InputDataset, with_output: bool = False):
        k = input_data.kwargs["k"]
        v = input_data.kwargs["v"]
        beta = input_data.kwargs["beta"]
        A = input_data.kwargs["A"]
        g = input_data.kwargs["g"]
        cu_seqlens = input_data.kwargs.get("cu_seqlens", None)
        chunk_indices = input_data.kwargs.get("chunk_indices", None)
        chunk_size = input_data.kwargs["chunk_size"]

        w, u = ascendc_ops.npu_recompute_w_u_fwd(
            k, v, beta, A, chunk_size,
            g=g, gk=None, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )

        return w, u

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        k = input_data.kwargs["k"]
        if self.device == "npu":
            return self.npu_recompute_w_u_fwd(input_data, with_output)
        elif k.dtype == torch.float32:
            return self.cpu_benchmark(input_data, with_output)
        else:
            return self.cpu(input_data, with_output)

    def init_by_input_data(self, input_data: InputDataset):
        torch.manual_seed(self.task_result.case_config.id)
        B, HK, T_json, K = input_data.kwargs["k"].shape
        HV = input_data.kwargs["v"].shape[1]
        V = input_data.kwargs["v"].shape[3]
        k = input_data.kwargs["k"]
        v = input_data.kwargs["v"]
        beta = input_data.kwargs["beta"]
        A = input_data.kwargs["A"]
        g = input_data.kwargs["g"]
        cu_seqlens = input_data.kwargs["cu_seqlens"]
        chunk_indices = input_data.kwargs["chunk_indices"]
        chunk_size = input_data.kwargs["chunk_size"]

        is_fix = input_data.kwargs["is_fix"]
        self.qkv_type = input_data.kwargs["qkv_type"]

        qkv_type = input_data.kwargs["k"].dtype
        beta_type = input_data.kwargs["beta"].dtype

        # is_fix = False -> 变长：cu_seqlens 累积和决定 T，chunk_indices 重推导
        if not is_fix:
            cu_seqlens = cumsum_cu_seqlens(cu_seqlens).tolist()
            T = cu_seqlens[-1]
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
            # k: [B, HK, T, K]; v, A: [B, HV, T, V/ C]; beta, g: [B, HV, T]
            k = (torch.rand(B, HK, T, K, dtype=torch.bfloat16) * 5e-2).to(qkv_type)
            v = (torch.rand(B, HV, T, V, dtype=torch.bfloat16) * 5e-2).to(qkv_type)
            A = (torch.rand(B, HV, T, chunk_size, dtype=torch.bfloat16) * 5e-2).to(qkv_type)
            beta = (torch.rand(B, HV, T, dtype=torch.bfloat16) * 0.8 + 0.1).to(beta_type)
            g = create_gate_g(B, HV, T, torch.bfloat16).to(beta_type)
        else:
            cu_seqlens = None
            chunk_indices = None
            T = T_json
            k = (torch.randn(B, HK, T, K, dtype=torch.bfloat16) * 5e-2).to(qkv_type)
            v = (torch.randn(B, HV, T, V, dtype=torch.bfloat16) * 5e-2).to(qkv_type)
            A = (torch.randn(B, HV, T, chunk_size, dtype=torch.bfloat16) * 5e-2).to(qkv_type)
            beta = (torch.rand(B, HV, T, dtype=torch.bfloat16) * 0.8 + 0.1).to(beta_type)
            g = create_gate_g(B, HV, T, torch.bfloat16).to(beta_type)    # g 必须递减且为负数

        if self.device == "npu":
            k = k.npu()
            v = v.npu()
            beta = beta.npu()
            A = A.npu()
            g = g.npu()
            # cu_seqlens / chunk_indices 走 aclnn int_array，必须是 Python int 序列；
            # 传 NPU tensor 会在 _AclIntArray 转换处报错
            if cu_seqlens is not None:
                cu_seqlens = [int(x) for x in cu_seqlens]
            if chunk_indices is not None:
                chunk_indices = [int(x) for x in chunk_indices]

        input_data.kwargs["k"] = k
        input_data.kwargs["v"] = v
        input_data.kwargs["beta"] = beta
        input_data.kwargs["A"] = A
        input_data.kwargs["g"] = g
        input_data.kwargs["cu_seqlens"] = cu_seqlens
        input_data.kwargs["chunk_indices"] = chunk_indices
        input_data.kwargs["chunk_size"] = chunk_size
        input_data.kwargs.pop("is_fix", None)
        input_data.kwargs.pop("beta_type", None)
        input_data.kwargs.pop("qkv_type", None)
