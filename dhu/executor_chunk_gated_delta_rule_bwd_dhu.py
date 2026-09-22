"""chunk_gated_delta_rule_bwd_dhu 的 ATK executor。

ATK 按 chunk_gated_delta_rule_bwd_dhu.yaml 生成张量与属性骨架后，
init_by_input_data 按参考分布重建业务输入（g 沿 T 单调递减，避免
exp(bg_last - bg) 上溢）并写回 input_data.kwargs：
- is_fix=True  → 定长模式（cu_seqlens/chunk_indices 均为 None）；
- is_fix=False → 变长模式（B 固定为 1，确定性生成 cu_seqlens 与
  chunk_indices，段长总和精确等于 T）。
CPU 标杆为本文件内自包含实现（定长与变长路径，不跨文件调用）：ATK 双标杆
（cv_fused_double_benchmark）下 CPU 执行两次，低位宽轮（输入 fp16/bf16）
采用 npu 模式（fp32 计算并复刻 kernel 数据流的中间 DT 量化），升精度轮
（输入 fp32）以纯 fp64 计算；NPU DUT 走
fla_npu.ops.ascendc.chunk_gated_delta_rule_bwd_dhu。
"""
# numactl --cpunodebind=0,1 --membind=0,1
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import copy
import torch
import os
import ctypes

# 避免在循环中反复调用 torch.set_num_threads() 导致线程池反复销毁/重建而卡死
torch.set_num_threads(16)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor, AclIntArray, nnopbase, Int64


OP_NAME = "chunk_gated_delta_rule_bwd_dhu"

_LN2 = 0.69314718055994530942

def _gate(shape):
    """生成沿 T 维单调递减的 GDN gate，避免 exp 溢出。"""
    data = torch.rand(tuple(int(x) for x in shape) dtype=torch.float32) * 0.01 + 0.001
    data = -torch.cumsum(data, dim=-1)
    return data

def _randn(shape, calc_dtype: torch.dtype, scale: float = 0.05):
    """生成确定性正态分布输入；先量化到原始 dtype，再转到计算 dtype。"""
    data = torch.randn(tuple(int(x) for x in shape), dtype=torch.float32) * float(scale)
    return data.to(calc_dtype)

def _to_int(value, default=None) -> int:
    """把 ATK 传入的 int/str/tensor 标量统一转成 Python int（None 时取 default）。"""
    if value is None:
        if default is None:
            raise ValueError("必需的整型属性缺失（None）。")
        return default
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().reshape(-1)[0].item())
    return int(value)


def _to_float(value, default=None) -> float:
    """把 ATK 传入的 float/str/tensor 标量统一转成 Python float（None 时取 default）。"""
    if value is None:
        if default is None:
            raise ValueError("必需的浮点属性缺失（None）。")
        return default
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1)[0].item())
    return float(value)


def _to_bool(value) -> bool:
    """把 ATK 传入的 bool/str/tensor 标量统一转成 Python bool。"""
    if isinstance(value, torch.Tensor):
        return bool(value.detach().cpu().reshape(-1)[0].item())
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _as_int_list(values: Optional[Sequence[int]]) -> Optional[List[int]]:
    """把 list/tensor 形式的整型元数据统一转成 List[int]。"""
    if values is None:
        return None
    if isinstance(values, torch.Tensor):
        return [int(x) for x in values.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in values]


def _generate_cu_seqlens(total_length: int, num_seqs: int) -> List[int]:
    """确定性生成变长 cu_seqlens：各段总和精确等于 total_length。

    公平切分后按「小配大」交错排列，避免相邻段长度过于接近；
    不做段长夹紧，任意 total_length >= num_seqs 均可精确对齐。
    """
    num_seqs = max(1, int(num_seqs))
    lengths = [
        (total_length * (i + 1)) // num_seqs - (total_length * i) // num_seqs
        for i in range(num_seqs)
    ]
    sorted_l = sorted(lengths)
    seq_lengths: List[int] = []
    i, j = 0, len(sorted_l) - 1
    while i <= j:
        if i == j:
            seq_lengths.append(sorted_l[i])
        else:
            seq_lengths.append(sorted_l[i])
            seq_lengths.append(sorted_l[j])
        i += 1
        j -= 1
    cu_seqlens = [0]
    for seq_len in seq_lengths:
        cu_seqlens.append(cu_seqlens[-1] + seq_len)
    if cu_seqlens[-1] != total_length:
        raise ValueError(
            f"_generate_cu_seqlens: 各段之和 {cu_seqlens[-1]} 与 total_length={total_length} 不一致。")
    return cu_seqlens


def _prepare_chunk_indices(cu_seqlens: List[int], chunk_size: int) -> List[int]:
    """由 cu_seqlens 生成扁平 chunk_indices：[seq_idx, chunk_idx, ...]。"""
    chunk_indices: List[int] = []
    for seq_idx in range(len(cu_seqlens) - 1):
        seq_len = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        if seq_len <= 0:
            continue
        chunk_num = (seq_len + chunk_size - 1) // chunk_size
        for chunk_idx in range(chunk_num):
            chunk_indices.append(seq_idx)
            chunk_indices.append(chunk_idx)
    return chunk_indices


def _build_case_inputs(
    B: int,
    HK: int,
    HV: int,
    T: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    seed: int = _DEFAULT_SEED,
) -> dict:
    """在 CPU 上按参考分布重建业务输入（固定种子，各节点逐位一致）。

    q/k: [B,HK,T,K]；w: [B,HV,T,K]；dO/dv: [B,HV,T,V]；
    g: [B,HV,T] 沿 T 单调递减（fp32），保证 exp(bg_last - bg) <= 1 不上溢。
    """
    return {
        "q": _randn((B, HK, T, K), dtype),
        "k": _randn((B, HK, T, K), dtype),
        "w": _randn((B, HV, T, K), dtype),
        "dO": _randn((B, HV, T, V), dtype),
        "dv": _randn((B, HV, T, V), dtype),
        "g": _gate((B, HV, T)),
    }

def chunk_gated_delta_rule_bwd_dhu_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    dO: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor,
    scale: float,
    chunk_size: int,
    use_exp2: bool = False,
    cu_seqlens: Optional[List[int]] = None,
    chunk_indices: Optional[List[int]] = None,
) -> tuple:
    t0 = time.perf_counter()
    dtype_ = q.dtype
    # 双标杆精度分层：低位宽轮（fp16/bf16 输入）fp32 计算，升精度轮（fp32 输入）fp64 计算。
    compute_dtype = torch.float64 if dtype_ == torch.float32 else torch.float32
    npu_mode = dtype_ != torch.float32

    def round_dt(x: torch.Tensor) -> torch.Tensor:
        """npu 模式：经 DT 表示域量化后回到计算精度。"""
        if not npu_mode:
            return x
        return x.to(dtype_).to(compute_dtype)
    device = q.device
    B, Hk, T, K = q.shape
    Hv = dO.shape[1]
    V = dO.shape[-1]
    BT = int(chunk_size)
    scale_f = float(scale)
    if Hk <= 0 or Hv % Hk != 0:
        raise ValueError(f"GVA: Hv % Hk == 0 required, Hk={Hk}, Hv={Hv}")
    hv_per_hk = Hv // Hk

    if cu_seqlens is not None:
        if chunk_indices is None:
            raise ValueError("变长模式要求 cu_seqlens 与 chunk_indices 同时提供。")
        seq_total = cu_seqlens[-1]
        if seq_total > T:
            raise ValueError(f"cu_seqlens 末元 {seq_total} 超过时间维 T={T}。")
        NT = len(chunk_indices) // 2
        varlen = True
    else:
        NT = (T + BT - 1) // BT
        varlen = False

    qf = q.to(compute_dtype)
    kf = k.to(compute_dtype)
    wf = w.to(compute_dtype)
    dof = dO.to(compute_dtype)
    dvf = dv.to(compute_dtype)
    gf = g.to(compute_dtype)

    def gate_exp(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x * _LN2) if use_exp2 else torch.exp(x)
    # dh/dv2 输出 dtype 与 q 一致：低位宽轮（bf16/fp16）与 NPU 输出 dtype
    # 对齐（ATK 双标杆的 local/remote dtype 检查），升精度轮为 fp32。
    # dh_states 中间态缓冲（compute_dtype）：循环内按 compute_dtype 写，避免
    # 逐 chunk 的 dtype 转换；循环结束后一次性 to(dtype_) 得到 dh。
    dh_states = torch.empty(B, Hv, NT, K, V, device=device, dtype=compute_dtype)
    # 变长模式下未被 chunk 覆盖的 [seq_total, T) 位置保持 dv 原值。
    dv2 = dv.clone() if varlen else torch.zeros(B, Hv, T, V, device=device, dtype=dtype_)
    num_tokens = len(cu_seqlens) - 1 if varlen else 1
    b_dh_buffers = torch.zeros(B, Hv, num_tokens, K, V, device=device, dtype=compute_dtype)
    b_dh = torch.zeros(B, Hv, K, V, device=device, dtype=compute_dtype)

    # ---- 循环外预计算：参考 recompute_w_u_fwd_cpu 的 batched bmm 模式 ----
    # GVA 头扩展：repeat_interleave 一次替代 NT 次 index_select
    if hv_per_hk > 1:
        kf = kf.repeat_interleave(hv_per_hk, dim=1)
        qf = qf.repeat_interleave(hv_per_hk, dim=1)
    # gf 转 float32 整序列一次完成
    g_seq = gf[:, :, :T].to(torch.float32)  # [B, Hv, T]

    # 定长模式下，将满 chunk 数据 reshape 为 [B, Hv, n_full, BT, D] 批量预计算 gate 和 term1；
    # 变长模式 chunk 不连续，走原 per-chunk 路径。
    _ragged = T % BT
    n_full = T // BT if not varlen else 0
    term1_full = gate_factor_full = exp_bg_last_full = dv2_full_pre = None
    if n_full > 0:
        T_full = n_full * BT
        # ---- Gate 批量预计算（3 次 gate_exp 替代 3*n_full 次）----
        g_full = g_seq[:, :, :T_full].reshape(B, Hv, n_full, BT)                     # [B, Hv, n_full, BT]
        last_pos = torch.arange(BT - 1, T_full, BT, device=device)                      # [n_full]
        bg_last_full = g_seq[:, :, last_pos]                                            # [B, Hv, n_full]
        exp_bg_last_full = gate_exp(bg_last_full)                                       # [B, Hv, n_full]
        exp_g_full = gate_exp(g_full)                                                   # [B, Hv, n_full, BT]
        gate_factor_full = gate_exp(bg_last_full.unsqueeze(-1) - g_full).unsqueeze(-1)   # [B, Hv, n_full, BT, 1]

        # ---- term1 批量预计算（1 次 bmm 替代 n_full 次小 matmul）----
        qf_chunks = qf[:, :, :T_full].reshape(B, Hv, n_full, BT, K)                   # [B, Hv, n_full, BT, K]
        dof_chunks = dof[:, :, :T_full].reshape(B, Hv, n_full, BT, V)                 # [B, Hv, n_full, BT, V]

        # q_gated = q_blk^T * gate_exp(b_g): 先在 chunk 维度上做元素级乘，再 reshape 为 bmm 输入
        q_gated_f = (qf_chunks.transpose(-1, -2) * exp_g_full.unsqueeze(3))            # [B, Hv, n_full, K, BT]
        do_flat = dof_chunks.reshape(B * Hv * n_full, BT, V)                            # [BHv*N, BT, V]
        q_flat = q_gated_f.reshape(B * Hv * n_full, K, BT)                              # [BHv*N, K, BT]
        term1_f = round_dt(torch.bmm(q_flat, do_flat)) * scale_f                          # [BHv*N, K, V]
        term1_full = term1_f.reshape(B, Hv, n_full, K, V)                                # [B, Hv, n_full, K, V]

        # 满 chunk 数据 reshape 为 [B, Hv, n_full, BT, D]，循环内 O(1) 索引
        kf_full = kf[:, :, :T_full].reshape(B, Hv, n_full, BT, K)
        wf_full = wf[:, :, :T_full].reshape(B, Hv, n_full, BT, K)
        dvf_full = dvf[:, :, :T_full].reshape(B, Hv, n_full, BT, V)
        # dv2 延迟写回缓冲：循环内按 chunk 写 compute_dtype（省 .to(dtype_) kernel），
        # 循环后一次性批量转换 + 写回 dv2（1 个 kernel 替代 n_full 个）。
        dv2_full_pre = torch.empty(B, Hv, n_full, BT, V, device=device, dtype=compute_dtype)
    # ---- 预展开常量，消除循环内 unsqueeze / transpose ----
    if n_full > 0:
        exp_bg_last_expanded = exp_bg_last_full.unsqueeze(-1).unsqueeze(-1)  # [B, Hv, n_full, 1, 1]
        wf_full_T = wf_full.transpose(-1, -2).contiguous()                  # [B, Hv, n_full, K, BT]

    def _per_chunk_step(gs, ge, global_last_idx, b_dh):
        """ragged tail / varlen per-chunk 路径（保留原始 gate_exp 计算顺序）。"""
        k_blk = kf[:, :, gs:ge, :]
        q_blk_ = qf[:, :, gs:ge, :]
        w_blk = wf[:, :, gs:ge, :]
        b_do = dof[:, :, gs:ge, :]
        b_dv_existing = dvf[:, :, gs:ge, :]
        bg_last = g_seq[:, :, global_last_idx]
        b_g = g_seq[:, :, gs:ge]
        gate_factor = gate_exp(bg_last.unsqueeze(-1) - b_g).unsqueeze(-1)
        b_dh_rd = round_dt(b_dh)
        b_dv = k_blk @ b_dh_rd
        b_dv = round_dt(b_dv)
        b_dv = b_dv * gate_factor + b_dv_existing
        dv2[:, :, gs:ge, :] = b_dv.to(dtype_)
        b_dh_for_update = b_dh * gate_exp(bg_last).unsqueeze(-1).unsqueeze(-1)
        b_q_gated = q_blk_.transpose(-1, -2) * gate_exp(b_g).unsqueeze(-2)
        b_q_gated_rd = round_dt(b_q_gated)
        term1 = b_q_gated_rd @ b_do
        term1 = round_dt(term1)
        term1 = term1 * scale_f
        b_dv_rd = round_dt(b_dv)
        term2 = w_blk.transpose(-1, -2) @ b_dv_rd
        term2 = round_dt(term2)
        return b_dh_for_update + term1 - term2

    # ---- 主循环：按模式拆分，循环内零分支 ----
    if varlen:
        for i_t in range(NT - 1, -1, -1):
            i_n = chunk_indices[i_t * 2]
            block_idx_in_token = chunk_indices[i_t * 2 + 1]
            bos = cu_seqlens[i_n]
            token_length = cu_seqlens[i_n + 1] - bos
            start_t = block_idx_in_token * BT
            end_t = min(start_t + BT, token_length)
            gs = bos + start_t
            ge = bos + end_t
            b_dh = b_dh_buffers[:, :, i_n, :, :]
            dh_states[:, :, i_t, :, :] = b_dh
            last_idx = min((block_idx_in_token + 1) * BT, token_length) - 1
            global_last_idx = bos + last_idx
            b_dh = _per_chunk_step(gs, ge, global_last_idx, b_dh)
            b_dh_buffers[:, :, i_n, :, :] = b_dh
    else:
        # 定长模式：先 ragged tail（若有），再满 chunk 批量路径
        if _ragged > 0:
            i_t = n_full
            gs = i_t * BT
            ge = T
            dh_states[:, :, i_t, :, :] = b_dh
            b_dh = _per_chunk_step(gs, ge, T - 1, b_dh)

        # 满 chunk：统一顺序循环（fp64 下 round_dt 为恒等，与 npu 轮同一套路径）
        for i_t in range(n_full - 1, -1, -1):
            dh_states[:, :, i_t, :, :] = b_dh
            b_dh_rd = round_dt(b_dh)
            b_dv = kf_full[:, :, i_t] @ b_dh_rd
            b_dv = round_dt(b_dv)
            b_dv = b_dv * gate_factor_full[:, :, i_t] + dvf_full[:, :, i_t]
            dv2_full_pre[:, :, i_t] = b_dv
            b_dh_for_update = b_dh * exp_bg_last_expanded[:, :, i_t]
            b_dv_rd = round_dt(b_dv)
            term2 = wf_full_T[:, :, i_t] @ b_dv_rd
            term2 = round_dt(term2)
            b_dh = b_dh_for_update + term1_full[:, :, i_t] - term2

    if dv2_full_pre is not None:
        T_full = n_full * BT
        dv2[:, :, :T_full, :] = dv2_full_pre.reshape(B, Hv, T_full, V).to(dtype_)

    dh = dh_states.to(dtype_)
    print(f"[{OP_NAME}] {varlen} {dtype_} {B} {T} chunk_gated_delta_rule_bwd_dhu_golden 总耗时: "
          f"{time.perf_counter() - t0:.6f} s")
    return dh, dv2


@register("executor_chunk_gated_delta_rule_bwd_dhu")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.high_precision = self.device == "cpu"

    def init_by_input_data(self, input_data: InputDataset):
        torch.manual_seed(self.task_result.case_config.id)
        t0 = time.perf_counter()
        q = input_data.kwargs["q"]
        dO = input_data.kwargs["dO"]
        device = q.device
        B, HK, T, K = q.shape
        _, HV, _, V = dO.shape
        input_data.kwargs["scale"] = scale = 1.0 / math.sqrt(K)
        chunk_size = _to_int(input_data.kwargs.get("chunkSize"), default=64)
        use_exp2 = _to_bool(input_data.kwargs.get("use_exp2", False))
        is_fix = _to_bool(input_data.kwargs.get("is_fix", True))
        t1 = time.perf_counter()
        # ATK 按 YAML 生成的张量取值不满足本算子输入约束（g 需沿 T 单调递减，
        # 否则 exp(bg_last - bg) 上溢），因此按参考分布以固定种子在 CPU 重建
        # 全部输入，再按节点设备搬运，保证 CPU 标杆与 NPU DUT 输入逐位一致。
        if is_fix:
            # 定长模式：cu_seqlens/chunk_indices 均为 None。
            B_use, cu_seqlens = int(B), None
            chunk_indices = None
        else:
            # 变长模式：算子约束仅支持 B=1；确定性生成 cu_seqlens（各段
            # 总和精确等于 T）与 chunk_indices。
            B_use = 1
            num_seqs = max(1, min(4, int(T) // 128))
            cu_seqlens = _generate_cu_seqlens(int(T), num_seqs)
            chunk_indices = _prepare_chunk_indices(cu_seqlens, chunk_size)
        t2 = time.perf_counter()
        inputs = _build_case_inputs(
            B_use, int(HK), int(HV), int(T), int(K), int(V), q.dtype)
        t3 = time.perf_counter()
        if self.device in ["pyaclnn", "npu"]:
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        t4 = time.perf_counter()
        input_data.kwargs["q"] = inputs["q"]
        input_data.kwargs["k"] = inputs["k"]
        input_data.kwargs["w"] = inputs["w"]
        input_data.kwargs["dO"] = inputs["dO"]
        input_data.kwargs["dv"] = inputs["dv"]
        input_data.kwargs["gOptional"] = inputs["g"]
        input_data.kwargs["gkOptional"] = None
        input_data.kwargs["h0Optional"] = None
        input_data.kwargs["dhtOptional"] = None
        input_data.kwargs["cuSeqlensOptional"] = cu_seqlens
        input_data.kwargs["chunkIndicesOptional"] = chunk_indices
        input_data.kwargs["chunkSize"] = chunk_size
        input_data.kwargs["use_exp2"] = use_exp2
        input_data.kwargs["is_fix"] = is_fix
        t5 = time.perf_counter()
        if self.device == "pyaclnn":

            null_void_ptr = ctypes.c_void_p(None)

            AclTensorPtr = ctypes.POINTER(AclTensor)
            AclIntArrayPtr = ctypes.POINTER(AclIntArray)

            null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)
            null_int_array_ptr = ctypes.cast(null_void_ptr, AclIntArrayPtr)

            input_data.kwargs["gkOptional"] = null_tensor_ptr
            input_data.kwargs["h0Optional"] = null_tensor_ptr
            input_data.kwargs["dhtOptional"] = null_tensor_ptr
            if is_fix:
                input_data.kwargs["cuSeqlensOptional"] = null_int_array_ptr
                input_data.kwargs["chunkIndicesOptional"] = null_int_array_ptr
            else:
                acl_int_array_cu_seqlens = nnopbase.create_x_list([Int64(v) for v in cu_seqlens])
                input_data.kwargs["cuSeqlensOptional"] = acl_int_array_cu_seqlens

                acl_int_array_chunk_indices = nnopbase.create_x_list([Int64(v) for v in chunk_indices])
                input_data.kwargs["chunkIndicesOptional"] = acl_int_array_chunk_indices

            input_data.kwargs["scale"] = ctypes.c_double(1.0 / math.sqrt(inputs["q"].shape[-1]))
            del input_data.kwargs["is_fix"]

        t_end = time.perf_counter()
        print(
            f"[{OP_NAME}] init_by_input_data 耗时(ms) | "
            f"解析属性={1e3 * (t1 - t0):.3f} | "
            f"cu_seqlens/indices={1e3 * (t2 - t1):.3f} | "
            f"build_case_inputs={1e3 * (t3 - t2):.3f} | "
            f"搬运to(device)={1e3 * (t4 - t3):.3f} | "
            f"kwargs回写={1e3 * (t5 - t4):.3f} | "
            f"pyaclnn_acl={1e3 * (t_end - t5):.3f} | "
            f"总计={1e3 * (t_end - t0):.3f}"
        )

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        q = input_data.kwargs["q"]
        k = input_data.kwargs["k"]
        w = input_data.kwargs["w"]
        dO = input_data.kwargs["dO"]
        dv = input_data.kwargs["dv"]
        g = input_data.kwargs["gOptional"]
        scale = _to_float(
            input_data.kwargs.get("scale"),
            default=1.0 / math.sqrt(q.shape[-1]))
        chunk_size = _to_int(input_data.kwargs.get("chunkSize"), default=64)
        use_exp2 = _to_bool(input_data.kwargs.get("use_exp2", False))
        cu_seqlens = _as_int_list(input_data.kwargs.get("cuSeqlensOptional"))
        chunk_indices = _as_int_list(input_data.kwargs.get("chunkIndicesOptional"))

        with torch.no_grad():
            if self.device in {"npu", "pyaclnn"}:
                from fla_npu.ops import ascendc

                dh, dh0, dv2 = ascendc.chunk_gated_delta_rule_bwd_dhu(
                    q, k, w, dO, dv, scale, chunk_size,
                    g=g, gK=None, h0=None, dht=None,
                    cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                    use_exp2=use_exp2, transpose_state_layout=False,
                )
            elif self.device == "cpu" or self.device == "gpu":
                dh, dv2 = chunk_gated_delta_rule_bwd_dhu_golden(
                    q, k, w, dO, dv, g, scale, chunk_size,
                    use_exp2=use_exp2, cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices)
            else:
                raise RuntimeError(
                    f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return dh, dv2


@register("executor_chunk_gated_delta_rule_bwd_dhu_aclnn")
class AclnnFunctionApi(AclnnBaseApi):

    def init_by_input_data(self, input_data: InputDataset):
        input_args, output_packages = super().init_by_input_data(input_data)
        if self.device == "pyaclnn":
            null_void_ptr = ctypes.c_void_p(None)
            AclTensorPtr = ctypes.POINTER(AclTensor)
            null_tensor_ptr = ctypes.cast(null_void_ptr, AclTensorPtr)

            input_args.insert(15, null_tensor_ptr)
        return input_args, output_packages
