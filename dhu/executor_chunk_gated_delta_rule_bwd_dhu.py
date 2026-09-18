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


def _sync_device(device: torch.device):
    """GPU 计时前同步，避免异步 launch 导致耗时被低估。"""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _new_timer() -> Dict[str, float]:
    return {}


def _tick(timer: Dict[str, float], key: str, device: torch.device):
    """记录某个阶段开始/结束时刻，累加到 timer[key]。"""
    _sync_device(device)
    timer.setdefault(key, 0.0)
    timer[f"__last_{key}"] = time.perf_counter()


def _tock(timer: Dict[str, float], key: str, device: torch.device):
    _sync_device(device)
    timer[key] = timer.get(key, 0.0) + (time.perf_counter() - timer.pop(f"__last_{key}", 0.0))


def _tick_ns(timer: Dict[str, float], key: str):
    """循环内细粒度计时（不同步 GPU），累加 dispatch+launch 耗时。"""
    timer.setdefault(key, 0.0)
    timer[f"__last_{key}"] = time.perf_counter()


def _tock_ns(timer: Dict[str, float], key: str):
    timer[key] = timer.get(key, 0.0) + (time.perf_counter() - timer.pop(f"__last_{key}", 0.0))


def _print_timer(timer: Dict[str, float]):
    total = sum(v for k, v in timer.items() if not k.startswith("__"))
    if total <= 0:
        return
    print(f"[chunk_gated_delta_rule_bwd_dhu_golden] 耗时统计（总计 {total*1000:.2f} ms）：", flush=True)
    for key, val in sorted(timer.items(), key=lambda kv: kv[1], reverse=True):
        if key.startswith("__"):
            continue
        pct = (val / total * 100.0) if total > 0 else 0.0
        print(f"  - {key:<32s}: {val*1000:8.2f} ms ({pct:5.1f}%)", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor, AclIntArray, nnopbase, Int64


OP_NAME = "chunk_gated_delta_rule_bwd_dhu"

_LN2 = 0.69314718055994530942
_DEFAULT_SEED = 20260817

_DTYPE_NAMES = {
    torch.bfloat16: "bf16",
    torch.float16: "fp16",
    torch.float32: "fp32",
}

_DTYPE_MAP = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}
def _gate(shape, calc_dtype: torch.dtype, device: torch.device, seed: int):
    """生成沿 T 维单调递减的 GDN gate，避免 exp 溢出。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.rand(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32) * 0.01 + 0.001
    data = -torch.cumsum(data, dim=-1)
    return data.to(calc_dtype).to(device)
def _orig_dtype(name: str) -> torch.dtype:
    """把 case_spec 中的 dtype 名称转成 torch dtype。"""
    return _DTYPE_MAP.get(str(name).lower(), torch.bfloat16)

def _randn(shape, dtype_name: str, calc_dtype: torch.dtype, device: torch.device, seed: int, scale: float = 0.05):
    """生成确定性正态分布输入；先量化到原始 dtype，再转到计算 dtype。"""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    data = torch.randn(tuple(int(x) for x in shape), generator=gen, dtype=torch.float32) * float(scale)
    return data.to(_orig_dtype(dtype_name)).to(calc_dtype).to(device)

def _finite_tuple(outputs, *, golden: bool = False) -> Tuple[torch.Tensor, ...]:
    """过滤 None 输出，规范 golden dtype，并检查浮点输出是否有限。"""
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    visible = []
    for output in outputs:
        if output is None or not isinstance(output, torch.Tensor):
            continue
        check = output.detach()
        if check.is_floating_point() and not torch.isfinite(check.float()).all().item():
            raise RuntimeError("输出包含 NaN 或 Inf")
        if golden and output.dtype == torch.float64:
            output = output.to(torch.float32)
        elif golden and output.dtype == torch.complex128:
            output = output.to(torch.complex64)
        visible.append(output)
    return tuple(visible)

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
    dtype_name = _DTYPE_NAMES.get(dtype, "bf16")
    cpu = torch.device("cpu")
    return {
        "q": _randn((B, HK, T, K), dtype_name, dtype, cpu, seed + 1),
        "k": _randn((B, HK, T, K), dtype_name, dtype, cpu, seed + 2),
        "w": _randn((B, HV, T, K), dtype_name, dtype, cpu, seed + 3),
        "dO": _randn((B, HV, T, V), dtype_name, dtype, cpu, seed + 4),
        "dv": _randn((B, HV, T, V), dtype_name, dtype, cpu, seed + 5),
        "g": _gate((B, HV, T), torch.float32, cpu, seed + 6),
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
    _timer = _new_timer()
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

    _tick(_timer, "01_dtype_convert", device)
    qf = q.to(compute_dtype)
    kf = k.to(compute_dtype)
    wf = w.to(compute_dtype)
    dof = dO.to(compute_dtype)
    dvf = dv.to(compute_dtype)
    gf = g.to(compute_dtype)
    _tock(_timer, "01_dtype_convert", device)

    def gate_exp(x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x * _LN2) if use_exp2 else torch.exp(x)
    _tick(_timer, "02_buffer_init", device)
    # dh/dv2 输出 dtype 与 q 一致：低位宽轮（bf16/fp16）与 NPU 输出 dtype
    # 对齐（ATK 双标杆的 local/remote dtype 检查），升精度轮为 fp32。
    dh = torch.zeros(B, Hv, NT, K, V, device=device, dtype=dtype_)
    # 变长模式下未被 chunk 覆盖的 [seq_total, T) 位置保持 dv 原值。
    dv2 = dv.clone() if varlen else torch.zeros(B, Hv, T, V, device=device, dtype=dtype_)
    num_tokens = len(cu_seqlens) - 1 if varlen else 1
    b_dh_buffers = torch.zeros(B, Hv, num_tokens, K, V, device=device, dtype=compute_dtype)
    b_dh = torch.zeros(B, Hv, K, V, device=device, dtype=compute_dtype)
    _tock(_timer, "02_buffer_init", device)

    # ---- 循环外预计算：参考 recompute_w_u_fwd_cpu 的 batched bmm 模式 ----
    _tick(_timer, "03_gva_repeat_interleave", device)
    # GVA 头扩展：repeat_interleave 一次替代 NT 次 index_select
    if hv_per_hk > 1:
        kf = kf.repeat_interleave(hv_per_hk, dim=1)
        qf = qf.repeat_interleave(hv_per_hk, dim=1)
    # gf 转 float32 整序列一次完成
    g_seq = gf[:, :, :T].to(torch.float32)  # [B, Hv, T]
    _tock(_timer, "03_gva_repeat_interleave", device)

    # 定长模式下，将满 chunk 数据 reshape 为 [B, Hv, n_full, BT, D] 批量预计算 gate 和 term1；
    # 变长模式 chunk 不连续，走原 per-chunk 路径。
    _ragged = T % BT
    n_full = T // BT if not varlen else 0
    term1_full = gate_factor_full = exp_bg_last_full = None
    if n_full > 0:
        _tick(_timer, "04_precompute_full", device)
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
        term1_f = round_dt(round_dt(torch.bmm(q_flat, do_flat))) * scale_f               # [BHv*N, K, V]
        term1_full = term1_f.reshape(B, Hv, n_full, K, V)                                # [B, Hv, n_full, K, V]

        # 满 chunk 数据 reshape 为 [B, Hv, n_full, BT, D]，循环内 O(1) 索引
        kf_full = kf[:, :, :T_full].reshape(B, Hv, n_full, BT, K)
        wf_full = wf[:, :, :T_full].reshape(B, Hv, n_full, BT, K)
        dvf_full = dvf[:, :, :T_full].reshape(B, Hv, n_full, BT, V)
        _tock(_timer, "04_precompute_full", device)

    _tick(_timer, "05_loop_total", device)
    for i_t in range(NT - 1, -1, -1):
        if varlen:
            i_n = chunk_indices[i_t * 2]
            block_idx_in_token = chunk_indices[i_t * 2 + 1]
            bos = cu_seqlens[i_n]
            token_length = cu_seqlens[i_n + 1] - bos
            start_t = block_idx_in_token * BT
            end_t = min(start_t + BT, token_length)
            gs = bos + start_t
            ge = bos + end_t
            b_dh = b_dh_buffers[:, :, i_n, :, :]
        else:
            gs = i_t * BT
            ge = min(gs + BT, T)
            token_length = T
            block_idx_in_token = i_t
            bos = 0
        dh[:, :, i_t, :, :] = b_dh
        last_idx = min((block_idx_in_token + 1) * BT, token_length) - 1
        global_last_idx = bos + last_idx

        # 满 chunk 走预计算分支；ragged tail / varlen 走原 per-chunk 分支
        if not varlen and i_t < n_full:
            _tick_ns(_timer, "06_loop_slice")
            b_idx = i_t
            k_blk = kf_full[:, :, b_idx, :, :]          # [B, Hv, BT, K]
            w_blk = wf_full[:, :, b_idx, :, :]          # [B, Hv, BT, K]
            b_dv_existing = dvf_full[:, :, b_idx, :, :]  # [B, Hv, BT, V]
            gate_factor = gate_factor_full[:, :, b_idx, :, :]  # [B, Hv, BT, 1]
            exp_bg_last = exp_bg_last_full[:, :, b_idx]         # [B, Hv]
            term1 = term1_full[:, :, b_idx, :, :]                # [B, Hv, K, V]
            _tock_ns(_timer, "06_loop_slice")

            _tick_ns(_timer, "08_loop_round_dt")
            b_dh_rd = round_dt(b_dh)
            _tock_ns(_timer, "08_loop_round_dt")
            _tick_ns(_timer, "07_loop_matmul")
            b_dv = k_blk @ b_dh_rd
            _tock_ns(_timer, "07_loop_matmul")
            _tick_ns(_timer, "08_loop_round_dt")
            b_dv = round_dt(b_dv)
            _tock_ns(_timer, "08_loop_round_dt")
            _tick_ns(_timer, "09_loop_elem")
            b_dv = b_dv * gate_factor + b_dv_existing
            dv2[:, :, gs:ge, :] = b_dv.to(dtype_)
            b_dh_for_update = b_dh * exp_bg_last.unsqueeze(-1).unsqueeze(-1)
            _tock_ns(_timer, "09_loop_elem")
            _tick_ns(_timer, "08_loop_round_dt")
            b_dv_rd = round_dt(b_dv)
            _tock_ns(_timer, "08_loop_round_dt")
            _tick_ns(_timer, "07_loop_matmul")
            term2 = w_blk.transpose(-1, -2) @ b_dv_rd
            _tock_ns(_timer, "07_loop_matmul")
            _tick_ns(_timer, "08_loop_round_dt")
            term2 = round_dt(term2)
            _tock_ns(_timer, "08_loop_round_dt")
            _tick_ns(_timer, "09_loop_elem")
            b_dh = b_dh_for_update + term1 - term2
            _tock_ns(_timer, "09_loop_elem")
        else:
            # --- 变长 / ragged tail：per-chunk 路径（保留原始 gate_exp 计算顺序）---
            _tick_ns(_timer, "06_loop_slice")
            k_blk = kf[:, :, gs:ge, :]
            q_blk_ = qf[:, :, gs:ge, :]  # kf/qf 已在循环外 repeat_interleave 展开 GVA
            w_blk = wf[:, :, gs:ge, :]
            b_do = dof[:, :, gs:ge, :]
            b_dv_existing = dvf[:, :, gs:ge, :]

            bg_last = g_seq[:, :, global_last_idx]
            b_g = g_seq[:, :, gs:ge]
            _tock_ns(_timer, "06_loop_slice")

            _tick_ns(_timer, "10_loop_gate")
            gate_factor = gate_exp(bg_last.unsqueeze(-1) - b_g).unsqueeze(-1)       # [B, Hv, bt, 1]
            _tock_ns(_timer, "10_loop_gate")

            # ---- dvState ----
            _tick_ns(_timer, "08_loop_round_dt")
            b_dh_rd = round_dt(b_dh)
            _tock_ns(_timer, "08_loop_round_dt")
            _tick_ns(_timer, "07_loop_matmul")
            b_dv = k_blk @ b_dh_rd
            _tock_ns(_timer, "07_loop_matmul")
            _tick_ns(_timer, "08_loop_round_dt")
            b_dv = round_dt(b_dv)
            _tock_ns(_timer, "08_loop_round_dt")
            _tick_ns(_timer, "09_loop_elem")
            b_dv = b_dv * gate_factor + b_dv_existing
            dv2[:, :, gs:ge, :] = b_dv.to(dtype_)
            _tock_ns(_timer, "09_loop_elem")

            _tick_ns(_timer, "10_loop_gate")
            b_dh_for_update = b_dh * gate_exp(bg_last).unsqueeze(-1).unsqueeze(-1)
            b_q_gated = q_blk_.transpose(-1, -2) * gate_exp(b_g).unsqueeze(-2)
            _tock_ns(_timer, "10_loop_gate")
            _tick_ns(_timer, "08_loop_round_dt")
            b_q_gated_rd = round_dt(b_q_gated)
            _tock_ns(_timer, "08_loop_round_dt")
            _tick_ns(_timer, "07_loop_matmul")
            term1 = b_q_gated_rd @ b_do
            _tock_ns(_timer, "07_loop_matmul")
            _tick_ns(_timer, "08_loop_round_dt")
            term1 = round_dt(term1)
            _tock_ns(_timer, "08_loop_round_dt")
            _tick_ns(_timer, "09_loop_elem")
            term1 = term1 * scale_f
            _tock_ns(_timer, "09_loop_elem")
            _tick_ns(_timer, "08_loop_round_dt")
            b_dv_rd = round_dt(b_dv)
            _tock_ns(_timer, "08_loop_round_dt")
            _tick_ns(_timer, "07_loop_matmul")
            term2 = w_blk.transpose(-1, -2) @ b_dv_rd
            _tock_ns(_timer, "07_loop_matmul")
            _tick_ns(_timer, "08_loop_round_dt")
            term2 = round_dt(term2)
            _tock_ns(_timer, "08_loop_round_dt")
            _tick_ns(_timer, "09_loop_elem")
            b_dh = b_dh_for_update + term1 - term2
            if varlen:
                b_dh_buffers[:, :, i_n, :, :] = b_dh
            _tock_ns(_timer, "09_loop_elem")
            continue

    _tock(_timer, "05_loop_total", device)

    _print_timer(_timer)
    return dh, dv2


@register("executor_chunk_gated_delta_rule_bwd_dhu")
class FunctionApi(BaseApi):
    """ATK 执行入口。"""
    def __init__(self, task_result: TaskResult):
        super(FunctionApi, self).__init__(task_result)
        self.high_precision = self.device == "cpu"

    def init_by_input_data(self, input_data: InputDataset):
        q = input_data.kwargs["q"]
        dO = input_data.kwargs["dO"]
        device = q.device
        B, HK, T, K = q.shape
        _, HV, _, V = dO.shape
        input_data.kwargs["scale"] = scale = 1.0 / math.sqrt(K)
        chunk_size = _to_int(input_data.kwargs.get("chunkSize"), default=64)
        use_exp2 = _to_bool(input_data.kwargs.get("use_exp2", False))
        is_fix = _to_bool(input_data.kwargs.get("is_fix", True))
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
        inputs = _build_case_inputs(
            B_use, int(HK), int(HV), int(T), int(K), int(V), q.dtype)
        if self.device in ["pyaclnn", "npu"]:
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
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

                res = _finite_tuple(
                    ascendc.chunk_gated_delta_rule_bwd_dhu(
                        q, k, w, dO, dv, scale, chunk_size,
                        g=g, gK=None, h0=None, dht=None,
                        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
                        use_exp2=use_exp2, transpose_state_layout=False,
                    ),
                    golden=False,
                )
            elif self.device == "cpu" or self.device == "gpu":
                dh, dv2 = chunk_gated_delta_rule_bwd_dhu_golden(
                    q, k, w, dO, dv, g, scale, chunk_size,
                    use_exp2=use_exp2, cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices)
                res = _finite_tuple((dh, None, dv2), golden=True)
            else:
                raise RuntimeError(
                    f"{OP_NAME} 仅支持 NPU DUT 与 CPU 标杆节点，当前设备：{self.device!r}")
        return res


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
