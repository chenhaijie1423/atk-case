import torch
import torch_npu
import os
from typing import Tuple, Optional, List
import time

# ---------------------------------------------------------------------------
# 耗时统计工具（仅诊断用，不影响计算结果）
# 通过 recompute_w_u_fwd_cpu(..., verbose_timing=True) 或环境变量
# FLA_NPU_PROFILE=1 启用，会按 stage 汇总耗时并打印明细。
# ---------------------------------------------------------------------------
class _StageTimer:
    def __init__(self):
        self.totals = {}
        self.counts = {}
        self._starts = {}

    def start(self, stage: str):
        self._starts[stage] = time.perf_counter()

    def stop(self, stage: str):
        t0 = self._starts.pop(stage, None)
        if t0 is None:
            return
        dt = time.perf_counter() - t0
        self.totals[stage] = self.totals.get(stage, 0.0) + dt
        self.counts[stage] = self.counts.get(stage, 0) + 1

    # 容器型 stage：内部还嵌套了子 stage，汇总时不应计入 sum_stages，
    # 否则会与子 stage 重复累加（表现为 sum_stages > total）。
    _CONTAINER_STAGES = {"main_loop", "batched_total"}

    def summary(self, total_time: float, chunks: int) -> str:
        if not self.totals:
            return ""
        rows = sorted(self.totals.items(), key=lambda x: -x[1])
        leaf_totals = {k: v for k, v in self.totals.items() if k not in self._CONTAINER_STAGES}
        sum_stages = sum(leaf_totals.values())
        overhead = max(total_time - sum_stages, 0.0)
        lines = []
        lines.append(f"[recompute_w_u_fwd_cpu] chunks={chunks} total={total_time*1000:.3f}ms "
                     f"sum_leaf_stages={sum_stages*1000:.3f}ms overhead={overhead*1000:.3f}ms "
                     f"({overhead/total_time*100:.1f}%)")
        lines.append(f"[recompute_w_u_fwd_cpu]   {'stage':<24}{'total_ms':>12}{'calls':>10}{'avg_us':>12}{'pct':>8}")
        for stage, tot in rows:
            cnt = self.counts[stage]
            avg_us = (tot / cnt) * 1e6 if cnt else 0.0
            pct = (tot / total_time * 100) if total_time > 0 else 0.0
            tag = " *" if stage in self._CONTAINER_STAGES else ""
            lines.append(f"[recompute_w_u_fwd_cpu]   {stage:<24}{tot*1000:>12.3f}{cnt:>10d}{avg_us:>12.2f}{pct:>7.1f}%{tag}")
        return "\n".join(lines)


class _NullTimer:
    def start(self, stage: str): pass
    def stop(self, stage: str): pass
    def summary(self, total_time: float, chunks: int) -> str: return ""


def recompute_w_u_fwd_cpu(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    g: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor],
    chunk_size: int = 64,
    benchmark: bool = False,
    verbose_timing: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    CPU Equivalent of recompute_w_u_fwd (head-major layout, same as NPU op).

    输入布局（与 aclnn 算子一致，无需转置）：
      k:    [B, HK, T, K]   (elem dtype: bf16/fp16)
      v:    [B, HV, T, V]   (elem dtype: bf16/fp16)
      beta: [B, HV, T]      (gtype: fp32/bf16/fp16)
      A:    [B, HV, T, C]   (elem dtype: bf16/fp16, C = chunk_size)
      g:    [B, HV, T]      (gtype，与 beta 相同)
    输出：
      w: [B, HV, T, K], u: [B, HV, T, V]  (elem dtype)

    数学语义（逐 chunk）：
      vb[t]      = v[t] * beta[t]                       (AIV: fp32 计算，CAST_RINT 回 elem)
      kbg_exp[t] = k[t] * beta[t] * exp(g[t])           (AIV: fp32 计算，CAST_RINT 回 elem)
      u_chunk    = A_chunk[:, :cur] @ vb_chunk           (AIC: cube fp32 累加，输出 elem)
      w_chunk    = A_chunk[:, :cur] @ kbg_exp_chunk      (AIC: cube fp32 累加，输出 elem)
    其中 A_chunk = A[b, h, s:e, :cur] 仅取前 cur 列（ragged tail 列截断）。

    优化说明：将原按 (B, HV, chunk) 展开的 Python 循环改为按块批量 (bmm) 计算。
    CPU 上 bmm 对每个 batch 调用与单次 mm 相同的 gemm，因此每个块的矩阵乘
    与原循环逐位等价；所有 dtype 往返（.to(datatype).to(calc_type)）和运算顺序
    均原样保留，维持与 kernel 一致的精度模拟。

    verbose_timing=True 或环境变量 FLA_NPU_PROFILE=1 时，按 stage 打印耗时明细，
    便于定位瓶颈（仅诊断用，关闭时零开销）。
    """
    if os.environ.get("FLA_NPU_PROFILE", "") == "1":
        verbose_timing = True
    timer = _StageTimer() if verbose_timing else _NullTimer()
    t_total_start = time.perf_counter()
    chunk_count = 0

    # 大规模 shape（float64 benchmark 下尤其严重）时，一次性 batched 处理所有
    # chunk 会生成超大中间张量导致内存耗尽/卡死。限制每次 batched 调用的最大
    # chunk 数，超出时自动拆分为多次子批量调用（chunk 间无数据依赖，结果等价）。
    _MAX_DENSE_BATCH_CHUNKS = int(os.environ.get("FLA_NPU_CPU_MAX_BATCH_CHUNKS", "32"))

    calc_type = torch.float64 if benchmark else torch.float32
    B, HK, T, K = k.shape
    _, HV, _, V = v.shape
    if HK <= 0 or HV <= 0 or HV % HK != 0:
        raise ValueError(f"GVA requires HV divisible by HK, got HV={HV}, HK={HK}")
    n_ratio = HV // HK  # HV = n_ratio * HK
    C = chunk_size
    datatype = k.dtype
    gtype = g.dtype
    if benchmark:
        datatype = torch.float64
        gtype = torch.float64

    # ---- Pre-cast: beta/g -> calc_type, k -> expand to HV ----
    timer.start("gate_exp")
    beta_ct = beta.to(calc_type)                       # [B, HV, T]
    g_ct = g.to(calc_type)                             # [B, HV, T]
    gate = beta_ct * torch.exp(g_ct)                   # [B, HV, T]  (beta * exp(g))
    timer.stop("gate_exp")

    # k: [B, HK, T, K] -> [B, HV, T, K]（head h 对应 hk = h // n_ratio）
    timer.start("k_expand")
    if n_ratio == 1:
        k_hv = k
    else:
        k_hv = k.repeat_interleave(n_ratio, dim=1)
    timer.stop("k_expand")

    # ---- AIV stage 模拟：fp32 计算 + CAST_RINT 回 elem ----
    timer.start("vb_kbg")
    vb_elem = (v.to(calc_type) * beta_ct.unsqueeze(-1)).to(datatype)          # [B, HV, T, V]
    kbg_elem = (k_hv.to(calc_type) * gate.unsqueeze(-1)).to(datatype)         # [B, HV, T, K]
    timer.stop("vb_kbg")

    w = torch.empty((B, HV, T, K), dtype=datatype)
    u = torch.empty((B, HV, T, V), dtype=datatype)

    # 模拟 kernel 中间结果的 dtype 往返；datatype == calc_type 时短路（避免无谓分配）
    def cast_round(t):
        if datatype == calc_type:
            return t
        return t.to(datatype).to(calc_type)

    def pad_block(x: torch.Tensor, cur: int) -> torch.Tensor:
        """把 [HV, cur, D] 的块 pad 成 [HV, C, D]。
           cur == C 时返回原切片（无拷贝）。"""
        if cur == C:
            return x
        return torch.nn.functional.pad(x, (0, 0, 0, C - cur))

    def pad_A_block(x: torch.Tensor, cur: int) -> torch.Tensor:
        """把 [HV, cur, cur] 的 A 块（已截断到前 cur 列）pad 成 [HV, C, C]。
           列 pad 补零正是 ragged tail 的列截断语义；
           行 pad 补零不影响结果（scatter 时只写回前 cur 行）。"""
        if cur == C:
            return x
        return torch.nn.functional.pad(x, (0, C - cur, 0, C - cur))

    def process_blocks(blocks: List[Tuple[int, int, int]]):
        """通用块批量路径：blocks 为 (b_idx, s, e) 列表，逐块 gather（列截断+pad）
        后堆叠成一次大 bmm，完成按块写回。chunk 间无数据依赖，结果与逐块循环等价。"""
        nonlocal chunk_count
        if not blocks:
            return
        chunk_count += len(blocks)
        timer.start("batched_total")
        N = len(blocks)
        timer.start("batched_data_prep")
        A_list = [pad_A_block(A[b, :, s:e, :e - s], e - s) for b, s, e in blocks]
        vb_list = [pad_block(vb_elem[b, :, s:e], e - s) for b, s, e in blocks]
        kbg_list = [pad_block(kbg_elem[b, :, s:e], e - s) for b, s, e in blocks]
        A_b = torch.stack(A_list, dim=0)                # [N, HV, C, C]
        vb_b = torch.stack(vb_list, dim=0)              # [N, HV, C, V]
        kbg_b = torch.stack(kbg_list, dim=0)            # [N, HV, C, K]
        A_calc = A_b.to(calc_type).reshape(N * HV, C, C)
        vb_calc = vb_b.to(calc_type).reshape(N * HV, C, V)
        kbg_calc = kbg_b.to(calc_type).reshape(N * HV, C, K)
        timer.stop("batched_data_prep")

        # u = A @ vb ; w = A @ kbg_exp（无依赖，顺序 bmm）
        timer.start("batched_bmm")
        u_out = cast_round(torch.bmm(A_calc, vb_calc))   # [N*HV, C, V]
        w_out = cast_round(torch.bmm(A_calc, kbg_calc))  # [N*HV, C, K]
        timer.stop("batched_bmm")

        timer.start("batched_write_back")
        u_out = u_out.reshape(N, HV, C, V)
        w_out = w_out.reshape(N, HV, C, K)
        for i, (b, s, e) in enumerate(blocks):
            cur = e - s
            if datatype == calc_type:
                u[b, :, s:e] = u_out[i, :, :cur]
                w[b, :, s:e] = w_out[i, :, :cur]
            else:
                u[b, :, s:e] = u_out[i, :, :cur].to(datatype)
                w[b, :, s:e] = w_out[i, :, :cur].to(datatype)
        timer.stop("batched_write_back")
        timer.stop("batched_total")

    def process_dense_full_batched(b_idx: int, n_full: int, t_offset: int):
        """dense 满块快路径：token 区间连续 [t_offset, t_offset + n_full*C)，
           无需逐块 pad，直接 reshape 后一次大 bmm。"""
        nonlocal chunk_count
        if n_full <= 0:
            return
        chunk_count += n_full
        timer.start("batched_total")
        N = n_full
        s0 = t_offset
        timer.start("batched_data_prep")
        # A: [HV, N*C, C] -> [N, HV, C, C]（满块列完整，无需 pad）
        A_b = A[b_idx, :, s0:s0 + N * C, :].reshape(HV, N, C, C).permute(1, 0, 2, 3)
        A_calc = A_b.to(calc_type).reshape(N * HV, C, C)
        # vb/kbg: [HV, N*C, V/K] -> [N, HV, C, V/K]
        vb_b = vb_elem[b_idx, :, s0:s0 + N * C].reshape(HV, N, C, V).permute(1, 0, 2, 3)
        kbg_b = kbg_elem[b_idx, :, s0:s0 + N * C].reshape(HV, N, C, K).permute(1, 0, 2, 3)
        vb_calc = vb_b.to(calc_type).reshape(N * HV, C, V)
        kbg_calc = kbg_b.to(calc_type).reshape(N * HV, C, K)
        timer.stop("batched_data_prep")

        timer.start("batched_bmm")
        u_out = cast_round(torch.bmm(A_calc, vb_calc))   # [N*HV, C, V]
        w_out = cast_round(torch.bmm(A_calc, kbg_calc))  # [N*HV, C, K]
        timer.stop("batched_bmm")

        timer.start("batched_write_back")
        # [N*HV, C, V] -> [N, HV, C, V] -> [HV, N*C, V]
        if datatype == calc_type:
            u[b_idx, :, s0:s0 + N * C] = u_out.reshape(N, HV, C, V).permute(1, 0, 2, 3).reshape(HV, N * C, V)
            w[b_idx, :, s0:s0 + N * C] = w_out.reshape(N, HV, C, K).permute(1, 0, 2, 3).reshape(HV, N * C, K)
        else:
            u[b_idx, :, s0:s0 + N * C] = u_out.reshape(N, HV, C, V).permute(1, 0, 2, 3).reshape(HV, N * C, V).to(datatype)
            w[b_idx, :, s0:s0 + N * C] = w_out.reshape(N, HV, C, K).permute(1, 0, 2, 3).reshape(HV, N * C, K).to(datatype)
        timer.stop("batched_write_back")
        timer.stop("batched_total")

    def _run_blocks_subbatched(blocks: List[Tuple[int, int, int]]):
        """将 blocks 按 _MAX_DENSE_BATCH_CHUNKS 拆分为多次批量调用，
           避免大 T / fp64 下中间张量内存耗尽。"""
        if len(blocks) <= _MAX_DENSE_BATCH_CHUNKS:
            process_blocks(blocks)
            return
        if verbose_timing:
            n_sub = (len(blocks) + _MAX_DENSE_BATCH_CHUNKS - 1) // _MAX_DENSE_BATCH_CHUNKS
            print(f"[recompute_w_u_fwd_cpu] sub-batching: n_blocks={len(blocks)} > "
                  f"max={_MAX_DENSE_BATCH_CHUNKS}, splitting into {n_sub} sub-batches")
        for i in range(0, len(blocks), _MAX_DENSE_BATCH_CHUNKS):
            process_blocks(blocks[i:i + _MAX_DENSE_BATCH_CHUNKS])

    # Main Loop
    mode = "varlen" if cu_seqlens is not None else "dense"
    if verbose_timing:
        print(f"[recompute_w_u_fwd_cpu] start: B={B} T={T} HK={HK} HV={HV} K={K} V={V} "
              f"n_ratio={n_ratio} chunk_size={chunk_size} mode={mode} benchmark={benchmark}")
    # 小矩阵 bmm 场景：多线程 BLAS 的 spawn/join 开销远超实际计算。
    # 降为 1 线程消除开销，单线程 BLAS 对小矩阵 cache 更友好。
    _saved_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    timer.start("main_loop")
    if cu_seqlens is None:
        # Dense 模式：T 按 chunk_size 切分
        n_full = T // C
        ragged = T % C
        for b in range(B):
            # Phase 1: 满块走 reshape 快路径（按子批量拆分防内存膨胀）
            if n_full > 0:
                remaining = n_full
                s = 0
                while remaining > 0:
                    sub_n = min(remaining, _MAX_DENSE_BATCH_CHUNKS)
                    process_dense_full_batched(b, sub_n, s)
                    s += sub_n * C
                    remaining -= sub_n
            # Phase 2: ragged tail 走通用 pad 路径
            if ragged > 0:
                process_blocks([(b, T - ragged, T)])
    else:
        # Variable length: B = 1
        # 收集所有序列的 chunk 区间 (b=0, s, e)
        blocks = []
        for i in range(len(cu_seqlens) - 1):
            bos = int(cu_seqlens[i].item())
            eos = int(cu_seqlens[i + 1].item())
            pos = bos
            while pos < eos:
                e = min(pos + C, eos)
                blocks.append((0, pos, e))
                pos = e
        _run_blocks_subbatched(blocks)
    timer.stop("main_loop")

    t_total = time.perf_counter() - t_total_start
    summary = timer.summary(t_total, chunk_count)
    if summary:
        print(summary)

    torch.set_num_threads(_saved_num_threads)
    return w, u
