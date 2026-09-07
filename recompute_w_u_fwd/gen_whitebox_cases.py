# Copyright (c) Tianjin University, Ltd. 2025. All rights reserved.
"""Generate white-box ATK cases for recompute_w_u_fwd.

Covers every functional branch in op_host tiling and op_kernel (cube/vector,
A5 L1-resident/regbase, GetChunkOffset, dtype template instantiations,
GVA, varlen, partial tail, ring-slot reuse).
"""
import json
import os

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "whitebox_aclnn_recompute_w_u_fwd.json")

# Each case: (id, desc, qkv_type, beta_type, B, HK, HV, T, V, chunk_size,
#             is_fix, cu_seqlens_shape, cu_seqlens_range, chunk_indices_shape)
#   - is_fix=True  : cu_seqlens/chunk_indices are placeholders (ignored by executor)
#   - is_fix=False : cu_seqlens_range=[lo,hi] controls seq lengths (lo==hi for determinism)
#                     chunk_indices_shape is a placeholder (executor recomputes)
CASES = [
    # --- Group A: TilingKey / dtype / chunk_size baseline (noGVA, fixed) ---
    (0,  "TilingKey1_V128_cs64_bf16_betaFp32_singleChunk",
     "bf16", "fp32", 1, 4, 4, 64, 128, 64, True, [1], [1, 1], [1, 2]),
    (1,  "TilingKey2_V256_cs64_bf16_betaFp32",
     "bf16", "fp32", 1, 4, 4, 64, 256, 64, True, [1], [1, 1], [1, 2]),
    (2,  "cs128_V128_bf16_betaFp32",
     "bf16", "fp32", 1, 4, 4, 128, 128, 128, True, [1], [1, 1], [1, 2]),
    (3,  "fp16_kBetaFp32_V128_cs64",
     "fp16", "fp32", 1, 4, 4, 64, 128, 64, True, [1], [1, 1], [1, 2]),
    (4,  "bf16_betaBf16_castPath_V128_cs64",
     "bf16", "bf16", 1, 4, 4, 64, 128, 64, True, [1], [1, 1], [1, 2]),
    (5,  "fp16_betaFp16_castPath_V128_cs64",
     "fp16", "fp16", 1, 4, 4, 64, 128, 64, True, [1], [1, 1], [1, 2]),
    (6,  "V256_cs128_bf16_betaFp32",
     "bf16", "fp32", 1, 4, 4, 128, 256, 128, True, [1], [1, 1], [1, 2]),

    # --- Group B: GVA ratios (fixed) ---
    (7,  "GVA_r2_V128_cs64_bf16_betaFp32_multiChunk",
     "bf16", "fp32", 1, 4, 8, 128, 128, 64, True, [1], [1, 1], [1, 2]),
    (8,  "GVA_r3_V256_cs128_bf16_betaFp32",
     "bf16", "fp32", 1, 4, 12, 256, 256, 128, True, [1], [1, 1], [1, 2]),
    (9,  "GVA_r4_V128_cs64_bf16_betaBf16",
     "bf16", "bf16", 1, 4, 16, 128, 128, 64, True, [1], [1, 1], [1, 2]),
    (10, "GVA_r6_V256_cs64_fp16_betaFp32",
     "fp16", "fp32", 1, 2, 12, 128, 256, 64, True, [1], [1, 1], [1, 2]),

    # --- Group C: Partial / tail chunk (fixed) ---
    (11, "fix_partialTail_V128_cs64_T96",
     "bf16", "fp32", 1, 4, 4, 96, 128, 64, True, [1], [1, 1], [1, 2]),
    (12, "fix_partialTail_V256_cs128_T192",
     "bf16", "fp32", 1, 4, 4, 192, 256, 128, True, [1], [1, 1], [1, 2]),
    (13, "fix_GVA_partialTail_V128_cs64_T100",
     "bf16", "fp32", 1, 4, 8, 100, 128, 64, True, [1], [1, 1], [1, 2]),

    # --- Group D: Multi-batch (fixed) ---
    (14, "multiBatch_B4_noGVA_V128_cs64",
     "bf16", "fp32", 4, 4, 4, 128, 128, 64, True, [1], [1, 1], [1, 2]),
    (15, "multiBatch_B2_GVA_V256_cs128_fp16_betaFp16",
     "fp16", "fp16", 2, 4, 8, 256, 256, 128, True, [1], [1, 1], [1, 2]),

    # --- Group E: Variable-length ---
    (16, "varlen_1seq_fullChunk_V128_cs64",
     "bf16", "fp32", 1, 4, 4, 64, 128, 64, False, [1], [64, 64], [1, 2]),
    (17, "varlen_1seq_partialTail_V128_cs64",
     "bf16", "fp32", 1, 4, 4, 100, 128, 64, False, [1], [100, 100], [2, 2]),
    (18, "varlen_2seqs_V256_cs128",
     "bf16", "fp32", 1, 4, 4, 256, 256, 128, False, [2], [128, 128], [2, 2]),
    (19, "varlen_3seqs_fp16_betaFp32_V128_cs64",
     "fp16", "fp32", 1, 4, 4, 192, 128, 64, False, [3], [64, 64], [3, 2]),
    (20, "varlen_seqLessThanChunk_V128_cs64",
     "bf16", "fp32", 1, 4, 4, 30, 128, 64, False, [1], [30, 30], [1, 2]),
    (21, "varlen_GVA_V256_cs64",
     "bf16", "fp32", 1, 4, 8, 128, 256, 64, False, [2], [64, 64], [2, 2]),

    # --- Group F: A5 ring-slot reuse (tasks > GM_RING_DEPTH=8) ---
    (22, "ringReuse_fix_noGVA_8chunks_x2H_V128_cs64",
     "bf16", "fp32", 1, 2, 2, 512, 128, 64, True, [1], [1, 1], [1, 2]),
    (23, "ringReuse_fix_GVA_10chunks_x8H_V256_cs64_fp16",
     "fp16", "fp32", 1, 4, 8, 640, 256, 64, True, [1], [1, 1], [1, 2]),
    (24, "ringReuse_varlen_5seqs_x2H_V128_cs64_betaBf16",
     "bf16", "bf16", 1, 2, 2, 320, 128, 64, False, [5], [64, 64], [5, 2]),
    (25, "ringReuse_fix_cs128_8chunks_x2H_V128",
     "bf16", "fp32", 1, 2, 2, 1024, 128, 128, True, [1], [1, 1], [1, 2]),
    (26, "ringReuse_multiBatch_B8_8chunks_x4H_V128_cs64",
     "bf16", "fp32", 8, 4, 4, 64, 128, 64, True, [1], [1, 1], [1, 2]),
    (27, "ringReuse_varlen_GVA_partialTail_V256_cs64_fp16",
     "fp16", "fp32", 1, 4, 8, 200, 256, 64, False, [2], [100, 100], [4, 2]),

    # --- Group G: additional varlen + cs128 combo ---
    (28, "varlen_cs128_partialTail_V128",
     "bf16", "fp32", 1, 4, 4, 200, 128, 128, False, [1], [200, 200], [2, 2]),
]


def _tensor_input(name, dtype, shape, rv=(-5, 5)):
    return {
        "name": name,
        "type": "tensor",
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": list(rv),
        "backward": True,
        "align_32B": None,
        "outlier_values": None,
    }


def _attr_input(name, dtype, range_values):
    return {
        "name": name,
        "type": "attr",
        "required": True,
        "dtype": dtype,
        "shape": None,
        "range_values": range_values,
        "backward": False,
        "align_32B": None,
        "outlier_values": None,
    }


def build_case(case):
    (cid, _desc, qkv, beta_type, B, HK, HV, T, V, cs,
     is_fix, cu_shape, cu_range, ci_shape) = case

    K = 128  # K is always 128 per op constraints

    inputs = [
        _tensor_input("k", qkv, [B, HK, T, K]),
        _tensor_input("v", qkv, [B, HV, T, V]),
        _tensor_input("beta", beta_type, [B, HV, T]),
        _tensor_input("A", qkv, [B, HV, T, cs]),
        _tensor_input("g", beta_type, [B, HV, T]),
        _tensor_input("cu_seqlens", "int64", cu_shape, cu_range),
        _tensor_input("chunk_indices", "int64", ci_shape, (1, 100)),
        _attr_input("chunk_size", "int", cs),
        _attr_input("beta_type", "string", beta_type),
        _attr_input("is_fix", "bool", is_fix),
        _attr_input("qkv_type", "string", qkv),
    ]

    return {
        "id": cid,
        "default_seed": None,
        "name": "aclnn.recompute.w.u.fwd",
        "aclnn_name": "RecomputeWUFwd",
        "triton_name": None,
        "kernel_name": None,
        "version": "v2.1",
        "expected_error_msg": None,
        "api": "pytorch",
        "api_type": "executor_recompute_w_u_fwd",
        "aclnn_api_type": "aclnn_function",
        "triton_api_type": "triton_function",
        "fusion_api_type": "fusion_function",
        "fusion_mode": None,
        "dist_api_type": "dist_function",
        "kernel_api_type": "kernel_function",
        "backward": False,
        "standard": {
            "acc": {
                "cv_fused_double_benchmark": {
                    "max_re_ratio": 5,
                    "avg_re_ratio": 1.5,
                    "root_mean_squared_ratio": 1.5,
                }
            },
            "perf": "not_key",
            "mem": 1.1,
        },
        "outputs": None,
        "inputs": inputs,
        "acl_json": "",
        "method_inputs": None,
        "tensor_input": None,
        "compute_times": None,
        "save_name": None,
        "uuid": None,
        "downloaded": False,
        "is_boundary": False,
        "xrun_cs_name": None,
        "xrun_data": None,
        "strategy": None,
    }


def main():
    cases = [build_case(c) for c in CASES]
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Generated {len(cases)} white-box cases -> {OUT_PATH}")

    # Print branch coverage summary
    print("\n=== Branch Coverage Summary ===")
    tk1 = sum(1 for c in CASES if c[8] == 128)
    tk2 = sum(1 for c in CASES if c[8] == 256)
    cs64 = sum(1 for c in CASES if c[9] == 64)
    cs128 = sum(1 for c in CASES if c[9] == 128)
    fix = sum(1 for c in CASES if c[10] is True)
    varlen = sum(1 for c in CASES if c[10] is False)
    gva = sum(1 for c in CASES if c[5] != c[6])
    nogva = sum(1 for c in CASES if c[5] == c[6])
    partial_fix = sum(1 for c in CASES if c[10] is True and c[7] % c[9] != 0)
    partial_var = sum(1 for c in CASES if c[10] is False and c[13][0] % c[9] != 0)
    bf16_bf16 = sum(1 for c in CASES if c[2] == "bf16" and c[3] == "bf16")
    bf16_fp32 = sum(1 for c in CASES if c[2] == "bf16" and c[3] == "fp32")
    fp16_fp32 = sum(1 for c in CASES if c[2] == "fp16" and c[3] == "fp32")
    fp16_fp16 = sum(1 for c in CASES if c[2] == "fp16" and c[3] == "fp16")
    ring = sum(1 for c in CASES if _task_count(c) > 8)

    print(f"  TilingKey=1 (V=128): {tk1} cases")
    print(f"  TilingKey=2 (V=256): {tk2} cases")
    print(f"  chunk_size=64: {cs64} cases")
    print(f"  chunk_size=128: {cs128} cases")
    print(f"  Fixed-length: {fix} cases")
    print(f"  Variable-length: {varlen} cases")
    print(f"  noGVA (HV==HK): {nogva} cases")
    print(f"  GVA (HV>HK): {gva} cases")
    print(f"  Partial tail (fixed): {partial_fix} cases")
    print(f"  Partial tail (varlen): {partial_var} cases")
    print(f"  dtype bf16+fp32: {bf16_fp32} cases")
    print(f"  dtype bf16+bf16: {bf16_bf16} cases")
    print(f"  dtype fp16+fp32: {fp16_fp32} cases")
    print(f"  dtype fp16+fp16: {fp16_fp16} cases")
    print(f"  Ring-slot reuse (tasks>8): {ring} cases")


def _task_count(case):
    (_cid, _desc, _qkv, _beta, B, _HK, HV, T, _V, cs,
     is_fix, cu_shape, cu_range, _ci_shape) = case
    if is_fix:
        chunk_num = B * ((T + cs - 1) // cs)
    else:
        seq_len = cu_range[0]
        chunk_num = cu_shape[0] * ((seq_len + cs - 1) // cs)
    return chunk_num * HV


if __name__ == "__main__":
    main()
