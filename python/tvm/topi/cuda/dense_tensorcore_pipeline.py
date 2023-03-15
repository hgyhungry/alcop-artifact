"""
Guyue: this schedule template is deprecated. 
The new version for pipelined GEMM is topi.exp_cuda.dense_tensorcore.py
"""

"""Compute and Schedule definition for dense tensorcore with cuda backend"""
from __future__ import absolute_import as _abs
import tvm
from tvm import te
import tvm.autotvm as autotvm
from tvm.te.tensor import Tensor
from .. import tag
from ..utils import traverse_inline, get_const_tuple
from .tensor_intrin import (
    intrin_asm_ldmatrix,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)

@autotvm.register_topi_compute("dense_tensorcore_pipeline.cuda")
def dense_tensorcore_pipeline(cfg, data, weight, out_dtype=None):
    """Dense tensorcore operator on CUDA with pipeline for Ampere arch"""
    return dense_tensorcore_ldmatrix_cuda(data, weight, out_dtype=out_dtype)

@autotvm.register_topi_schedule("dense_tensorcore_pipeline.cuda")
def schedule_dense_tensorcore_pipeline(cfg, outs):
    """Schedule dense operator using Tensorcore and auto-pipelining for Ampere arch"""
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if op.tag == "dense_tensorcore_pipeline":
            _schedule_dense_tensorcore_pipeline(cfg, s, op.output(0))
    
    traverse_inline(s, outs[0].op, _callback)
    return s

def dense_tensorcore_ldmatrix_cuda(A, B, out_dtype=None):
    """Dense tensorcore operator on CUDA using ldmatrix"""
    assert len(A.shape) == 2 and len(B.shape) == 2, "only support 2-dim dense"
    if out_dtype is None:
        out_dtype = A.dtype
    m, k = get_const_tuple(A.shape)
    n, _ = get_const_tuple(B.shape)

    assert A.dtype == B.dtype
    assert A.dtype in ["float16"]
    wmma_m, wmma_n, wmma_k = 16, 16, 16
    assert (
        (m % wmma_m == 0 and n % wmma_n == 0 and k % wmma_k == 0)
    ), (
        "The shape of (batch, in_dim, out_dim) "
        f"must be multiple of ({wmma_m}, {wmma_k}, {wmma_n}) for now"
    )

    # special pipeline because we want to use ldmatrix
    AS = te.compute(A.shape, lambda *i: A(*i), name="AS")
    BS = te.compute(B.shape, lambda *i: B(*i), name="BS")
    ldsm_len = 4 if A.dtype in ['float32','float'] else 8
    AF = te.compute((m//16, 16, k//wmma_k, (wmma_k//ldsm_len), ldsm_len), 
        lambda i0, i1, i2, i3, i4: AS[(i1 + i0*16), (i4 + ldsm_len*(i3 + (wmma_k//ldsm_len)*i2))], name="AF")
    BF = te.compute((n//16, 2, 8, k//wmma_k, (wmma_k//ldsm_len), ldsm_len), 
        lambda i0, i1, i2, i3, i4, i5: BS[(i2 + 8*(i1 + 2*i0)), (i5 + ldsm_len*(i4 + (wmma_k//ldsm_len)*i3))], name="BF")

    r0 = te.reduce_axis((0, k//wmma_k), name="r0")
    r1 = te.reduce_axis((0, (wmma_k//ldsm_len)), name="r1")
    r2 = te.reduce_axis((0, ldsm_len), name="r2")
    C = te.compute((m, n), 
        lambda i, j: te.sum(AF[i//wmma_m, i%wmma_m, r0, r1, r2].astype(out_dtype) 
                          * BF[j//wmma_n, j%wmma_n//8, j%8, r0, r1, r2].astype(out_dtype), 
                          axis=[r0, r1, r2]), 
        name="C",
        tag="dense_tensorcore_pipeline")
    return C

def _schedule_dense_tensorcore_pipeline(cfg, s, C):
    """Schedule dense operator using Tensorcore and auto-pipelining for Ampere arch"""
    AF, BF = s[C].op.input_tensors
    (AS,) = s[AF].op.input_tensors
    (BS,) = s[BF].op.input_tensors
    (A,)  = s[AS].op.input_tensors
    (B,)  = s[BS].op.input_tensors
    m, n = get_const_tuple(C.shape)
    dtype = A.dtype
    out_dtype = C.dtype
    _, k = get_const_tuple(A.shape)

    # Explicit memory access
    smem_scope = "shared.dyn" # or "shared"
    s[AS].set_scope(smem_scope)
    s[BS].set_scope(smem_scope)
    s[AF].set_scope("wmma.matrix_a")
    s[BF].set_scope("wmma.matrix_b")
    CF = s.cache_write(C, "wmma.accumulator")
    CS = s.cache_read(CF, smem_scope, [C])

    # create tuning space
    cfg.define_knob("block_row_warps", [1, 2, 4])
    cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4])
    cfg.define_knob("warp_col_tiles", [1, 2, 4])
    cfg.define_knob("chunk", [1, 2, 4, 8])
    cfg.define_knob("input_smem_swizzle", [True, False])
    cfg.define_knob("out_smem_swizzle", [True, False])
    cfg.define_knob("vec", [1, 2, 4, 8])
    if out_dtype in ['float','float32']:
        cfg.define_knob("out_vec", [1, 2, 4])
    else:
        cfg.define_knob("out_vec", [1, 2, 4, 8])
    cfg.define_knob("smem_pipestage", [1, 2, 3, 4, 5, 6, 7, 8])
    cfg.define_knob("regf_pipestage", [1, 2, 3, 4])

    wmma_shape = (16, 16, 16)
    wmma_m, wmma_n, wmma_k = wmma_shape
    warp_size = 32
    block_row_warps = cfg["block_row_warps"].val
    block_col_warps = cfg["block_col_warps"].val
    warp_row_tiles = cfg["warp_row_tiles"].val
    warp_col_tiles = cfg["warp_col_tiles"].val
    chunk = cfg["chunk"].val
    input_swizzle = cfg["input_smem_swizzle"].val
    # out_swizzle = cfg["out_smem_swizzle"].val
    vec = cfg["vec"].val
    out_vec = cfg["out_vec"].val
    stage_smem = cfg["smem_pipestage"].val
    stage_reg = cfg["regf_pipestage"].val

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    # Schedule for dense computation
    block_factor_b = wmma_m * warp_row_tiles * block_row_warps
    block_factor_o = wmma_n * warp_col_tiles * block_col_warps
    b, o = C.op.axis
    block_i, bc = s[C].split(b, factor=block_factor_b)
    block_j, oc = s[C].split(o, factor=block_factor_o)
    s[C].reorder(block_i, block_j, bc, oc)
    t = s[C].fuse(bc, oc)
    if out_vec >1:
        t, vi = s[C].split(t, factor=out_vec)
    t, tx = s[C].split(t, factor=warp_size)
    t, ty = s[C].split(t, factor=block_row_warps)
    t, tz = s[C].split(t, factor=block_col_warps)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(tz, thread_z)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    if out_vec >1:
        s[C].vectorize(vi)

    # Schedule for wmma store
    s[CS].compute_at(s[C], block_j)
    bb, oo = CS.op.axis
    bb, bbi = s[CS].split(bb, factor=wmma_m)
    oo, ooi = s[CS].split(oo, factor=wmma_n)
    bb, bbii = s[CS].split(bb, factor=warp_row_tiles)
    oo, ooii = s[CS].split(oo, factor=warp_col_tiles)
    s[CS].reorder(bb, oo, bbii, ooii, bbi, ooi)
    s[CS].bind(bb, thread_y)
    s[CS].bind(oo, thread_z)

    # Schedule for wmma computation
    s[CF].compute_at(s[CS], oo)
    warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
    r, ri2, ri8 = CF.op.reduce_axis
    ko, ki = s[CF].split(r, chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, ri2, ri8)

    # Schedule for  wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    i, wmma_i, r, ro, rva = AF.op.axis
    s[AF].reorder(r, i, ro, wmma_i, rva)
    t =s[AF].fuse(ro, wmma_i)
    s[AF].bind(t, thread_x)

    # Schedule for  wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    j, j2, j8, r, ro, rvb = BF.op.axis
    s[BF].reorder(r, j, j2, ro, j8, rvb)
    t = s[BF].fuse(j2, ro, j8)
    s[BF].bind(t, thread_x)
    

    # Schedule for A's(B's) shared memory load
    def shared_shedule(stage, ):
        s[stage].compute_at(s[CF], ko)
        xo, yo = stage.op.axis
        t = s[stage].fuse(xo, yo)
        if vec>1:
            t, vi = s[stage].split(t, factor=vec)
        t, tx = s[stage].split(t, factor=warp_size)
        t, ty = s[stage].split(t, factor=block_row_warps)
        _, tz = s[stage].split(t, factor=block_col_warps)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        if vec>1:
            s[stage].vectorize(vi)

    shared_shedule(AS, )
    shared_shedule(BS, )

    # lower the computation loops down to TensorCore hardware intrinsics
    # by mapping the dense tensorcore to tensor intrinsics
    s[AF].tensorize(rva, intrin_asm_ldmatrix(
        strides_dst=[wmma_k, 1], strides_from=[chunk*wmma_k, 1], shape=wmma_shape,
        fragment_name="matrix_a", dtype=dtype, from_scope=smem_scope, 
        dst_scope="wmma.matrix_a"
    ))
    s[BF].tensorize(rvb, intrin_asm_ldmatrix(
        strides_dst=[wmma_k, 1], strides_from=[chunk*wmma_k, 1], shape=wmma_shape,
        fragment_name="matrix_b", dtype=dtype, from_scope=smem_scope, 
        dst_scope="wmma.matrix_b"
    ))
    def wmma_schedule(stage, axis):
        ldsm_len = 4 if dtype in ['float32','float'] else 8
        A_ = te.placeholder((16, 1, (wmma_k//ldsm_len), ldsm_len), dtype=dtype)
        B_ = te.placeholder((2, 8, 1, (wmma_k//ldsm_len), ldsm_len), dtype=dtype)
        r0_ = te.reduce_axis((0, 1))
        r1_ = te.reduce_axis((0, (wmma_k//ldsm_len)))
        r2_ = te.reduce_axis((0, ldsm_len))
        C_ = te.compute((16, 16), 
            lambda i, j: te.sum(A_[i, r0_, r1_, r2_].astype(out_dtype) 
                            * B_[j//8, j%8, r0_, r1_, r2_].astype(out_dtype), 
                            axis=[r0_, r1_, r2_]))
        s[stage].tensorize(axis, intrin_wmma_gemm(
            AL_gemm=A_, 
            WL_gemm=B_,
            CL_compute=C_,
            strides_A=[wmma_k, wmma_k, ldsm_len, 1],
            strides_W=[wmma_k*8, wmma_k, wmma_k, ldsm_len, 1],
            strides_Conv=[wmma_n*warp_col_tiles, 1],
            shape=wmma_shape,
        ))
    
    wmma_schedule(CF, _ii)

    s[CS].tensorize(
        bbi,
        intrin_wmma_store_matrix(
            [wmma_n*warp_col_tiles*block_col_warps, 1], [wmma_n*warp_col_tiles, 1], 
            wmma_shape, out_dtype, (wmma_m, wmma_n), (wmma_m, wmma_n),
            C_scope=smem_scope
        ),
    )

    # add pipelining optimization
    s[AS].pipelined_buffer(stage_smem)
    s[BS].pipelined_buffer(stage_smem)
    s[AF].pipelined_buffer(stage_reg)
    s[BF].pipelined_buffer(stage_reg)

    # add swizzling optimization
    if input_swizzle:
        s[AS].swizzled_buffer()
        s[BS].swizzled_buffer()
