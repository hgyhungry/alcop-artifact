"""cuda batch_matmul operators"""
from torch import Tensor, tensor
import tvm
from tvm import autotvm
from tvm import te
from ..utils import traverse_inline, get_const_tuple
from ..cuda.tensor_intrin import (
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)
from .tensor_intrin import intrin_asm_ldmatrix


@autotvm.register_topi_compute("batch_matmul_tensorcore.exp_cuda")
def batch_matmul_tensorcore(
    cfg, x, y, out_shape=None, out_dtype=None, transpose_a=False, transpose_b=True
):
    """batch matmul tensorcore operator on cuda"""
    # TODO(jcf94): Deal with different transpose combinations
    assert not transpose_a and transpose_b
    # TODO(liuxin.ai): Deal with out_shape for broadcast
    del out_shape
    return batch_matmul_tensorcore_cuda(x, y, out_dtype)


@autotvm.register_topi_schedule("batch_matmul_tensorcore.exp_cuda")
def schedule_batch_matmul_tensorcore(cfg, outs, exp=True, double_buffer=False):
    """Schedule for batch_matmul operator using Tensorcore

    Parameters
    ----------
    outs: Array of Tensor
          The computation graph description of batch_matmul
          in the format of an array of tensors.

    Returns
    -------
    s: Schedule
        The computation schedule for the op.
    """
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])
    def _callback(op):
        if op.tag == "batch_matmul_tensorcore":
            _schedule_batch_matmul_tensorcore(cfg, s, op.output(0), exp, double_buffer)
    
    traverse_inline(s, outs[0].op, _callback)
    return s
    

def batch_matmul_tensorcore_cuda(x, y, out_dtype=None):
    """Computes batch matrix multiplication of `x` and `y` when `x` and `y` are
    data in batch.

    Parameters
    ----------
    x : tvm.te.Tensor
        3-D with shape [batch, M, K]

    y : tvm.te.Tensor
        3-D with shape [batch, N, K]

    Returns
    -------
    output : tvm.te.Tensor
        3-D with shape [batch, M, N]
    """
    assert len(x.shape) == 3 and len(y.shape) == 3, "only support 3-dim batch_matmul"
    x_shape = get_const_tuple(x.shape)
    y_shape = get_const_tuple(y.shape)
    assert x_shape[0] == y_shape[0], "batch dimension doesn't match"
    assert x_shape[2] == y_shape[2], "shapes of x and y is inconsistent"
    batch, M, K = x.shape
    N = y.shape[1]

    if out_dtype is None:
        out_dtype = x.dtype

    assert x.dtype == y.dtype
    assert x.dtype in ["float16"]
    wmma_m, wmma_n, wmma_k = 16, 16, 16
    assert (
        (M % wmma_m == 0 and N % wmma_n == 0 and K % wmma_k == 0)
    ), (
        "The shape of (batch, in_dim, out_dim) "
        f"must be multiple of ({wmma_m}, {wmma_k}, {wmma_n}) for now"
    )

    AS = te.compute(x.shape, lambda *i: x(*i), name="AS")
    BS = te.compute(y.shape, lambda *i: y(*i), name="BS")
    ldsm_len = 4 if x.dtype in ['float32', 'float'] else 8
    AF = te.compute(
        (batch, M//16, 16, K//wmma_k, (wmma_k//ldsm_len), ldsm_len), 
        lambda i, i0,i1,i2,i3,i4: AS[i, (i1 + i0*16), (i4 + ldsm_len*(i3 + (wmma_k//ldsm_len)*i2))],
        name="AF"
        )
    BF = te.compute(
        (batch, N//16, 2, 8, K//wmma_k, wmma_k//ldsm_len, ldsm_len),
        lambda i, i0, i1, i2, i3, i4, i5: BS[i, (i2 + 8*(i1 + 2*i0)), (i5 + ldsm_len*(i4 + (wmma_k//ldsm_len)*i3))],
        name="BF"
    )

    r0 = te.reduce_axis((0, K//wmma_k), name="r0")
    r1 = te.reduce_axis((0, (wmma_k//ldsm_len)), name="r1")
    r2 = te.reduce_axis((0, ldsm_len), name="r2")
    C = te.compute((batch, M, N), 
        lambda b, i, j: te.sum(AF[b, i//wmma_m, i%wmma_m, r0, r1, r2].astype(out_dtype) 
                          * BF[b, j//wmma_n, j%wmma_n//8, j%8, r0, r1, r2].astype(out_dtype), 
                          axis=[r0, r1, r2]), 
        name="C",
        tag="batch_matmul_tensorcore")
    return C

def _schedule_batch_matmul_tensorcore(cfg, s, C, exp=True, double_buffer=False):
    assert not (double_buffer and exp), "double_buffer and pipelined buffer cannot be used at the same time"
    """Schedule dense operator using Tensorcore and auto-pipelining for Ampere arch"""
    AF, BF = s[C].op.input_tensors
    (AS,) = s[AF].op.input_tensors
    (BS,) = s[BF].op.input_tensors
    (A,)  = s[AS].op.input_tensors
    (B,)  = s[BS].op.input_tensors
    batch, m, n = get_const_tuple(C.shape)
    dtype = A.dtype
    out_dtype = C.dtype
    _, _, k = get_const_tuple(A.shape)

    # Explicit memory access
    smem_scope = "shared.dyn" # or "shared"
    s[AS].set_scope(smem_scope)
    s[BS].set_scope(smem_scope)
    s[AF].set_scope("wmma.matrix_a")
    s[BF].set_scope("wmma.matrix_b")
    CF = s.cache_write(C, "wmma.accumulator")
    CS = s.cache_read(CF, smem_scope, [C])

    # create tuning space
    if exp:
        cfg.define_knob("smem_pipestage", [2,3,4,5,6])
        cfg.define_knob("regf_pipestage", [1,2,4])
    cfg.define_knob("block_row_warps", [1, 2, 4])
    cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4])
    cfg.define_knob("warp_col_tiles", [1, 2, 4])
    cfg.define_knob("chunk", [1, 2, 4, 8])
    cfg.define_knob("input_smem_swizzle", [True, ])
    cfg.define_knob("vec", [8])
    if out_dtype in ['float','float32']:
        cfg.define_knob("out_vec", [4])
    else:
        cfg.define_knob("out_vec", [8])

    wmma_shape = (16, 16, 16)
    wmma_m, wmma_n, wmma_k = wmma_shape
    warp_size = 32
    block_row_warps = cfg["block_row_warps"].val
    block_col_warps = cfg["block_col_warps"].val
    warp_row_tiles = cfg["warp_row_tiles"].val
    warp_col_tiles = cfg["warp_col_tiles"].val
    chunk = cfg["chunk"].val
    if exp:
        stage_smem = cfg["smem_pipestage"].val
        stage_reg = cfg["regf_pipestage"].val
    input_swizzle = cfg["input_smem_swizzle"].val
    vec = cfg["vec"].val
    out_vec = cfg["out_vec"].val

    # vec legalization
    vec_a = vec
    while block_row_warps * warp_row_tiles * chunk * (wmma_m*wmma_k) % \
        (block_row_warps * block_col_warps * warp_size * vec_a) != 0:
        vec_a = vec_a//2
    if vec_a==0:
        vec_a=1
    vec_b = vec
    while block_col_warps * warp_col_tiles * chunk * (wmma_n*wmma_k) % \
        (block_row_warps * block_col_warps * warp_size * vec_b) != 0:
        vec_b = vec_b//2
    if vec_b==0:
        vec_b=1

    block_x = te.thread_axis("blockIdx.x")
    block_y = te.thread_axis("blockIdx.y")
    block_z = te.thread_axis("blockIdx.z")
    thread_x = te.thread_axis("threadIdx.x")
    thread_y = te.thread_axis("threadIdx.y")
    thread_z = te.thread_axis("threadIdx.z")

    # Schedule for dense computation
    block_factor_b = wmma_m * warp_row_tiles * block_row_warps
    block_factor_o = wmma_n * warp_col_tiles * block_col_warps
    batch, b, o = C.op.axis
    block_i, bc = s[C].split(b, factor=block_factor_b)
    block_j, oc = s[C].split(o, factor=block_factor_o)
    s[C].reorder(batch, block_i, block_j, bc, oc)
    t = s[C].fuse(bc, oc)
    if out_vec >1:
        t, vi = s[C].split(t, factor=out_vec)
    t, tx = s[C].split(t, factor=warp_size)
    t, ty = s[C].split(t, factor=block_row_warps)
    t, tz = s[C].split(t, factor=block_col_warps)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(batch, block_z)
    s[C].bind(tz, thread_z)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    if out_vec >1:
        s[C].vectorize(vi)

    # Schedule for wmma store
    s[CS].compute_at(s[C], block_j)
    bs, bb, oo = CS.op.axis
    bb, bbi = s[CS].split(bb, factor=wmma_m)
    oo, ooi = s[CS].split(oo, factor=wmma_n)
    bb, bbii = s[CS].split(bb, factor=warp_row_tiles)
    oo, ooii = s[CS].split(oo, factor=warp_col_tiles)
    s[CS].reorder(bs, bb, oo, bbii, ooii, bbi, ooi)
    s[CS].bind(bb, thread_y)
    s[CS].bind(oo, thread_z)
    s[CS].tensorize(
        bbi,
        intrin_wmma_store_matrix(
            [wmma_n*warp_col_tiles*block_col_warps, 1], [wmma_n*warp_col_tiles, 1], 
            wmma_shape, out_dtype, (wmma_m, wmma_n), (wmma_m, wmma_n),
            C_scope=smem_scope
        ),
    )
    s[CF].compute_at(s[CS], oo)

    # Schedule for wmma computation
    bs, warp_i, warp_j = CF.op.axis
    warp_i, _ii = s[CF].split(warp_i, factor=wmma_m)
    warp_j, _jj = s[CF].split(warp_j, factor=wmma_n)
    r, ri2, ri8 = CF.op.reduce_axis
    ko, ki = s[CF].split(r, chunk)
    s[CF].reorder(bs, ko, ki, warp_i, warp_j, _ii, _jj, ri2, ri8)

    # Schedule for  wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    bs, i, wmma_i, r, ro, rva = AF.op.axis
    s[AF].reorder(bs, r, i, ro, wmma_i, rva)
    t =s[AF].fuse(ro, wmma_i)
    s[AF].bind(t, thread_x)

    # Schedule for  wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    bs, j, j2, j8, r, ro, rvb = BF.op.axis
    s[BF].reorder(bs, r, j, j2, ro, j8, rvb)
    t = s[BF].fuse(j2, ro, j8)
    s[BF].bind(t, thread_x)
    

    # Schedule for A's(B's) shared memory load
    def shared_shedule(stage, vec_sz):
        s[stage].compute_at(s[CF], ko)
        _, xo, yo = stage.op.axis
        t = s[stage].fuse(xo, yo)
        if vec_sz>1:
            t, vi = s[stage].split(t, factor=vec_sz)
        t, tx = s[stage].split(t, factor=warp_size)
        t, ty = s[stage].split(t, factor=block_row_warps)
        _, tz = s[stage].split(t, factor=block_col_warps)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        if vec_sz>1:
            s[stage].vectorize(vi)

    shared_shedule(AS, vec_a)
    shared_shedule(BS, vec_b)

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

    if exp:
    # add pipelining optimization
        if stage_smem > 1:
            s[AS].pipelined_buffer(stage_smem)
            s[BS].pipelined_buffer(stage_smem)
        if stage_reg > 1:
            s[AF].pipelined_buffer(stage_reg)
            s[BF].pipelined_buffer(stage_reg)

    # add swizzling optimization
    if input_swizzle:
        s[AS].swizzled_buffer()
        s[BS].swizzled_buffer()
        
    if double_buffer:
        s[AS].double_buffer()
        s[BS].double_buffer()
