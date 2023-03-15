"""Test code for dense tensorcore operator with auto pipelining"""
import numpy as np
import tvm
from tvm import topi
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple
from tvm import te
from tvm.contrib.pickle_memoize import memoize
import tvm.testing
from tvm.topi.cuda.tensor_intrin import (
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)
from tvm.topi.exp_cuda.tensor_intrin import intrin_asm_ldmatrix
from tvm.contrib.auto_pipeline import (
    InjectSharedMemSwizzle,
    InjectPipelinedBuffer
)

"""
Guyue: this operator creation and scheduling code can also
be found in python/tvm/topi/exp_cuda/dense_tensorcore.py
"""

def _dense_tensorcore(A, B, out_dtype=None):
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

    # shared memory buffer
    AS = te.compute(A.shape, lambda *i: A(*i), name="AS")
    BS = te.compute(B.shape, lambda *i: B(*i), name="BS")
    
    # register buffer
    #
    # we use a multi-dimensional tensor so that we can 
    # manipulate the address mapping for each subtile of 16 values
    # this is for swizzled-buffer optimization
    #
    ldsm_len = 4 if A.dtype in ['float32','float'] else 8
    AF = te.compute((m//16, 16, k//wmma_k, (wmma_k//ldsm_len), ldsm_len), 
        lambda i0, i1, i2, i3, i4: AS[(i1 + i0*16), (i4 + ldsm_len*(i3 + (wmma_k//ldsm_len)*i2))], name="AF")
    BF = te.compute((n//16, 2, 8, k//wmma_k, (wmma_k//ldsm_len), ldsm_len), 
        lambda i0, i1, i2, i3, i4, i5: BS[(i2 + 8*(i1 + 2*i0)), (i5 + ldsm_len*(i4 + (wmma_k//ldsm_len)*i3))], name="BF")

    # generate the output
    r0 = te.reduce_axis((0, k//wmma_k), name="r0")
    r1 = te.reduce_axis((0, (wmma_k//ldsm_len)), name="r1")
    r2 = te.reduce_axis((0, ldsm_len), name="r2")
    C = te.compute((m, n), 
        lambda i, j: te.sum(AF[i//wmma_m, i%wmma_m, r0, r1, r2].astype(out_dtype) 
                          * BF[j//wmma_n, j%wmma_n//8, j%8, r0, r1, r2].astype(out_dtype), 
                          axis=[r0, r1, r2]), 
        name="C",
        tag="dense_tensorcore_ldmatrix_cuda")

    return C

def _schedule_dense_tensorcore(outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    C = outs[0]
    AF, BF = s[C].op.input_tensors
    (AS,) = s[AF].op.input_tensors
    (BS,) = s[BF].op.input_tensors
    (A,)  = s[AS].op.input_tensors
    (B,)  = s[BS].op.input_tensors
    m, n = get_const_tuple(C.shape)
    dtype = A.dtype
    out_dtype = C.dtype
    _, k = get_const_tuple(A.shape)

    # set buffer scope
    smem_scope = "shared.dyn"

    s[AS].set_scope(smem_scope)
    s[BS].set_scope(smem_scope)
    s[AF].set_scope("wmma.matrix_a")
    s[BF].set_scope("wmma.matrix_b")
    CF = s.cache_write(C, "wmma.accumulator")
    CS = s.cache_read(CF, smem_scope, [C])
    
    # scheduling parameters
    wmma_shape = (16, 16, 16)
    wmma_m, wmma_n, wmma_k = wmma_shape
    warp_size = 32
    block_row_warps = 4
    block_col_warps = 2
    warp_row_tiles = 4
    warp_col_tiles = 4
    chunk = 2
    stage_smem = 3
    stage_reg = 2
    input_swizzle = True
    vec = 8
    out_vec = 4 if out_dtype in ['float','float32'] else 8

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
    def shared_shedule(stage, vec_sz):
        s[stage].compute_at(s[CF], ko)
        xo, yo = stage.op.axis
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

    return s

def evaluate(batch, in_dim, out_dim, dtype):
    use_bias = False
    """Dense tensorcore verify function"""
    A = te.placeholder((batch, in_dim), dtype=dtype)
    B = te.placeholder((out_dim, in_dim), dtype=dtype)

    assert dtype in ["float16"]

    out_dtype = "float32"

    def get_ref_data():
        a_np = np.random.uniform(size=(batch, in_dim)).astype(dtype)
        b_np = np.random.uniform(size=(out_dim, in_dim)).astype(dtype)
        d_np = tvm.topi.testing.dense(a_np, b_np, None, use_bias, False, out_dtype)
        return (a_np, b_np, d_np)

    # get the test data
    a_np, b_np, d_np = get_ref_data()

    def _evaluate(device):
        dev = tvm.device(device, 0)
        print("Running on target: %s" % device)

        with tvm.target.Target(device):
            D = _dense_tensorcore(A, B, out_dtype)
            s = _schedule_dense_tensorcore([D])
        a_tvm = tvm.nd.array(a_np, dev)
        b_tvm = tvm.nd.array(b_np, dev)
        c_tvm = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=out_dtype), dev)

        # calling pipelining passes
        config={"tir.debug_keep_trivial_loop": True,
                "tir.add_lower_pass": [(3, tvm.tir.transform.Apply(InjectPipelinedBuffer())),
                                        (3, tvm.tir.transform.Apply(InjectSharedMemSwizzle())),
                                        (3, tvm.tir.transform.Simplify())]}
        with tvm.transform.PassContext(config=config):
            mod = tvm.lower(s, [A, B, D])
        func = tvm.build(mod, target=device)

        func(a_tvm, b_tvm, c_tvm)
        tvm.testing.assert_allclose(c_tvm.numpy(), d_np, rtol=1e-1)
        print("Result correct.")
        
        evaluator = func.time_evaluator(func.entry_name, dev, number=400)
        dur = evaluator(a_tvm, b_tvm, c_tvm).mean
        print("Time cost of %f throughput %f Tflops" \
               % (dur, batch*out_dim*in_dim*2/dur/1e12) )

    _evaluate("cuda")


def evaluate_dense_tensorcore_exp():
    """Test cases"""
    for dtype in ["float16"]:
        
        print("Test with pipelining")
        evaluate(4096, 4096, 4096, dtype)


if __name__ == "__main__":
    evaluate_dense_tensorcore_exp()
