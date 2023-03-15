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
    intrin_wmma_load_matrix_A,
    intrin_wmma_load_matrix_W,
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)

"""
Guyue: this operator creation and scheduling code can also
be found in python/tvm/topi/cuda/dense_tensorcore.py
"""

def _dense_tensorcore(data, weight, out_dtype=None):
    """Dense tensorcore operator on CUDA"""
    assert len(data.shape) == 2 and len(weight.shape) == 2, "only support 2-dim dense"
    if out_dtype is None:
        out_dtype = data.dtype
    batch, in_dim = get_const_tuple(data.shape)
    out_dim, _ = get_const_tuple(weight.shape)

    assert data.dtype == weight.dtype
    assert data.dtype in ["float16", "int8", "uint8", "int4", "uint4"]
    if data.dtype in ["float16", "int8", "uint8"]:
        assert (
            (batch % 8 == 0 and in_dim % 16 == 0 and out_dim % 32 == 0)
            or (batch % 16 == 0 and in_dim % 16 == 0 and out_dim % 16 == 0)
            or (batch % 32 == 0 and in_dim % 16 == 0 and out_dim % 8 == 0)
        ), (
            "The shape of (batch, in_dim, out_dim) "
            "must be multiple of (16, 16, 16) or (32, 16, 8) or (8, 16, 32) for now"
        )
    else:
        assert (
            batch % 8 == 0 and in_dim % 32 == 0 and out_dim % 8 == 0
        ), "The shape of (batch, in_dim, out_dim) must be multiple of (8, 32, 8)"

    k = te.reduce_axis((0, in_dim), name="k")
    matmul = te.compute(
        (batch, out_dim),
        lambda i, j: te.sum(data[i, k].astype(out_dtype) * weight[j, k].astype(out_dtype), axis=k),
        name="T_dense",
        tag="dense_tensorcore",
    )
    return matmul

def _schedule_dense_tensorcore(outs):
    outs = [outs] if isinstance(outs, te.tensor.Tensor) else outs
    s = te.create_schedule([x.op for x in outs])

    C = outs[0]
    """Schedule dense operator using Tensorcore"""
    A, B = s[C].op.input_tensors
    if len(B.op.input_tensors) == 1 and B.op.input_tensors[0] == A:
        s[B].compute_inline()
    batch, out_dim = get_const_tuple(C.shape)
    data_dtype = A.dtype
    out_dtype = C.dtype

    # Explicit memory access
    AS = s.cache_read(A, "shared", [C])
    BS = s.cache_read(B, "shared", [C])
    AF = s.cache_read(AS, "wmma.matrix_a", [C])
    BF = s.cache_read(BS, "wmma.matrix_b", [C])
    CF = s.cache_write(C, "wmma.accumulator")
    CS = s.cache_read(CF, "shared", [C])

    # Deal with op fusion, such as bias and relu
    if C.op not in s.outputs:
        s[C].compute_inline()
        C = s.outputs[0].output(0)

    # schedule parameters
    warp_size = 32
    block_row_warps = 2
    block_col_warps = 2
    warp_row_tiles = 2
    warp_col_tiles = 2
    chunk = 2
    offset = 8
    offsetCS = 8
    vec = 8
    wmma_k = 16
    wmma_m = 16
    wmma_n = 16

    # Define the stride of intrin functions
    AS_align = chunk * wmma_k + offset
    BS_align = chunk * wmma_k + offset
    CS_align = warp_col_tiles * block_col_warps * wmma_n + offsetCS
    AS_stride = [AS_align, 1]
    BS_stride = [BS_align, 1]
    AF_stride = [wmma_k, 1]
    BF_stride = [wmma_k, 1]
    CF_stride = [warp_col_tiles * wmma_n, 1]
    CS_stride = [CS_align, 1]

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
    t, vi = s[C].split(t, factor=vec)
    t, tx = s[C].split(t, factor=warp_size)
    t, ty = s[C].split(t, factor=block_row_warps)
    t, tz = s[C].split(t, factor=block_col_warps)
    s[C].bind(block_i, block_x)
    s[C].bind(block_j, block_y)
    s[C].bind(tz, thread_z)
    s[C].bind(ty, thread_y)
    s[C].bind(tx, thread_x)
    s[C].vectorize(vi)

    # Schedule for wmma store
    s[CS].compute_at(s[C], block_j)
    bb, oo = CS.op.axis
    s[CS].storage_align(bb, CS_align - 1, CS_align)
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
    (k,) = CF.op.reduce_axis
    k, _k = s[CF].split(k, factor=wmma_k)
    ko, ki = s[CF].split(k, factor=chunk)
    s[CF].reorder(ko, ki, warp_i, warp_j, _ii, _jj, _k)

    # Schedule for  wmma_matrix_a load
    s[AF].compute_at(s[CF], ki)
    b, i = AF.op.axis
    b, b_ii = s[AF].split(b, factor=wmma_m)
    i, i_jj = s[AF].split(i, factor=wmma_k)
    s[AF].reorder(b, i, b_ii, i_jj)

    # Schedule for  wmma_matrix_b load
    s[BF].compute_at(s[CF], ki)
    o, i = BF.op.axis
    o, o_ii = s[BF].split(o, factor=wmma_n)
    i, i_ii = s[BF].split(i, factor=wmma_k)
    s[BF].reorder(o, i, o_ii, i_ii)

    # Schedule for A's(B's) shared memory load
    def shared_shedule(stage, strides):
        s[stage].compute_at(s[CF], ko)
        xo, yo = stage.op.axis
        s[stage].storage_align(xo, strides - 1, strides)
        t = s[stage].fuse(xo, yo)
        t, vi = s[stage].split(t, factor=vec)
        t, tx = s[stage].split(t, factor=warp_size)
        t, ty = s[stage].split(t, factor=block_row_warps)
        _, tz = s[stage].split(t, factor=block_col_warps)
        s[stage].bind(ty, thread_y)
        s[stage].bind(tz, thread_z)
        s[stage].bind(tx, thread_x)
        s[stage].vectorize(vi)

    shared_shedule(AS, AS_align)
    shared_shedule(BS, BS_align)

    shape = (wmma_m, wmma_n, wmma_k)
    AL_gemm = te.placeholder((wmma_m, wmma_k), name="AL_gemm", dtype=data_dtype)
    BL_gemm = te.placeholder((wmma_n, wmma_k), name="BL_gemm", dtype=data_dtype)
    k_gemm = te.reduce_axis((0, wmma_k), name="k_gemm")
    CL_compute = te.compute(
        (wmma_m, wmma_n),
        lambda ii, jj: te.sum(
            AL_gemm[ii, k_gemm].astype(out_dtype) * BL_gemm[jj, k_gemm].astype(out_dtype),
            axis=k_gemm,
        ),
        name="CL_compute",
    )

    # lower the computation loops down to TensorCore hardware intrinsics
    # by mapping the dense tensorcore to tensor intrinsics
    s[AF].tensorize(
        b_ii,
        intrin_wmma_load_matrix_A(
            AF_stride, AS_stride, shape, "row_major", (wmma_m, wmma_k), (wmma_m, wmma_k), data_dtype
        ),
    )
    s[BF].tensorize(
        o_ii,
        intrin_wmma_load_matrix_W(
            BF_stride, BS_stride, shape, "col_major", (wmma_n, wmma_k), (wmma_n, wmma_k), data_dtype
        ),
    )
    s[CF].tensorize(
        _ii, intrin_wmma_gemm(AL_gemm, BL_gemm, CL_compute, AF_stride, BF_stride, CF_stride, shape)
    )
    s[CS].tensorize(
        bbi,
        intrin_wmma_store_matrix(
            CS_stride, CF_stride, shape, out_dtype, (wmma_m, wmma_n), (wmma_m, wmma_n)
        ),
    )

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
        with tvm.transform.PassContext():
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
