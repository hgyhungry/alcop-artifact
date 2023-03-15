"""TODO"""
"""Tensorcore template for cuda backend"""
import numpy as np
import tvm
from tvm import te
from tvm import autotvm
from ..utils import get_const_tuple, traverse_inline, simplify
from ..nn.pad import pad
from ..nn.utils import get_pad_tuple3d
from ..cuda.tensor_intrin import (
    intrin_wmma_store_matrix,
    intrin_wmma_gemm,
)
from .tensor_intrin import intrin_asm_ldmatrix

def ndhwc_tensorcore_cuda(cfg, Input, Filter, stride, padding, dilation, out_dtype):
    """Compute declaration for conv3d tensorcore function"""
    assert isinstance(stride, int) or len(stride) == 3
    assert isinstance(dilation, int) or len(dilation) == 3

    if isinstance(stride, int):
        stride_d = stride_h = stride_w = stride
    else:
        stride_d, stride_h, stride_w = stride

    if isinstance(dilation, int):
        dilation_d = dilation_h = dilation_w = dilation
    else:
        dilation_d, dilation_h, dilation_w = dilation

    batch, in_depth, in_height, in_width, in_channel = get_const_tuple(Input.shape)
    kernel_d, kernel_h, kernel_w, num_filter, _, = get_const_tuple(Filter.shape)
    assert (
        (batch % 16 == 0 and in_channel % 16 == 0 and num_filter % 16 == 0)
    ), (
        "The shape of (batch, in_channel, num_filter) "
        "must be multiple of (16, 16, 16) for now"
    )

    # compute the output shape
    dilated_kernel_d = (kernel_d - 1) * dilation_d + 1
    dilated_kernel_h = (kernel_h - 1) * dilation_h + 1
    dilated_kernel_w = (kernel_w - 1) * dilation_w + 1
    pad_front, pad_top, pad_left, pad_back, pad_down, pad_right = get_pad_tuple3d(
        padding, (dilated_kernel_d, dilated_kernel_h, dilated_kernel_w)
    )
    out_channel = num_filter
    out_depth = simplify((in_depth - dilated_kernel_d + pad_front + pad_back) // stride_d + 1)
    out_height = simplify((in_height - dilated_kernel_h + pad_top + pad_down) // stride_h + 1)
    out_width = simplify((in_width - dilated_kernel_w + pad_left + pad_right) // stride_w + 1)
    pad_before = [0, pad_front, pad_top, pad_left, 0]
    pad_after = [0, pad_back, pad_down, pad_right, 0]
    PaddedInput = pad(Input, pad_before, pad_after, name="PaddedInput")
    
    # convert data type of input feature maps and weights
    # TODO: add checking here, datatype casting may cause precision loss
    TransPaddedInput = te.compute(
        PaddedInput.shape, lambda n, d, h, w, c: PaddedInput[n, d, h, w, c].astype("float16")
    )
    TransFilter = te.compute(
        Filter.shape, lambda d, h, w, i, o: Filter[d, h, w, i, o].astype("float16")
    )

    IS = te.compute(TransPaddedInput.shape, lambda *i: TransPaddedInput(*i), name="IS")
    FS = te.compute(TransFilter.shape, lambda *i: TransFilter(*i), name="FS")
    ldsm_len = 8

    _, padded_in_depth, padded_in_height, padded_in_width, _ = TransPaddedInput.shape
    IF = te.compute((batch//16, 16, padded_in_depth, padded_in_height, padded_in_width, in_channel//16, 16//ldsm_len, ldsm_len), 
        lambda no, ni, d, h, w, co, ci, cii: IS[ni+no*16, d, h, w, cii+ldsm_len*(ci+(16//ldsm_len)*co)], 
        name="IF")
    FF = te.compute((kernel_d, kernel_h, kernel_w, num_filter//16, 2, 8, in_channel//16, 16//ldsm_len, ldsm_len), 
        lambda d, h, w, fo, fi, fii, co, ci, cii: FS[d, h, w, fii+8*(fi + (16//8)*fo), cii+ldsm_len*(ci+(16//ldsm_len)*co)], 
        name="FF")
    
    r0 = te.reduce_axis((0, in_channel//16), name="r0")
    r1 = te.reduce_axis((0, (16//ldsm_len)), name="r1")
    r2 = te.reduce_axis((0, ldsm_len), name="r2")
    rz = te.reduce_axis((0, kernel_d), name="rz")
    ry = te.reduce_axis((0, kernel_h), name="ry")
    rx = te.reduce_axis((0, kernel_w), name="rx")

    Output = te.compute(
        (batch, out_depth, out_height, out_width, out_channel),
        lambda nn, zz, yy, xx, ff: te.sum(
            IF[
                nn//16, nn%16,
                zz * stride_d + rz * dilation_d,
                yy * stride_h + ry * dilation_h,
                xx * stride_w + rx * dilation_w,
                r0, r1, r2,
            ].astype(out_dtype)
            * FF[rz, ry, rx, ff//16, (ff%16)//8, ff%8, r0, r1, r2].astype(out_dtype),
            axis=[rz, ry, rx, r0, r1, r2],
        ),
        name="Conv3dOutput",
        tag="conv3d_ndhwc_tensorcore",
    )
    return Output


def schedule_ndhwc_tensorcore_cuda(cfg, s, Conv, exp=True, double_buffer=False):
    """Schedule tensorcore template"""
    AF, WF = s[Conv].op.input_tensors
    (AS,)  = s[AF].op.input_tensors
    (WS,)  = s[WF].op.input_tensors
    (trans_paddata,) = s[AS].op.input_tensors
    (kernel,) = s[WS].op.input_tensors
    paddata = s[trans_paddata].op.input_tensors

    in_dtype = trans_paddata.dtype
    out_dtype = Conv.dtype
    batch, _, _, _, _ = get_const_tuple(Conv.shape)
    _, _, _, out_channels, _,= get_const_tuple(kernel.shape)
    paddata = s[trans_paddata].op.input_tensors

    # inline the pad and dtype transform
    s[trans_paddata].compute_inline()
    s[kernel].compute_inline()
    s[paddata[0]].compute_inline()

    # Designate the memory hierarchy
    smem_scope = "shared.dyn" 
    s[AS].set_scope(smem_scope)
    s[WS].set_scope(smem_scope)
    s[AF].set_scope("wmma.matrix_a")
    s[WF].set_scope("wmma.matrix_b")
    CF = s.cache_write(Conv, "wmma.accumulator")
    CS = s.cache_read(CF, smem_scope, [Conv])
    output = Conv

    # Schedule for autotvm
    cfg.define_knob("block_row_warps", [1, 2, 4])
    cfg.define_knob("block_col_warps", [1, 2, 4])
    cfg.define_knob("warp_row_tiles", [1, 2, 4])
    cfg.define_knob("warp_col_tiles", [1, 2, 4])
    cfg.define_knob("chunk", [1, 2, 4,])
    cfg.define_knob("input_smem_swizzle", [True, ])
    cfg.define_knob("vec", [ 8])
    if out_dtype in ['float','float32']:
        cfg.define_knob("out_vec", [4])
    else:
        cfg.define_knob("out_vec", [8])
    if exp:
        cfg.define_knob("smem_pipestage", [2,3,4,5,6])
        cfg.define_knob("regf_pipestage", [1,2,4])

    # cfg.define_knob("block_row_warps", [2])
    # cfg.define_knob("block_col_warps", [2])
    # cfg.define_knob("warp_row_tiles", [1])
    # cfg.define_knob("warp_col_tiles", [2])
    # cfg.define_knob("chunk", [2,])
    # cfg.define_knob("input_smem_swizzle", [True, ])
    # cfg.define_knob("vec", [ 8])
    # if out_dtype in ['float','float32']:
    #     cfg.define_knob("out_vec", [4])
    # else:
    #     cfg.define_knob("out_vec", [8])
    # if exp:
    #     cfg.define_knob("smem_pipestage", [4])
    #     cfg.define_knob("regf_pipestage", [2])

    wmma_shape = (16, 16, 16)
    wmma_m, wmma_n, wmma_k = wmma_shape
    warp_size = 32
    block_row_warps = cfg["block_row_warps"].val
    block_col_warps = cfg["block_col_warps"].val
    warp_row_tiles = cfg["warp_row_tiles"].val
    warp_col_tiles = cfg["warp_col_tiles"].val
    chunk = cfg["chunk"].val
    input_swizzle = cfg["input_smem_swizzle"].val
    vec = cfg["vec"].val
    out_vec = cfg["out_vec"].val
    if exp:
        stage_smem = cfg["smem_pipestage"].val
        stage_reg = cfg["regf_pipestage"].val
    
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

    block_factor_n = wmma_m * warp_row_tiles * block_row_warps
    block_factor_o = wmma_n * warp_col_tiles * block_col_warps

    # Schedule for output
    nc, dc, hc, wc, oc = output.op.axis
    block_k = s[output].fuse(dc, hc, wc)
    s[output].bind(block_k, block_z)
    block_i, nc = s[output].split(nc, factor=block_factor_n)
    block_j, oc = s[output].split(oc, factor=block_factor_o)
    s[output].reorder(block_k, block_i, block_j, nc, oc)
    t = s[output].fuse(nc, oc)
    t, ti = s[output].split(t, factor=out_vec)
    t, tx = s[output].split(t, factor=warp_size)
    t, ty = s[output].split(t, factor=block_row_warps)
    t, tz = s[output].split(t, factor=block_col_warps)
    s[output].bind(block_i, block_x)
    s[output].bind(block_j, block_y)
    s[output].bind(tz, thread_z)
    s[output].bind(ty, thread_y)
    s[output].bind(tx, thread_x)
    s[output].vectorize(ti)

    # Schedule wmma store
    s[CS].compute_at(s[output], block_j)
    nc, dc, hc, wc, oc = CS.op.axis
    s[CS].reorder(dc, hc, wc, nc, oc)
    oc, ooc = s[CS].split(oc, factor=wmma_n)
    oc, oci = s[CS].split(oc, factor=warp_col_tiles)
    _, oc = s[CS].split(oc, factor=block_col_warps)
    nc, nnc = s[CS].split(nc, factor=wmma_m)
    nc, nci = s[CS].split(nc, factor=warp_row_tiles)
    _, nc = s[CS].split(nc, factor=block_row_warps)
    s[CS].reorder(nc, oc, nci, oci, nnc, ooc)
    s[CS].bind(nc, thread_y)
    s[CS].bind(oc, thread_z)

    # Schedule wmma computation
    s[CF].compute_at(s[CS], oc)
    n, d, h, w, o = CF.op.axis
    n, nnf = s[CF].split(n, factor=wmma_m)
    o, oof = s[CF].split(o, factor=wmma_n)
    kd, kh, kw, r, ri2, ri8 = CF.op.reduce_axis
    ko, ki = s[CF].split(r, factor=chunk)
    s[CF].reorder(kd, kh, kw, ko, ki, n, o, nnf, oof, ri2, ri8)

    s[AF].compute_at(s[CF], ki)
    s[WF].compute_at(s[CF], ki)

    # Schedule wmma load
    n, wmma_i, d, h, w, r, r1, rva = AF.op.axis
    s[AF].reorder(d, h, w, r, n, r1, wmma_i, rva)
    t = s[AF].fuse(r1, wmma_i)
    s[AF].bind(t, thread_x)

    d, h, w, ff, ff1, ff2, r, r1, rvw = WF.op.axis
    s[WF].reorder(d, h, w, r, ff, ff1, r1, ff2, rvw)
    t = s[WF].fuse(ff1, r1, ff2, )
    s[WF].bind(t, thread_x)

    s[AS].compute_at(s[CF], ko)
    s[WS].compute_at(s[CF], ko)

    # Schedule for data's share memory
    n, d, h, w, i = AS.op.axis
    s[AS].reorder(d, h, w, n, i)
    t = s[AS].fuse(n, i)
    if vec_a > 1:
        t, ti = s[AS].split(t, factor=vec_a)
    t, tx = s[AS].split(t, factor=warp_size)
    t, ty = s[AS].split(t, factor=block_row_warps)
    _, tz = s[AS].split(t, factor=block_col_warps)
    s[AS].bind(ty, thread_y)
    s[AS].bind(tz, thread_z)
    s[AS].bind(tx, thread_x)
    if vec_a > 1:
        s[AS].vectorize(ti)

    # Schedule for kernel's share memory
    kd, kh, kw, ic, o = WS.op.axis
    t = s[WS].fuse(ic, o)
    if vec_b > 1:
        t, ti = s[WS].split(t, factor=vec_b)
    t, tx = s[WS].split(t, factor=warp_size)
    t, ty = s[WS].split(t, factor=block_row_warps)
    _, tz = s[WS].split(t, factor=block_col_warps)
    s[WS].bind(ty, thread_y)
    s[WS].bind(tz, thread_z)
    s[WS].bind(tx, thread_x) 
    if vec_b > 1:
        s[WS].vectorize(ti)   

    s[AF].tensorize(rva, intrin_asm_ldmatrix(
        strides_dst=[wmma_k, 1], 
        strides_from=[chunk*wmma_k, 1],
        shape=wmma_shape,
        fragment_name="matrix_a",
        dtype=in_dtype,
        from_scope=smem_scope,
        dst_scope="wmma.matrix_a"
    ))
    s[WF].tensorize(rvw, intrin_asm_ldmatrix(
        strides_dst=[wmma_k, 1], 
        strides_from=[chunk*wmma_k, 1], 
        shape=wmma_shape,
        fragment_name="matrix_b", dtype=in_dtype,
        from_scope=smem_scope,
        dst_scope="wmma.matrix_b",
    ))

    def wmma_schedule(stage, axis):
        ldsm_len = 4 if in_dtype in ['float32','float'] else 8
        A_ = te.placeholder((16, 1, 1, 1, 1, (wmma_k//ldsm_len), ldsm_len), dtype=in_dtype)
        B_ = te.placeholder((2, 8, 1, (wmma_k//ldsm_len), ldsm_len), dtype=in_dtype)
        r0_ = te.reduce_axis((0, 1))
        r1_ = te.reduce_axis((0, (wmma_k//ldsm_len)))
        r2_ = te.reduce_axis((0, ldsm_len))
        C_ = te.compute((16, 1, 1, 1, 16), 
            lambda i, d, h, w, j: te.sum(A_[i, d, h, w,  r0_, r1_, r2_].astype(out_dtype) 
                            * B_[j//8, j%8, r0_, r1_, r2_].astype(out_dtype), 
                            axis=[r0_, r1_, r2_]))
        s[stage].tensorize(axis, intrin_wmma_gemm(
            AL_gemm=A_, 
            WL_gemm=B_,
            CL_compute=C_,
            strides_A=[wmma_k]*5 +[ldsm_len, 1],
            strides_W=[wmma_k*8] + [wmma_k] *2 + [ldsm_len, 1],
            strides_Conv=[wmma_n*warp_col_tiles]*4 +[1],
            shape=wmma_shape,
        ))
    
    wmma_schedule(CF, nnf)
    
    s[CS].tensorize(
        nnc,
        intrin_wmma_store_matrix(
            [wmma_n*warp_col_tiles*block_col_warps]*4+[1],
            [wmma_n*warp_col_tiles]*4+ [1],
            wmma_shape, 
            out_dtype,
            (wmma_m, 1, 1, 1, wmma_n),
            (wmma_m, 1, 1, 1, wmma_n),
            C_scope=smem_scope
        ),
    )

    N, OD, OH, OW, CO = get_const_tuple(output.shape)
    KD, KH, KW, _, CI = get_const_tuple(kernel.shape)
    cfg.add_flop(2 * N * OD * OH * OW * CO * CI * KD * KH * KW)

    if exp:
        # add pipelining optimization
        if stage_smem > 1:
            s[AS].pipelined_buffer(stage_smem)
            s[WS].pipelined_buffer(stage_smem)
        if stage_reg > 1:
            s[AF].pipelined_buffer(stage_reg)
            s[WF].pipelined_buffer(stage_reg)
    
    if input_swizzle:
        s[AS].swizzled_buffer()
        s[WS].swizzled_buffer()

    if double_buffer:
        s[AS].double_buffer()
        s[WS].double_buffer()


@autotvm.register_topi_compute("conv3d_ndhwc_tensorcore.exp_cuda")
def conv3d_ndhwc_tensorcore(cfg, data, kernel, strides, padding, dilation, out_dtype):
    """Compute conv3d with tensorcore for NDHWC layout"""
    return ndhwc_tensorcore_cuda(cfg, data, kernel, strides, padding, dilation, out_dtype)


@autotvm.register_topi_schedule("conv3d_ndhwc_tensorcore.exp_cuda")
def schedule_conv3d_ndhwc_tensorcore(cfg, outs, exp=True, double_buffer=False):
    """TOPI schedule callback"""
    s = te.create_schedule([x.op for x in outs])

    def _callback(op):
        if "conv3d_ndhwc_tensorcore" in op.tag:
            schedule_ndhwc_tensorcore_cuda(cfg, s, op.output(0), exp, double_buffer)

    traverse_inline(s, outs[0].op, _callback)
    return s
