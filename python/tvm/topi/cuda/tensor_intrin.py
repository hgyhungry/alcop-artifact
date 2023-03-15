# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unnecessary-lambda, too-many-arguments
"""Tensor intrinsics on CUDA."""
import tvm
from tvm import te

# TODO(Guyue): I think all changes to this file can be deprecated.

def dp4a(x_scope="local", y_scope="local", z_scope="local"):
    """
    Int8 dot product reduced by every 4 elements using __dp4a

    Parameters
    ----------
    x_scope : str, optional
        The storage scope of buffer for lhs
    y_scope : str, optional
        The storage scope of buffer for rhs
    z_scope : str, optional
        The storage scope of buffer for result

    Returns
    -------
    intrin : TensorIntrin
        The dp4a TensorIntrin that can be used in tensorizing schedule.
    """

    n = 4  # dp4a requires operands packed by 4
    x = te.placeholder((n,), name="x", dtype="int8")
    y = te.placeholder((n,), name="y", dtype="int8")

    k = te.reduce_axis((0, n), name="rc")

    z = te.compute((1,), lambda i: te.sum(x[k].astype("int32") * y[k].astype("int32"), axis=[k]))

    def _intrin_func(ins, outs):
        def _instr(index):
            xx, yy = ins
            zz = outs[0]

            if index == 1:
                return zz.vstore(0, 0)

            ib = tvm.tir.ir_builder.create()

            vec_x = xx.vload(0, dtype="int8x4")
            vec_y = yy.vload(0, dtype="int8x4")
            prev_z = 0 if index == 0 else zz.vload(0)

            new_z = tvm.tir.call_pure_extern("int32", "__dp4a", vec_x, vec_y, prev_z)
            ib.emit(zz.vstore(0, new_z))

            return ib.get()

        return _instr(0), _instr(1), _instr(2)  # body, reset, update

    default_buffer_params = {"data_alignment": 4, "offset_factor": 1}
    scopes = {x: x_scope, y: y_scope, z: z_scope}
    binds = {
        t: tvm.tir.decl_buffer(
            t.shape, t.dtype, t.op.name, scope=scopes[t], **default_buffer_params
        )
        for t in [x, y, z]
    }

    return te.decl_tensor_intrin(
        z.op, _intrin_func, binds=binds, default_buffer_params=default_buffer_params
    )


def intrin_load_shared(strides_dst, strides_from, shape, in_dtype, layout="linear", use_async="", tgt_scope="shared", thread_extent=None):
    A = te.placeholder(shape, name="A", dtype=in_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope="global", data_alignment=32, offset_factor=shape[1], strides=strides_from
    )
    C = te.compute(shape, lambda i, j: A[i, j], name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope=tgt_scope, data_alignment=32, offset_factor=shape[1], strides=strides_dst
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        if thread_extent is not None:
            ib.scope_attr(thread_extent[0], "thread_extent", thread_extent[1])
        BA = ins[0]
        BC = outs[0]
        ib.emit(
            tvm.tir.call_intrin(
                in_dtype,
                "tir.tvm_load_shared",
                BA.access_ptr("r"),
                BC.access_ptr("w"),
                strides_from[0],
                strides_dst[0],
                layout,
                use_async
            )
        )
        return ib.get()
    
    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_load_matrix_A(strides_dst, strides_from, shape, layout, A_shape, C_shape, in_dtype, flags=[], A_scope="shared", thread_extent=None):
    """Intrin function for loading data from shared memory to wmma.matrix_a"""
    wmma_m, wmma_n, wmma_k = shape

    A = te.placeholder(A_shape, name="A", dtype=in_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope=A_scope, strides=strides_from, data_alignment=32, offset_factor=8
    )
    C = te.compute(C_shape, lambda *i: A(*i), name="C")
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope="wmma.matrix_a",
        strides=strides_dst,
        data_alignment=32,
        offset_factor=8,
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        if thread_extent is not None:
            ib.scope_attr(thread_extent[0], "thread_extent", thread_extent[1])
        BA = ins[0]
        BC = outs[0]
        row = wmma_m * wmma_k
        warp_index = BC.elem_offset // row + BC.elem_offset % row // wmma_k
        intrin_args = [in_dtype, "tir.tvm_load_matrix_sync",
            BC.data, wmma_m, wmma_n, wmma_k, warp_index,
            BA.access_ptr("r"), strides_from[0], layout] + flags
        ib.emit(
            tvm.tir.call_intrin(*intrin_args)
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_load_matrix_W(strides_dst, strides_from, shape, layout, A_shape, C_shape, in_dtype, flags=[], W_scope="shared", thread_extent=None):
    """Intrin function for loading data from shared memory to wmma.matrix_b"""
    wmma_m, wmma_n, wmma_k = shape

    A = te.placeholder(A_shape, name="A", dtype=in_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape, A.dtype, scope=W_scope, strides=strides_from, data_alignment=32, offset_factor=8
    )
    C = te.compute(C_shape, lambda *i: A(*i), name="C")
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        scope="wmma.matrix_b",
        strides=strides_dst,
        data_alignment=32,
        offset_factor=8,
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        if thread_extent is not None:
            ib.scope_attr(thread_extent[0], "thread_extent", thread_extent[1])
        BA = ins[0]
        BC = outs[0]
        row = wmma_n * wmma_k
        warp_index = BC.elem_offset // row + BC.elem_offset % row // wmma_n
        intrin_args = [in_dtype, "tir.tvm_load_matrix_sync", 
            BC.data, wmma_m, wmma_n, wmma_k, warp_index, 
            BA.access_ptr("r"), strides_from[0], layout] + flags
        ib.emit(
            tvm.tir.call_intrin(*intrin_args)
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_store_matrix(strides_dst, strides_from, shape, out_dtype, A_shape, C_shape, C_scope="shared"):
    """Intrin function for storing the results from wmma.accumulator to shared"""
    wmma_m, wmma_n, wmma_k = shape
    A = te.placeholder(A_shape, name="A", dtype=out_dtype)
    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        scope="wmma.accumulator",
        strides=strides_from,
        data_alignment=32,
        offset_factor=8,
    )
    C = te.compute(C_shape, lambda *i: A(*i), name="C")
    BC = tvm.tir.decl_buffer(
        C.shape, C.dtype, scope=C_scope, strides=strides_dst, data_alignment=32, offset_factor=8
    )

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()

        BA = ins[0]
        BC = outs[0]
        row = wmma_m * wmma_n
        warp_index = BA.elem_offset // row + BA.elem_offset % row // wmma_n
        ib.emit(
            tvm.tir.call_intrin(
                "handle",
                "tir.tvm_store_matrix_sync",
                BA.data,
                wmma_m,
                wmma_n,
                wmma_k,
                warp_index,
                BC.access_ptr("w"),
                strides_dst[0],
                "row_major",
            )
        )
        return ib.get()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, C: BC})


def intrin_wmma_gemm(AL_gemm, WL_gemm, CL_compute, strides_A, strides_W, strides_Conv, shape,
                     A_scope="wmma.matrix_a", W_scope="wmma.matrix_b"):
    """Intrin for wmma fill_fragment and mma_sync

    Parameters
    ----------
    AL_gemm : tvm.te.placeholder
        wmma matrix A
    WL_gemm : tvm.te.placeholder
        wmma matrix B
    CL_compute : tvm.te.compute
        The definition of wmma gemm
    """
    wmma_m, wmma_n, wmma_k = shape
    A = AL_gemm
    B = WL_gemm
    C = CL_compute

    BA = tvm.tir.decl_buffer(
        A.shape,
        A.dtype,
        name="BA",
        scope=A_scope,
        data_alignment=32,
        offset_factor=8,
        strides=strides_A,
    )
    BB = tvm.tir.decl_buffer(
        B.shape,
        B.dtype,
        name="BB",
        scope=W_scope,
        data_alignment=32,
        offset_factor=8,
        strides=strides_W,
    )
    BC = tvm.tir.decl_buffer(
        C.shape,
        C.dtype,
        name="BC",
        scope="wmma.accumulator",
        data_alignment=32,
        offset_factor=8,
        strides=strides_Conv,
    )

    def intrin_func(ins, outs):
        BA, BB = ins
        (BC,) = outs

        def warp_idnex(offset, row, col):
            row = row * col
            return offset // row + offset % row // col

        warp_index_A = warp_idnex(BA.elem_offset, wmma_m, wmma_k)
        warp_index_B = warp_idnex(BB.elem_offset, wmma_k, wmma_n)
        warp_index_C = warp_idnex(BC.elem_offset, wmma_m, wmma_n)

        def init():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_fill_fragment",
                    BC.data,
                    wmma_m,
                    wmma_n,
                    wmma_k,
                    warp_index_C,
                    0.0,
                )
            )
            return ib.get()

        def update():
            ib = tvm.tir.ir_builder.create()
            ib.emit(
                tvm.tir.call_intrin(
                    "handle",
                    "tir.tvm_mma_sync",
                    BC.data,
                    warp_index_C,
                    BA.data,
                    warp_index_A,
                    BB.data,
                    warp_index_B,
                    BC.data,
                    warp_index_C,
                )
            )
            return ib.get()

        return update(), init(), update()

    return te.decl_tensor_intrin(C.op, intrin_func, binds={A: BA, B: BB, C: BC})


def intrin_asm_ldmatrix(strides_dst, strides_from, shape, fragment_name, dtype, from_scope, 
                        dst_scope, trans=False, ):
    if dtype in ['float16']:
        v = 8
        A_shape = (v, 1) if trans else (v, )
        offset_factor = v if trans else 1
    elif dtype in ['float', 'float32']:
        v = 4
        if trans:
            raise ValueError("float32 ldmatrix does not support trans")
        A_shape = (v, )
        offset_factor = v
    else:
        raise ValueError(f"unsupported datatype {dtype} for ldmatrix")
    if not trans:
        strides_from = (1,)
        strides_dst = (1,)
    
    A = te.placeholder(A_shape, dtype=dtype)
    C = te.compute(A_shape, lambda *i: A(*i))
    BA = tvm.tir.decl_buffer(A.shape, A.dtype, strides=strides_from, offset_factor=offset_factor, scope=from_scope)
    BC = tvm.tir.decl_buffer(C.shape, C.dtype, strides=strides_dst, offset_factor=offset_factor, scope=dst_scope)

    mma_m, mma_n, mma_k = shape
    if "matrix_a" in fragment_name:
        index = BC.elem_offset // (mma_m * mma_k)
        num = (mma_m * mma_k) // (8*8)
        layout = "row_major"
    elif "matrix_b" in fragment_name:
        index = BC.elem_offset // (mma_n * mma_k)
        num = (mma_n * mma_k) // (8*8)
        layout = "col_major"

    def intrin_func(ins, outs):
        ib = tvm.tir.ir_builder.create()
        BA = ins[0]
        BC = outs[0]
        ib.emit(tvm.tir.call_intrin(
            "handle",
            "tir.tvm_asm_ldmatrix",
            BC.data,
            mma_m, mma_n, mma_k,
            index,
            BA.access_ptr("r"),
            num, 
            layout,
            "trans" if trans else ""
        ))
        return ib.get()
    return te.decl_tensor_intrin(C.op, intrin_func, binds={A:BA, C:BC})
