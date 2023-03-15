import tvm
from tvm import te
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
