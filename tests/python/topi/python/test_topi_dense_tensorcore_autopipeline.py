"""Test code for dense tensorcore operator with auto pipelining"""
import numpy as np
import tvm
from tvm import topi
import tvm.topi.testing
from tvm.topi.utils import get_const_tuple
from tvm import te
from tvm.contrib.pickle_memoize import memoize
import tvm.testing

from tvm.contrib.auto_pipeline import (
    InjectSharedMemSwizzle,
    InjectPipelinedBuffer
)

_dense_implement = {"gpu": [(topi.exp_cuda.dense_tensorcore, topi.exp_cuda.schedule_dense_tensorcore)]}

def verify_dense(batch, in_dim, out_dim, dtype):
    use_bias = False
    """Dense tensorcore verify function"""
    A = te.placeholder((batch, in_dim), name="A", dtype=dtype)
    B = te.placeholder((out_dim, in_dim), name="B", dtype=dtype)

    assert dtype in ["float16"]

    out_dtype = "float32"

    # use memoize to pickle the test data for next time use
    @memoize("topi.tests.test_topi_dense_tensorcore_exp")
    def get_ref_data():
        a_np = np.random.uniform(size=(batch, in_dim)).astype(dtype)
        b_np = np.random.uniform(size=(out_dim, in_dim)).astype(dtype)
        d_np = tvm.topi.testing.dense(a_np, b_np, None, use_bias, False, out_dtype)
        return (a_np, b_np, d_np)

    # get the test data
    a_np, b_np, d_np = get_ref_data()

    def check_device(device):
        dev = tvm.device(device, 0)
        print("Running on target: %s" % device)
        for fcompute, fschedule in tvm.topi.testing.dispatch(device, _dense_implement):
            with tvm.target.Target(device):
                D = fcompute(A, B, out_dtype)
                s = fschedule([D])
            a = tvm.nd.array(a_np, dev)
            b = tvm.nd.array(b_np, dev)
            d = tvm.nd.array(np.zeros(get_const_tuple(D.shape), dtype=out_dtype), dev)

            # calling pipelining passes
            config={"tir.debug_keep_trivial_loop": True,
                    "tir.add_lower_pass": [(3, tvm.tir.transform.Apply(InjectPipelinedBuffer())),
                                            (3, tvm.tir.transform.Apply(InjectSharedMemSwizzle())),
                                            (3, tvm.tir.transform.Simplify())]}
            with tvm.transform.PassContext(config=config):
                mod = tvm.lower(s, [A, B, D])
            f = tvm.build(mod, target=device, name="dense")
            f(a, b, d)
            tvm.testing.assert_allclose(d.numpy(), d_np, rtol=1e-3)

    check_device("cuda")


@tvm.testing.requires_tensorcore
def test_dense_tensorcore_exp():
    """Test cases"""
    for dtype in ["float16"]:
        verify_dense(1024, 1024, 1024, dtype,)


if __name__ == "__main__":
    test_dense_tensorcore_exp()
    print("test completed.")
