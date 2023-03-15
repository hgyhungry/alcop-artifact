import logging, sys
import numpy as np
import tvm
from tvm.topi.exp_cuda import dense_tensorcore, schedule_dense_tensorcore
from tvm import autotvm, te
import tvm.testing
from tvm.contrib.auto_pipeline import (
    InjectSharedMemSwizzle,
    InjectPipelinedBuffer
)

DEBUG=False
TUNING=True
shapes = [
    (1024,768,768),
    # (1024,768*3,768),
    # (1024,3072,768),
    # (1024,768,3072),
    # (512,768,768),
    # (512,768*3,768),
    # (512,3072,768),
    # (512,768,3072),
    # (512,1024,1024),
    # (512,1024*3,1024),
    # (512,1024,4096),
    # (512,4096,1024),
    # (1024,1024,1024),
    # (2048,2048,2048),
    # (4096,4096,4096),
    # (1024,64,512),
    # (50176,128,64),
    # (12544,256,128),
    # (3136,512,256),
    # (200704,64,64),
    # (200704,64,256),
    # (200704,256,64),
    # (200704,256,64),
    # (50176,128,256),
    # (50176,128,512),
    # (50176,512,128),
    # (50176,512,256),
    # (50176,512,128),
    # (12544,256,512),
    # (12544,256,1024),
    # (12544,1024,256),
    # (12544,1024,512),
    # (12544,1024,256),
    # (3136,512,1024),
    # (3136,512,2048),
    # (3136,2048,512),
    # (3136,2048,1024),
    # (3136,2048,512),
    # (1024,64,2048),
    # (200704,256,256),
    # (12544,512,512),
    # (3136,512,512),
    # (4096,64,512),
    # (4096,64,4096),
    # (1024,64,4096 )
    ]
# shapes=[(2048,2048,2048),]
MAX_TRIAL=5000
DEV=0

@autotvm.template("exp_single_op_dense_tensorcore")
def dense(M, N, K):
    # do not turn on 'exp' flag, so no pipelining is used
    A = te.placeholder((M, K), name="A", dtype='float16')
    B = te.placeholder((N, K), name="B", dtype="float16")
    C = dense_tensorcore(A, B, 'float32')
    s = schedule_dense_tensorcore([C])
    return s, [A, B, C]

@autotvm.template("exp_single_op_dense_tensorcore_baseline")
def dense_baseline(M, N, K):
    # do not turn on 'exp' flag, so no pipelining is used
    A = te.placeholder((M, K), name="A", dtype='float16')
    B = te.placeholder((N, K), name="B", dtype="float16")
    C = dense_tensorcore(A, B, 'float32')
    s = schedule_dense_tensorcore([C], False)
    return s, [A, B, C]

@autotvm.template("exp_single_op_dense_tensorcore_db")
def dense_db(M, N, K):
    # do not turn on 'exp' flag, so no pipelining is used
    # but turn on 'double_buffer' flag and enable TVM-builtin double-buffering
    A = te.placeholder((M, K), name="A", dtype='float16')
    B = te.placeholder((N, K), name="B", dtype="float16")
    C = dense_tensorcore(A, B, 'float32')

    s = schedule_dense_tensorcore([C], False, True)
    return s, [A, B, C]

measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, number=200, timeout=100,)
)

opt_config={"tir.debug_keep_trivial_loop": True,
        "tir.add_lower_pass": [(3, tvm.tir.transform.Apply(InjectPipelinedBuffer())),
                                (3, tvm.tir.transform.Apply(InjectSharedMemSwizzle())),
                                (3, tvm.tir.transform.Simplify())]}
def report_best(name, _template, M, N, K, tuning=True):
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    logfile = f"{dir_path}/result/{name}_{M}_{N}_{K}.log"
    if tuning:
        _task = autotvm.task.create(name, args=(M, N, K), target="cuda")
        _tuner = autotvm.tuner.GridSearchTuner(_task)
        # with tvm.transform.PassContext(config=_config):
        _tuner.tune(
            n_trial=MAX_TRIAL,
            measure_option=measure_option,
            callbacks=[autotvm.callback.log_to_file(logfile)],
        )

        dispatch_context = autotvm.apply_history_best(logfile)
        best_config = dispatch_context.query(_task.target, _task.workload)
        print(f"{name}: \nBest config:")
        print(best_config)

    # # apply history best from log file
    # with autotvm.apply_history_best(logfile):
    #     with tvm.target.Target("cuda"):
    #         s, arg_bufs = _template(M, N, K)
    #         # if DEBUG:
    #         #     print(tvm.lower(s, arg_bufs, simple_mode=True))
    #         with tvm.transform.PassContext(config=opt_config):
    #             ir_mod = tvm.lower(s, arg_bufs)
    #             if DEBUG:
    #                 print(ir_mod.script())
    #             func = tvm.build(ir_mod)
    #             if DEBUG:
    #                 with open(f"{name}.cu", "w") as f:
    #                     f.write(func.imported_modules[0].get_source())
                
    #         # func = tvm.build(s, arg_bufs)

    # a_np = np.random.uniform(size=(M, K)).astype(np.float16)
    # b_np = np.random.uniform(size=(N, K)).astype(np.float16)
    # c_np = a_np.astype(np.float32) @ (b_np.astype(np.float32).T)

    # dev = tvm.cuda(DEV)
    # a_tvm = tvm.nd.array(a_np, device=dev)
    # b_tvm = tvm.nd.array(b_np, device=dev)
    # c_tvm = tvm.nd.empty(c_np.shape, device=dev)
    # func(a_tvm, b_tvm, c_tvm)
    # # print(np.where(np.isnan(c_np)))
    # # print(np.where(np.isnan(c_tvm.numpy())))
    # _c = c_tvm.numpy()
    # if 'baseline' not in name and '_db' not in name:
    #     tvm.testing.assert_allclose(c_np, _c, rtol=1e-2)
    # evaluator = func.time_evaluator(func.entry_name, dev, number=400)
    # dur = evaluator(a_tvm, b_tvm, c_tvm).mean
    # print(f"Time cost of {name} %f throughput %f Tflops" % (dur, M*N*K*2/dur/1e12) )

if __name__ == "__main__":
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))

    for (M, N, K) in shapes:
        print("shape: " + str((M, N, K)))
        # baseline
        report_best("exp_single_op_dense_tensorcore_baseline", dense_baseline, M, N, K, TUNING)
        # double-buffer
        report_best("exp_single_op_dense_tensorcore_db", dense_db, M, N, K,TUNING)
        # pipelining
        report_best("exp_single_op_dense_tensorcore", dense, M, N, K, TUNING)