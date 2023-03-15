import logging, sys
import tvm
from tvm import autotvm, te
from tvm.topi.exp_cuda import (
    batch_matmul_tensorcore, schedule_batch_matmul_tensorcore
)
import numpy as np
import tvm.testing
from tvm.contrib.auto_pipeline import (
    InjectSharedMemSwizzle,
    InjectPipelinedBuffer
)

DEBUG=False
TUNING=True
DEV=0

shapes=[(12,512,512,64),
(12,512,64,512),
(12,1024,1024,64),
(12,1024,64,1024),
(16,512,512,64),
(16,512,64,512),
]
MAX_TRIAL=5000

@autotvm.template("exp_single_op_batch_matmul_tensorcore")
def bmm_tensorcore(B, M, N, K):
    A = te.placeholder((B, M, K), name="A", dtype="float16")
    B = te.placeholder((B, N, K), name="B", dtype="float16")
    C = batch_matmul_tensorcore(A, B, (B, M, N), "float32")
    s = schedule_batch_matmul_tensorcore([C]) # exp on (enable pipelining)
    return s, [A, B, C]

@autotvm.template("exp_single_op_batch_matmul_tensorcore_baseline")
def bmm_tensorcore_baseline(B, M, N, K):
    A = te.placeholder((B, M, K), name="A", dtype="float16")
    B = te.placeholder((B, N, K), name="B", dtype="float16")
    C = batch_matmul_tensorcore(A, B, (B, M, N), "float32")
    s = schedule_batch_matmul_tensorcore([C], False) # exp off, double-buffer off
    return s, [A, B, C]

@autotvm.template("exp_single_op_batch_matmul_tensorcore_db")
def bmm_tensorcore_db(B, M, N, K):
    A = te.placeholder((B, M, K), name="A", dtype="float16")
    B = te.placeholder((B, N, K), name="B", dtype="float16")
    C = batch_matmul_tensorcore(A, B, (B, M, N), "float32")
    s = schedule_batch_matmul_tensorcore([C], False, True) # exp off, double-buffer on
    return s, [A, B, C]

measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, number=200,  timeout=50)
)

def report_best(name, _template, args):
    B, M, N, K = args
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    logfile = f"{dir_path}/result/{name}_{B}_{M}_{N}_{K}.log"
    if TUNING:
        _task = autotvm.task.create(name, args=args, target="cuda")
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
    #         s, arg_bufs = _template(*args)
    #         with tvm.transform.PassContext(config=opt_config):
    #             ir_mod = tvm.lower(s, arg_bufs)
    #             if DEBUG:
    #                 print(ir_mod.script())
    #             func = tvm.build(ir_mod)
    #             if DEBUG:
    #                 with open(f"{name}.cu", "w") as f:
    #                     f.write(func.imported_modules[0].get_source())
                
    #         # func = tvm.build(s, arg_bufs)

    # a_np = np.random.uniform(size=(B, M, K)).astype(np.float16)
    # b_np = np.random.uniform(size=(B, N, K)).astype(np.float16)
    # c_np = np.matmul(a_np.astype(np.float32),(b_np.astype(np.float32).transpose((0,2,1))))

    # dev = tvm.cuda(DEV)
    # a_tvm = tvm.nd.array(a_np, device=dev)
    # b_tvm = tvm.nd.array(b_np, device=dev)
    # c_tvm = tvm.nd.empty(c_np.shape, device=dev)
    # func(a_tvm, b_tvm, c_tvm)
    # _c = c_tvm.numpy()

    # # print(np.where(np.isnan(c_np)))
    # # print(np.where(np.isnan(c_tvm.numpy())))
    # if 'baseline' not in name and '_db' not in name:
    #     tvm.testing.assert_allclose(c_np, _c, rtol=1e-2)
    # evaluator = func.time_evaluator(func.entry_name, dev, number=400)
    # dur = evaluator(a_tvm, b_tvm, c_tvm).mean
    # print(f"Time cost of {name} %f throughput %f Tflops" % (dur, B*M*N*K*2/dur/1e12) )

opt_config={"tir.debug_keep_trivial_loop": True,
        "tir.add_lower_pass": [(3, tvm.tir.transform.Apply(InjectPipelinedBuffer())),
                                (3, tvm.tir.transform.Apply(InjectSharedMemSwizzle())),
                                (3, tvm.tir.transform.Simplify())]}
if __name__ == "__main__":
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))
    
    with tvm.transform.PassContext(config=opt_config):
        for args in shapes:
            print("shape: " + str(args))
            # baseline
            report_best("exp_single_op_batch_matmul_tensorcore_baseline", bmm_tensorcore_baseline, args, )
            # double-buffer
            report_best("exp_single_op_batch_matmul_tensorcore_db", bmm_tensorcore_db, args,)
            # pipelining
            report_best("exp_single_op_batch_matmul_tensorcore", bmm_tensorcore, args, )