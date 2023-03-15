import logging, sys
import numpy as np
import tvm
from tvm.topi.exp_cuda import conv2d_nhwc_tensorcore, schedule_conv2d_nhwc_tensorcore
from tvm import autotvm, te
import tvm.testing
from tvm.topi.testing import conv2d_nhwc_python
from tvm.contrib.auto_pipeline import (
    InjectSharedMemSwizzle,
    InjectPipelinedBuffer
)

DEBUG=False
TUNING=True
shapes = [
### RN
(64, 224, 224, 16, 64, 7, 7, (2, 2), (3, 3)),
(64, 56, 56, 64, 64, 3, 3, (1, 1), (1, 1)),
(64, 28, 28, 128, 128, 3, 3, (2, 2), (1, 1)),
(64, 28, 28, 128, 128, 3, 3, (1, 1), (1, 1)),
(64, 14, 14, 256, 256, 3, 3, (2, 2), (1, 1)),
(64, 14, 14, 256, 256, 3, 3, (1, 1), (1, 1)),
(64, 7,  7,  512, 512, 3, 3, (2, 2), (1, 1)),
(64, 7,  7,  512, 512, 3, 3, (1, 1), (1, 1)),
(64, 56, 56, 64,  128, 3, 3, (2, 2), (1, 1)),
(64, 28, 28, 128, 512, 3, 3, (2, 2), (1, 1)),
(64, 14, 14, 256, 512, 3, 3, (2, 2), (1, 1)),

### VGG16
(64, 224, 224, 16, 64, 3, 3, (1, 1), (1, 1)),
(64, 224, 224, 64, 64, 3, 3, (1, 1), (1, 1)),
(64, 112, 112, 64, 128, 3, 3, (1, 1), (1, 1)),
(64, 112, 112, 128, 128, 3, 3, (1, 1), (1, 1)),
(64, 56, 56, 128, 256, 3, 3, (1, 1), (1, 1)),
(64, 56, 56, 256, 256, 3, 3, (1, 1), (1, 1)),
(64, 28, 28, 256, 512, 3, 3, (1, 1), (1, 1)),
(64, 28, 28, 512, 512, 3, 3, (1, 1), (1, 1)),
(64, 14, 14, 512, 512, 3, 3, (1, 1), (1, 1)),

]
MAX_TRIAL=5000
DEV=0

@autotvm.template("exp_single_op_conv2d_nhwc_tensorcore")
def conv2d_nhwc(N, H, W, CI, CO, KH, KW, stride, padding, dilation=(1,1)):
    Input = te.placeholder((N, H, W, CI), name="Input", dtype="float16")
    Filter= te.placeholder((KH, KW, CO, CI), name="Filter", dtype="float16")
    Output = conv2d_nhwc_tensorcore(Input, Filter, stride, padding, dilation, "float32")
    s = schedule_conv2d_nhwc_tensorcore([Output]) # exp on, double-buffer off
    return s, [Input, Filter, Output]
    
@autotvm.template("exp_single_op_conv2d_nhwc_tensorcore_baseline")
def conv2d_nhwc_baseline(N, H, W, CI, CO, KH, KW, stride, padding, dilation=(1,1)):
    Input = te.placeholder((N, H, W, CI), name="Input", dtype="float16")
    Filter= te.placeholder((KH, KW, CO, CI), name="Filter", dtype="float16")
    Output = conv2d_nhwc_tensorcore(Input, Filter, stride, padding, dilation, "float32")
    s = schedule_conv2d_nhwc_tensorcore([Output], False) # exp off, double-buffer off
    return s, [Input, Filter, Output]
    
@autotvm.template("exp_single_op_conv2d_nhwc_tensorcore_db")
def conv2d_nhwc_db(N, H, W, CI, CO, KH, KW, stride, padding, dilation=(1,1)):
    Input = te.placeholder((N, H, W, CI), name="Input", dtype="float16")
    Filter= te.placeholder((KH, KW, CO, CI), name="Filter", dtype="float16")
    Output = conv2d_nhwc_tensorcore(Input, Filter, stride, padding, dilation, "float32")

    s = schedule_conv2d_nhwc_tensorcore([Output], False, True) # exp off, double-buffer on
    return s, [Input, Filter, Output]
    

measure_option = autotvm.measure_option(
    builder=autotvm.LocalBuilder(),
    runner=autotvm.LocalRunner(repeat=3, number=200,  timeout=300)
)

def report_best(name, _template, args, ):
    N, H, W, CI, CO, KH, KW, strides, padding, = args[:9]
    import os 
    dir_path = os.path.dirname(os.path.realpath(__file__))
    logfile = f"{dir_path}/result/{name}_%d_%d_%d_%d_%d_%d_%d_{strides}_{padding}.log"%(N,H,W,CI,CO,KH, KW)
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
            
    
    # def prepare_data():
    #     from os.path import exists
    #     if exists(f"imap_{args}.npy"):
    #             imap_np = np.load(f"imap_{args}.npy")
    #     else:
    #         imap_np = np.random.uniform(size=(N, H, W, CI)).astype(np.float16)
    #         np.save(f"imap_{args}.npy", imap_np)
    #     if exists(f"filter_{args}.npy"):
    #         filter_np = np.load(f"filter_{args}.npy")
    #         filter_trans_np = filter_np.transpose((0, 1, 3, 2))
    #     else:
    #         filter_trans_np = np.random.uniform(size=(KH, KW, CI, CO)).astype(np.float16)
    #         filter_np = filter_trans_np.transpose((0, 1, 3, 2))
    #         np.save(f"filter_{args}.npy", filter_np)
    #     if exists(f"omap_{args}.npy"):
    #         omap_np = np.load(f"omap_{args}.npy")
    #     else:
    #         filter_trans_np = filter_np.transpose((0, 1, 3, 2))
    #         omap_np = conv2d_nhwc_python(imap_np, filter_trans_np, strides, padding)
    #         np.save(f"omap_{args}.npy", omap_np)
    #     return imap_np, filter_np, omap_np
    # imap_np, filter_np, omap_np = prepare_data()

    # dev = tvm.cuda(DEV)
    # imap_tvm = tvm.nd.array(imap_np, device=dev)
    # filter_tvm = tvm.nd.array(filter_np, device=dev)
    # omap_tvm = tvm.nd.empty(omap_np.shape, device=dev)
    
    # func(imap_tvm, filter_tvm, omap_tvm)
    # _o = omap_tvm.numpy()

    # if 'baseline' not in name and '_db' not in name:
    #     tvm.testing.assert_allclose(omap_np, _o, rtol=1e-2)

    # evaluator = func.time_evaluator(func.entry_name, dev, number=400)
    # dur = evaluator(imap_tvm, filter_tvm, omap_tvm).mean
    # _, OH, OW, _ = omap_np.shape
    # print(f"Time cost of {name} %f throughput %f Tflops" % (dur, N*OH*OW*CO*CI*KH*KW*2/1e12/dur) )

opt_config={"tir.debug_keep_trivial_loop": True,
        "tir.add_lower_pass": [(3, tvm.tir.transform.Apply(InjectPipelinedBuffer())),
                                (3, tvm.tir.transform.Apply(InjectSharedMemSwizzle())),
                                (3, tvm.tir.transform.Simplify())]}
if __name__ == "__main__":
    logging.getLogger("autotvm").setLevel(logging.DEBUG)
    logging.getLogger("autotvm").addHandler(logging.StreamHandler(sys.stdout))
    
    for args in shapes:
        print("shape: " + str(args))
        # baseline
        report_best("exp_single_op_conv2d_nhwc_tensorcore_baseline", conv2d_nhwc_baseline, args, )
        # double-buffer
        report_best("exp_single_op_conv2d_nhwc_tensorcore_db", conv2d_nhwc_db, args,)
        # pipelining
        report_best("exp_single_op_conv2d_nhwc_tensorcore", conv2d_nhwc, args, )