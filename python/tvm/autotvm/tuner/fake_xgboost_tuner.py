"""
Guyue: this tuner is just a tool for evaluating performance model. 
When i have run auto-tvm to exhaustively search an op for once, I have the 
log file for all schedules. Next, when I want to change the search model
and see the tuning efficiency, instead of compiling and profiling the kernel
again (very time consuming) I can simply read the measured result from log. 
This helps me quickly evaluate different search models. But this is not an 
actual tuner because it relies on full history logs.
"""

from tvm.autotvm.measure.measure import MeasureErrorNo, MeasureResult
from .xgboost_tuner import XGBTuner
from ..measure import MeasureInput, create_measure_batch
from ..utils import format_si_prefix
import logging
import numpy as np
import tempfile
from..record import load_from_file
import time
from ..env import GLOBAL_SCOPE

logger = logging.getLogger("autotvm")


def fake_measure(inputs):
    # this function just reads the log file instead of running test on hardware
    # Because I just want to evaluate the search efficiency, and I have all test logs,
    # I can use this function to improve evaluation efficiency without any change to
    # accuracy.
    #     
    ret = []
    for input in inputs:
        task_name = input.task.name
        args="_".join([str(arg) for arg in input.task.args])
        logfile_path = f"/work/tvm_memory/hgy_tvm/pipeline/exp/single_op/result/{task_name}_{args}.log"
        for (inp, res) in load_from_file(logfile_path):
            if inp.config.index == input.config.index:
                ret.append( MeasureResult(res.costs, MeasureErrorNo( res.error_no), res.all_cost, time.time()) )
                break
    # import pdb; pdb.set_trace()
    assert len(inputs) == len(ret)
    return ret


class FakeXGBoostTuner(XGBTuner):
    def __init__(self, task, plan_size=64, feature_type="itervar", loss_type="rank", num_threads=None, optimizer="sa", diversity_filter_ratio=None, log_interval=50):
        super().__init__(task, plan_size, feature_type, loss_type, num_threads, optimizer, diversity_filter_ratio, log_interval)
    
    def tune(self, n_trial, measure_option, early_stopping=None, callbacks=(), si_prefix="G"):
        ###
        ### This part of the code is copied from the Tuner.tune(), but 
        ### changes the measure() to the fake_measure() 
        ###

        # measure_batch = create_measure_batch(self.task, measure_option)
        # n_parallel = getattr(measure_batch, "n_parallel", 1)
        # import pdb; pdb.set_trace()
        n_parallel = 1
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping

        # Validate si_prefix arg
        format_si_prefix(0, si_prefix)

        old_level = logger.level

        GLOBAL_SCOPE.in_tuning = True
        i = error_ct = 0
        errors = []
        while i < n_trial:
            if not self.has_next():
                break

            configs = self.next_batch(min(n_parallel, n_trial - i))

            inputs = [MeasureInput(self.task.target, self.task, config) for config in configs]
            results = fake_measure(inputs)

            # keep best config
            for k, (inp, res) in enumerate(zip(inputs, results)):
                config = inp.config
                if res.error_no == 0:
                    flops = inp.task.flop / np.mean(res.costs)
                    error_ct = 0
                else:
                    flops = 0
                    error_ct += 1
                    error = res.costs[0]
                    if isinstance(error, str):
                        errors.append(error)
                    else:
                        errors.append(str(error))

                if flops > self.best_flops:
                    self.best_flops = flops
                    self.best_config = config
                    self.best_measure_pair = (inp, res)
                    self.best_iter = i + k

                logger.debug(
                    "No: %d\t%sFLOPS: %.2f/%.2f\tresult: %s\t%s",
                    i + k + 1,
                    si_prefix,
                    format_si_prefix(flops, si_prefix),
                    format_si_prefix(self.best_flops, si_prefix),
                    res,
                    config,
                )

            i += len(results)
            self.ttl = min(early_stopping + self.best_iter, n_trial) - i

            self.update(inputs, results)
            for callback in callbacks:
                callback(self, inputs, results)

            if i >= self.best_iter + early_stopping:
                logger.debug("Early stopped. Best iter: %d.", self.best_iter)
                break

            if error_ct > 150:
                logging.basicConfig()
                logger.warning("Too many errors happen in the tuning. Switching to debug mode.")
                logger.setLevel(logging.DEBUG)
            else:
                logger.setLevel(old_level)

        if error_ct == i:
            _, f = tempfile.mkstemp(prefix="tvm_tuning_errors_", suffix=".log", text=True)
            with open(f, "w") as file:
                file.write("\n".join(errors))
            logging.warning(
                "Could not find any valid schedule for task %s. "
                "A file containing the errors has been written to %s.",
                self.task,
                f,
            )
        GLOBAL_SCOPE.in_tuning = False
        # del measure_batch

        ### XGBTuner
        self.cost_model._close_pool()