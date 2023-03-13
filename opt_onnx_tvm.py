import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
import onnx
import numpy as np
import tvm
from tvm import relay



# 自动优化TVM
onnx_model = onnx.load('your_out.onnx')
target = tvm.target.Target('llvm -mcpu=skylake')
# target = tvm.target.Target('llvm -mcpu=core-avx2')

input_name = 'input'
shape_dict = {input_name: (1, 1, 128, 128)}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
number = 10
repeat = 1
min_repeat_ms = 0
timeout = 10
# create a TVM runner
runner = autotvm.LocalRunner(
    number=number,  # number specifies the number of different configurations that we will test
    repeat=repeat,  # repeat specifies how many measurements we will take of each configuration
    timeout=timeout,  # in seconds,The timeout places an upper limit on how long to run training code for each tested configuration.
    min_repeat_ms=min_repeat_ms,  # min_repeat_ms is a value that specifies how long need to run configuration test. If the number of repeats falls under this time, it will be increased. This option is necessary for accurate tuning on GPUs, and is not required for CPU tuning. Setting this value to 0 disables it.
    enable_cpu_cache_flush=True,
)

tuning_option = {
    "tuner": "xgb",
    "trials": 1500,
    "early_stopping": 100,
    "measure_option": autotvm.measure_option(
        builder=autotvm.LocalBuilder(build_func="default"), runner=runner
    ),
    "tuning_records": "resnet-50-v2-autotuning2.json",
}
# begin by extracting the tasks from the onnx model
tasks = autotvm.task.extract_from_program(mod["main"], target=target, params=params)

# Tune the extracted tasks sequentially.
for i, task in enumerate(tasks):
    prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
    tuner_obj = XGBTuner(task, loss_type="rank")
    tuner_obj.tune(
        n_trial=min(tuning_option["trials"], len(task.config_space)),
        early_stopping=tuning_option["early_stopping"],
        measure_option=tuning_option["measure_option"],
        callbacks=[
            autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
            autotvm.callback.log_to_file(tuning_option["tuning_records"]),
        ],
    )

with autotvm.apply_history_best(tuning_option["tuning_records"]):
    with tvm.transform.PassContext(opt_level=3):
        graph, lib, params = relay.build(mod, target, params=params)

from tvm.contrib import graph_runtime

libpath = "test.so"
lib.export_library(libpath)
graph_json_path = 'test.json'
with open(graph_json_path, 'w') as f:
    f.write(graph)

param_path = "./test.params"
with open(param_path, 'wb') as fo:
    fo.write(relay.save_param_dict(params))
