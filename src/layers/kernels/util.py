import triton
import triton.language as tl
import torch


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


# enable autotune printing
# os.environ['TRITON_AUTOTUNE_PRINT'] = '1'


# specify configs
if is_hip():
    NUM_STAGES_OPTIONS = [1]
else:
    NUM_STAGES_OPTIONS = [1, 2, 3, 4]