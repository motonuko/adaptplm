import warnings

import torch


def get_available_device(cpu_or_gpu: bool = False) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if not cpu_or_gpu and torch.backends.mps.is_available():
        # MPS does not work with PyCharm debugger somehow.
        return "mps"
    return "cpu"


def get_device(use_cpu: bool) -> str:
    if use_cpu:
        return 'cpu'
    elif torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        # MPS does not work with PyCharm debugger somehow.
        return "mps"
    warnings.warn("gpu and mps backend are not available")
    return "cpu"


class MockSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


# To avoid errors with SummaryWriter while debugging in PyCharm
def get_summary_writer(output_dir: str, is_debug=False):
    if is_debug:
        return MockSummaryWriter()
    else:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(output_dir)
