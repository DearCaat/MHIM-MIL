# benchmark_pytorch.py
# The script is copied from https://leimao.github.io/blog/PyTorch-Benchmark/
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torchvision
import timm
import torch.utils.benchmark as benchmark
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from options import _parse_args, more_about_config

from modules import build_model


@torch.no_grad()
def run_inference(model: nn.Module,
                  input_tensor: torch.Tensor) -> torch.Tensor:

    return model.forward(input_tensor)

@torch.no_grad()
def measure_time_device(
    model: nn.Module,
    input_tensor: torch.Tensor,
    num_repeats: int = 100,
    num_warmups: int = 10,
    synchronize: bool = True,
    continuous_measure: bool = True,
) -> float:

    '''
        synchronize and continuous_measure should always be True
    '''
    assert synchronize == True and continuous_measure == True

    for _ in range(num_warmups):
        _ = model.forward(input_tensor)
    torch.cuda.synchronize()

    elapsed_time_ms = 0

    if continuous_measure:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        for _ in range(num_repeats):
            _ = model.forward(input_tensor)


        end_event.record()
        if synchronize:
            torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)

    else:
        for _ in range(num_repeats):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            _ = model.forward(input_tensor)


            end_event.record()
            if synchronize:
                torch.cuda.synchronize()
            elapsed_time_ms += start_event.elapsed_time(end_event)

    return elapsed_time_ms / num_repeats

def main() -> None:

    num_warmups = 10
    num_repeats = 100
    input_shape = (10000, 3, 224, 224)

    device = torch.device("cuda:0")

    args, args_text = _parse_args()
    args,device = more_about_config(args)

    model,_ = build_model(args, device, None)

    model.to(device)
    model.eval()

    # Input tensor
    input_tensor = torch.rand(input_shape)

    torch.cuda.synchronize()
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        print("Latency Measurement Using PyTorch Benchmark...")
        num_threads = 1
        timer = benchmark.Timer(stmt="run_inference(model, input_tensor)",
                                setup="from __main__ import run_inference",
                                globals={
                                    "model": model,
                                    "input_tensor": input_tensor
                                },
                                num_threads=num_threads,
                                label="Latency Measurement",
                                sub_label="torch.utils.benchmark.")

        profile_result = timer.timeit(num_repeats)
    # https://pytorch.org/docs/stable/_modules/torch/utils/benchmark/utils/common.html#Measurement
    print(f"Latency: {profile_result.mean * 1000:.5f} ms")


if __name__ == "__main__":

    main()