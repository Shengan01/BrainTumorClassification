
# This file handles shared utilities like timing
import time
import torch
import psutil
import os
from statistics import mean, stdev

def measure_execution_time(model, device, input_tensor):
    torch.cuda.synchronize()
    start = time.time()
    _ = model(input_tensor)
    torch.cuda.synchronize()
    end = time.time()
    return (end - start) * 1000
