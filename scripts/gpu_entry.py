#!/usr/bin/env python3
"""Resource-limited entry into the unmodified official CLI."""
import os
import runpy
import sys
from pathlib import Path
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(2)
if not torch.cuda.is_available():
    raise RuntimeError('A verified L40S is required for this reproduction')
name = torch.cuda.get_device_name(0)
if 'L40S' not in name or 'H20' in name:
    raise RuntimeError(f'Unexpected GPU: {name}')
total = torch.cuda.get_device_properties(0).total_memory
cap = int(os.environ.get('EGO_GPU_CAP_MIB', '16384')) * 1024 ** 2
torch.cuda.set_per_process_memory_fraction(cap / total, 0)
target = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target.parent))
sys.argv = [str(target), *sys.argv[2:]]
print(f'GPU resource guard: {name}; allocator cap={cap // 1024 ** 2} MiB; MPS={os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")}', flush=True)
runpy.run_path(str(target), run_name='__main__')
