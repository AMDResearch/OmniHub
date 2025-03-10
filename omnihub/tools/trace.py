import contextlib
import importlib.util
import json
import os
import sys
import time

import torch

from omnihub.run import distributed as dist


# This class is used to enable/disable different tracers
class TraceManager:
    def __init__(self):
        self.use_omnitrace = False
        self.use_pytorch_profiler_stats = False
        self.use_pytorch_profiler_trace = False

    def enable_omnitrace(self):
        self.use_omnitrace = True

    def enable_pytorch_profiler_stats(self):
        self.use_pytorch_profiler_stats = True

    def enable_pytorch_profiler_trace(self):
        self.use_pytorch_profiler_trace = True

    # Add other tools in future

    @contextlib.contextmanager
    def omnihub_monitor_cm(self):
        try:
            import amdsmi

            amdsmi.amdsmi_init()
        except ImportError:
            print("Unable to find amdsmi module")
            sys.exit(1)

        def get_energy(device_num: int):
            handles = amdsmi.amdsmi_get_processor_handles()
            energy_dict = amdsmi.amdsmi_get_energy_count(handles[device_num])
            energy = energy_dict["power"] * round(energy_dict["counter_resolution"], 1)
            energy /= 1000000000
            return energy

        output_dir = os.getenv("OMNIHUB_MONITOR_OUTPUT_PATH", ".")
        os.makedirs(output_dir, exist_ok=True)
        node_id = int(os.getenv("RANK", 0)) // torch.cuda.device_count()
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        output_file = os.path.join(
            output_dir, f"omnihub_monitor_output_{node_id}_{local_rank}.json"
        )

        if dist.is_initialized():
            start_energies = [get_energy(local_rank)]
        else:
            start_energies = [get_energy(i) for i in range(torch.cuda.device_count())]
        start_time = time.time()
        yield
        end_time = time.time()
        if dist.is_initialized():
            end_energies = [get_energy(local_rank)]
        else:
            end_energies = [get_energy(i) for i in range(torch.cuda.device_count())]

        energy_kj = [
            round(end - start, 3) for start, end in zip(start_energies, end_energies)
        ]
        energy_kwh = [round(energy / 3600, 6) for energy in energy_kj]

        with open(output_file, "w") as f:
            data = {
                "Node": node_id,
                "Rank": local_rank,
                "TimeUnits": "seconds",
                "StartTime": round(start_time, 3),
                "EndTime": round(end_time, 3),
                "Duration": round(end_time - start_time, 3),
                "EnergyUnits": "kWh",
                "Energy": [
                    {"Device": i, "Energy": round(energy, 6)}
                    for i, energy in enumerate(energy_kwh)
                ],
                "TotalEnergy": round(sum(energy_kwh), 6),
            }

            json.dump(data, f, indent=4)

    def get_omnitrace_cm(self):
        # Attempt to add the omnitrace python module, which is installed in
        # /opt/omnitrace in the OmniHub images. To run outside of OmniHub with
        # omnitrace installed in a different path, set PYTHONPATH accordingly.
        sys.path.insert(0, "/opt/omnitrace/lib/python/site-packages/")
        omnitrace_spec = importlib.util.find_spec("omnitrace")
        if omnitrace_spec is None:
            print("Unable to find omnitrace module")
            sys.exit(1)
        from omnitrace import profile

        return profile()

    def _setup_pytorch_profiler(self):
        pre_hook_counter = 0

        # Define forward pre-hook
        def forward_pre_hook(module, input):
            nonlocal pre_hook_counter
            pre_hook_counter += 1
            if not hasattr(module, "call_stack"):
                module.call_stack = type(module).__name__

            for name, child in module.named_children():
                if not hasattr(child, "call_stack"):
                    child.call_stack = f"{module.call_stack}.{name}"

            with torch.profiler.record_function(
                f"s:{pre_hook_counter}:{module.call_stack}"
            ):
                pass

        # Define forward hook
        def forward_hook(module, input, output):
            with torch.profiler.record_function(f"f:{module.call_stack}"):
                pass

        torch.nn.modules.module.register_module_forward_pre_hook(forward_pre_hook)
        torch.nn.modules.module.register_module_forward_hook(forward_hook)

    def get_pytorch_profiler_cm(self):
        try:
            import torch.profiler as profiler
        except ImportError:
            print("Unable to find torch.profiler module")
            sys.exit(1)

        output_dir = os.getenv("PYTORCH_PROFILER_OUTPUT_PATH", ".")

        # TensorBoard trace handler
        tensorboard_handler = None
        if int(os.getenv("LOCAL_RANK", 0)) == 0:
            self._setup_pytorch_profiler()
            tensorboard_handler = profiler.tensorboard_trace_handler(output_dir)

        # Custom trace handler
        def stats_handler(prof):
            """
            Handles the profiling statistics and writes them to JSON files for both GPU and CPU.
            Note:
                The PyTorch profiler seems to be called for every rank and it somehow aggregates profiling stats for all GPUs if not called in DDP mode.
            """

            os.makedirs(output_dir, exist_ok=True)
            node_id = int(os.getenv("RANK", 0)) // torch.cuda.device_count()
            local_rank = int(os.getenv("LOCAL_RANK", 0))

            gpu_output_file = os.path.join(
                output_dir, f"gpu_profiler_output_{node_id}_{local_rank}.json"
            )
            gpu_trace_data = prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=10
            )
            with open(gpu_output_file, "w") as gpu_f:
                gpu_f.write(str(gpu_trace_data))

            cpu_output_file = os.path.join(
                output_dir, f"cpu_profiler_output_{node_id}_{local_rank}.json"
            )
            cpu_trace_data = prof.key_averages().table(
                sort_by="cpu_time_total", row_limit=10
            )
            with open(cpu_output_file, "w") as cpu_f:
                cpu_f.write(str(cpu_trace_data))

        trace_handler = None
        if self.use_pytorch_profiler_stats:
            trace_handler = stats_handler
        elif self.use_pytorch_profiler_trace:
            trace_handler = tensorboard_handler

        return profiler.profile(
            activities=[
                profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            profile_memory=True,
            on_trace_ready=trace_handler,
        )
