import contextlib
from contextlib import ExitStack

from omnihub.tools.trace import TraceManager

tracers = TraceManager()


@contextlib.contextmanager
def profile():
    with ExitStack() as stack:
        # default omnihub monitor is always used
        stack.enter_context(tracers.omnihub_monitor_cm())
        if tracers.use_omnitrace:
            # enter omnitrace context manager
            stack.enter_context(tracers.get_omnitrace_cm())
        if tracers.use_pytorch_profiler_stats or tracers.use_pytorch_profiler_trace:
            # enter pytorch profiler context manager
            stack.enter_context(tracers.get_pytorch_profiler_cm())
        # if use_other_tool:
        #    stack.enter_context(get_other_tool_cm())
        yield
