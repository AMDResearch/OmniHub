import contextlib
import importlib.util
import sys
from contextlib import ExitStack


class Tracers:
    def __init__(self):
        self.use_omnitrace = False

    def enable_omnitrace(self):
        self.use_omnitrace = True

    # Add other tools in future


tracers = Tracers()


def get_omnitrace_cm():
    # Attempt to add the omnitrace python module, which is installed in
    # /opt/omnitrace in the Docker image. To run outside of the Docker with
    # omnitrace installed in a different path, set PYTHONPATH accordingly.
    sys.path.insert(0, "/opt/omnitrace/lib/python/site-packages/")
    omnitrace_spec = importlib.util.find_spec("omnitrace")
    if omnitrace_spec is None:
        print("Unable to find omnitrace module")
        sys.exit(1)
    from omnitrace import profile

    return profile()


@contextlib.contextmanager
def profile():
    with ExitStack() as stack:
        if tracers.use_omnitrace:
            # enter omnitrace context manager
            stack.enter_context(get_omnitrace_cm())
        # enter other context managers in future if necessary
        # if use_other_tool:
        #    stack.enter_context(get_other_tool_cm())
        yield
