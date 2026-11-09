"""Own and cancel only process groups launched by this evaluation batch."""
from concurrent.futures import CancelledError
from contextlib import contextmanager
import os
import signal
import subprocess
import threading


def stop_group(process, grace=2.0):
    # Every process here was started with start_new_session=True. Do not use
    # caller-supplied/unresolved group IDs or signal the evaluation's own group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # A child can outlive its group leader or ignore SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


class ProcessRegistry:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = []
        self.stopping = False

    def start(self, command, **kwargs):
        if kwargs.get("start_new_session") is not True:
            raise ValueError("Managed workers require a dedicated process group")
        with self.lock:
            if self.stopping:
                raise CancelledError("Evaluation batch has been stopped")
            process = subprocess.Popen(command, **kwargs)
            self.active.append(process)
            return process

    def forget(self, process):
        with self.lock:
            if process in self.active:
                self.active.remove(process)

    def stop(self, grace=2.0):
        with self.lock:
            self.stopping = True
            processes = list(self.active)
        for process in processes:
            stop_group(process, grace)


@contextmanager
def termination_signals():
    """Convert batch SIGTERM to unwinding so worker groups get cleaned up."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt("Evaluation terminated")

    signal.signal(signal.SIGTERM, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)
