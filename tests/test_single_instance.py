"""Only one flight daemon may run: a second copy must get no lock."""
import os
import subprocess
import sys
import tempfile
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flight_daemon import acquire_single_instance_lock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def try_lock_in_another_process(path):
    """Ask a separate Python process to take the lock; returns 'GOT' or 'BUSY'."""
    code = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {HERE!r})
        from flight_daemon import acquire_single_instance_lock
        print('GOT' if acquire_single_instance_lock({path!r}) is not None else 'BUSY')
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    return out.stdout.strip().splitlines()[-1]


def test_the_first_copy_gets_the_lock():
    with tempfile.TemporaryDirectory() as d:
        assert acquire_single_instance_lock(os.path.join(d, "l")) is not None


def test_a_second_copy_is_refused_while_the_first_runs():
    """Flight 37: two daemons sent contradicting positions to the autopilot for four minutes."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "l")
        held = acquire_single_instance_lock(path)
        assert held is not None
        assert try_lock_in_another_process(path) == "BUSY"
        os.close(held)


def test_the_lock_frees_itself_when_the_holder_dies():
    """The kernel drops it on exit - even a crash cannot leave the drone unable to start a daemon."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "l")
        assert try_lock_in_another_process(path) == "GOT"   # that process has now exited
        assert try_lock_in_another_process(path) == "GOT"


def test_the_holder_writes_its_pid():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "l")
        held = acquire_single_instance_lock(path)
        with open(path) as fh:
            assert fh.read().strip() == str(os.getpid())
        os.close(held)
