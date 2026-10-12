"""Shared pytest configuration."""

import os
import sys

_exit_status = 0


def pytest_sessionfinish(session, exitstatus):
    global _exit_status
    _exit_status = int(exitstatus)


def pytest_unconfigure(config):
    # PySide6 6.12 can abort during interpreter finalisation ("bool_dealloc") after
    # every test has passed. Once pytest has reported, leave without finalising so
    # that a teardown bug in Qt cannot turn a green run red.
    if "PySide6" in sys.modules:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(_exit_status)
