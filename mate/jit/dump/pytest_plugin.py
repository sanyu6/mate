from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from .core import (
    MateJITDumpComplete,
    begin_pytest_dump_case,
    end_pytest_dump_case,
    finish_pytest_dump_session,
    pytest_dump_count,
    start_pytest_dump_session,
)
from .context import JITDumpContext


class JITDumpPytestPlugin:
    def __init__(
        self,
        output_root: Path,
        context: JITDumpContext | None = None,
    ) -> None:
        self.output_root = output_root.expanduser().resolve()
        if self.output_root.exists():
            warnings.warn(
                pytest.PytestWarning(
                    f"dump output directory already exists; generated artifacts "
                    f"will be overwritten: {self.output_root}"
                ),
                stacklevel=1,
            )
        start_pytest_dump_session(self.output_root, context)

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: pytest.Item | None):
        begin_pytest_dump_case(item.nodeid)
        try:
            result = yield
        finally:
            dumped = end_pytest_dump_case(item.nodeid)
            if not dumped:
                warnings.warn(
                    pytest.PytestWarning(
                        f"pytest item did not produce a dump artifact: {item.nodeid}"
                    ),
                    stacklevel=1,
                )
        return result

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo[None]):
        report = yield
        if (
            call.when == "call"
            and call.excinfo is not None
            and call.excinfo.errisinstance(MateJITDumpComplete)
        ):
            report.outcome = "passed"
            report.longrepr = None
        return report

    def pytest_terminal_summary(
        self, terminalreporter: pytest.TerminalReporter
    ) -> None:
        count = pytest_dump_count()
        terminalreporter.write_line(f"dumped {count} case(s) under {self.output_root}")

    def pytest_unconfigure(self, config: pytest.Config) -> None:
        finish_pytest_dump_session()
