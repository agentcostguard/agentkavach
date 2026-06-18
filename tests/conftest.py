"""Pytest fixtures for the AgentKavach SDK test suite."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _disable_client_tracer(request):
    """Keep client-built OTel tracers out of the suite.

    ``api_key`` is mandatory, so every ``AgentKavach(...)`` construction would
    otherwise build a network-bound ``BatchSpanProcessor`` pointed at the
    backend and route events to OTel instead of the in-process buffer that most
    tests inspect. Patching ``create_tracer_provider`` to fail leaves
    ``_tracer = None`` (the constructor catches it, fail-open), so events land in
    the buffer and no test touches the network.

    Tests that intentionally exercise the client -> tracer path opt out with
    ``@pytest.mark.real_tracer``.
    """
    if request.node.get_closest_marker("real_tracer"):
        yield
        return
    with patch(
        "agentkavach.client.create_tracer_provider",
        side_effect=RuntimeError("tracer disabled in tests"),
    ):
        yield
