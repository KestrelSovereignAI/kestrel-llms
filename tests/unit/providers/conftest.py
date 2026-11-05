"""Shared owned native-wire fixtures; no Core import or vendor calls."""

from completion_peer import native_peer as native_peer

import pytest


@pytest.fixture(autouse=True)
def owned_google_mode_environment(monkeypatch):
    """Each native factory test chooses its own route, independent of the shell."""
    for variable in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_GENAI_USE_ENTERPRISE"):
        monkeypatch.delenv(variable, raising=False)
