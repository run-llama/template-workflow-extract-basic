"""Tests for the FakeLlamaCloudServer lifecycle and configuration."""

from extraction_review.testing_utils import FakeLlamaCloudServer


def test_context_manager_installs_and_uninstalls():
    """Verify context manager properly installs and uninstalls the mock."""
    server = FakeLlamaCloudServer()
    assert not server._installed

    with server:
        assert server._installed

    assert not server._installed


def test_install_is_idempotent():
    """Verify calling install multiple times is safe."""
    server = FakeLlamaCloudServer()
    server.install()
    server.install()  # Should not raise
    assert server._installed
    server.uninstall()
    assert not server._installed


def test_selective_namespace_registration():
    """Verify only requested namespaces are registered."""
    with FakeLlamaCloudServer(namespaces=["files"]) as server:
        assert "files" in server._namespace_names
        assert "parse" not in server._namespace_names

    with FakeLlamaCloudServer(namespaces=["parse", "extract"]) as server:
        assert "parse" in server._namespace_names
        assert "extract" in server._namespace_names
        assert "files" not in server._namespace_names
