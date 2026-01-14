"""
Pytest configuration for testing_utils tests.

These tests create their own FakeLlamaCloudServer instances and need
the global server from extraction_review.clients to be uninstalled
to avoid route conflicts.
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_from_global_fake():
    """Uninstall the global fake server for testing_utils tests.

    The global server from extraction_review.clients intercepts HTTP
    requests before test-specific servers can handle them. This fixture
    temporarily uninstalls it so tests can use their own isolated instances.
    """
    from extraction_review.clients import fake

    was_installed = fake is not None and fake._installed

    if was_installed:
        fake.uninstall()

    yield

    if was_installed:
        fake.install()
