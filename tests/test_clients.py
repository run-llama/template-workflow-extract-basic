"""Regression tests for ``extraction_review.clients``."""

import os
import subprocess
import sys


def test_clients_import_does_not_load_fake_when_disabled():
    """Without FAKE_LLAMA_CLOUD set, importing clients must not pull in
    llama_cloud_fake. The fake server is a dev-only dependency and may not be
    installed in production environments.
    """
    src = (
        "import sys\n"
        "import extraction_review.clients\n"
        "assert 'llama_cloud_fake' not in sys.modules, "
        "'llama_cloud_fake was imported without FAKE_LLAMA_CLOUD'\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "FAKE_LLAMA_CLOUD"}
    subprocess.run([sys.executable, "-c", src], check=True, env=env)
