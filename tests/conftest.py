"""
Pytest configuration for extract-basic tests.

IMPORTANT: FAKE_LLAMA_CLOUD must be set before any test modules are imported,
as extraction_review.clients reads this at module load time to initialize
the mock server.
"""

import os

# Enable the fake LlamaCloud server for all tests
os.environ["FAKE_LLAMA_CLOUD"] = "true"
