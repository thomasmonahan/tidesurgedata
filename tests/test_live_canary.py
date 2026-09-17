"""Canary for the nightly live job.

It always runs in that job, so the job's failure notification can be exercised on demand:
``gh workflow run nightly-live.yml -f force_failure=true``.
"""

import os

import pytest


@pytest.mark.live
def test_nightly_canary():
    if os.environ.get("TSD_FORCE_LIVE_FAILURE") == "1":
        pytest.fail("Forced failure (force_failure=true) to exercise nightly failure notification.")
