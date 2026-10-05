import subprocess
import sys

import pytest

from photo_edit_studio.diagnostics import memory_snapshot
from photo_edit_studio.ui import _stream_run


def test_logging_captures_exceptions_and_shutdown(tmp_path):
    code = """
import logging
import sys
from pathlib import Path
from photo_edit_studio.diagnostics import configure_logging
configure_logging(Path(sys.argv[1]))
logger = logging.getLogger('photo_edit_studio.test')
logger.info('load-stage-start')
raise RuntimeError('synthetic-load-failure')
"""
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)],
                            capture_output=True, check=False)
    assert result.returncode != 0
    text = (tmp_path / "studio.log").read_text(encoding="utf-8")
    assert "load-stage-start" in text
    assert "Uncaught main-thread exception" in text
    assert "synthetic-load-failure" in text
    assert "Python interpreter shutdown" in text
    assert "pid=" in text and "rss_mib=" in text
    assert list(tmp_path.glob("fault-*.log"))


def test_worker_system_exit_is_forwarded_instead_of_hanging(caplog):
    def failure(progress):
        raise SystemExit(3)

    stream = _stream_run(failure)
    next(stream)
    assert "Failed:" in next(stream)[3]
    with pytest.raises(RuntimeError, match="SystemExit"):
        next(stream)
    assert "Generation worker failed" in caplog.text


def test_memory_snapshot_does_not_require_cuda():
    assert "rss_mib=" in memory_snapshot()