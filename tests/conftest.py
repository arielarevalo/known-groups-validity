"""Shared test fixtures. The test-data dir is resolved from pytest's rootdir
(config-anchored), not from ``__file__`` — consistent with the project's
no-location-dependent-path-resolution rule."""
import pathlib

import pytest


@pytest.fixture(scope="session")
def fixtures_dir(pytestconfig) -> pathlib.Path:
    """Committed, free-standing test data (a small copy of pilot outputs)."""
    return pathlib.Path(pytestconfig.rootpath) / "tests" / "fixtures"


@pytest.fixture(scope="session")
def repo_root(pytestconfig) -> pathlib.Path:
    """Repo root, anchored on pytest's config dir (not ``__file__``), so it is
    independent of how deep under ``tests/`` a test file lives."""
    return pathlib.Path(pytestconfig.rootpath)
