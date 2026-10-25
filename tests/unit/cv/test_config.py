"""Run selection is explicit (operations 0015): config.use_run() ahead of the
CV_RUN env var ahead of the profile's `run` field, and run_id() raises when none
of them is set rather than guessing a default."""
import pytest

from cv import config


def test_use_run_takes_precedence_over_env(monkeypatch):
    monkeypatch.setenv("CV_RUN", "from-env")
    try:
        assert config.run_id() == "from-env"      # env wins when there is no override
        config.use_run("from-api")
        assert config.run_id() == "from-api"       # override wins over env
        config.use_run("a/b")                      # slashes sanitised to a dir-safe id
        assert config.run_id() == "a_b"
    finally:
        config.use_run("")                         # clear the override
    assert config.run_id() == "from-env"


def test_run_id_raises_when_nothing_is_set(monkeypatch):
    monkeypatch.delenv("CV_RUN", raising=False)
    config.use_run("")                             # no override, no env, no profile `run`
    with pytest.raises(config.ConfigError):
        config.run_id()
