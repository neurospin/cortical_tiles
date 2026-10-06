"""Unit tests for cortical_tiles.config (TASK-213).

Config.champollion_data_root_dir is computed once, when the module is
imported, from the CHAMPOLLION_DATA_ROOT environment variable. Each test
therefore sets up the environment, then reloads the module, then reads the
value. The tests are hermetic: nothing under /neurospin is accessed.
"""

import importlib
import os

import pytest

import cortical_tiles.config as ct_config

ENV_VAR = "CHAMPOLLION_DATA_ROOT"
EXPECTED_DEFAULT = "/neurospin/dico/data/deep_folding/current"


@pytest.fixture
def isolated_env():
    """Save CHAMPOLLION_DATA_ROOT, then restore it and reload the module.

    Restoring and reloading happen in this fixture's own teardown (not via
    monkeypatch) so the module is reloaded only after the variable is back
    to its original state, leaving no reloaded value behind for other tests.
    """
    saved = os.environ.get(ENV_VAR)
    yield
    if saved is None:
        os.environ.pop(ENV_VAR, None)
    else:
        os.environ[ENV_VAR] = saved
    importlib.reload(ct_config)


def test_data_root_defaults_to_deep_folding_current_when_env_unset(
        isolated_env):
    """REQ-CTDATAROOT-01: with the variable unset at import, the default is
    the existing deep_folding data root."""
    os.environ.pop(ENV_VAR, None)
    module = importlib.reload(ct_config)

    assert module.Config().get_champollion_data_root_dir() == \
        EXPECTED_DEFAULT, (
            "REQ-CTDATAROOT-01: default data root must be "
            f"{EXPECTED_DEFAULT!r}")


def test_data_root_honours_env_override_set_before_import(
        isolated_env, tmp_path):
    """REQ-CTDATAROOT-02: with the variable set at import, its value is
    returned unchanged."""
    override = str(tmp_path / "champollion_data_root")
    os.environ[ENV_VAR] = override
    module = importlib.reload(ct_config)

    assert module.Config().get_champollion_data_root_dir() == override
