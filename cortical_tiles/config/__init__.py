
import os

config_instance = None

# Champollion model / regions data root used when CHAMPOLLION_DATA_ROOT is unset.
_DEFAULT_CHAMPOLLION_DATA_ROOT = "/neurospin/dico/data/deep_folding/current"


class Config:

    champollion_data_root_dir = os.environ.get(
        "CHAMPOLLION_DATA_ROOT",
        _DEFAULT_CHAMPOLLION_DATA_ROOT,
    )

    def get_champollion_data_root_dir(self):
        """ get directory of model / regions data.

        The directory should contain mask/2mm/regions/meshes/. It is read from
        the CHAMPOLLION_DATA_ROOT environment variable when that is set at
        import time, otherwise it defaults to
        /neurospin/dico/data/deep_folding/current.
        """
        return self.champollion_data_root_dir


def config():
    """ get the global unique instance of the Config
    """
    global config_instance
    if config_instance is None:
        config_instance = Config()
    return config_instance
