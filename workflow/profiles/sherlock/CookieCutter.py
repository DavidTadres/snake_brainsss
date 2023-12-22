#
# Based on lsf CookieCutter.py
#
import os
import json
import pathlib
scripts_path = pathlib.Path(__file__).parent.resolve()  # should be /.snake_brainsss/workflow/profiles

d = os.path.dirname(__file__)
with open(os.path.join(d, "settings.json")) as fh:
    settings = json.load(fh)


def from_entry_or_env(values, key):
    """Return value from ``values`` and override with environment variables."""
    if key in os.environ:
        return os.environ[key]
    else:
        return values[key]


class CookieCutter:

    SBATCH_DEFAULTS = from_entry_or_env(settings, "SBATCH_DEFAULTS")
    CLUSTER_NAME = from_entry_or_env(settings, "CLUSTER_NAME")
    #print("CLUSTER_CONFIG path" + str(pathlib.Path(scripts_path, 'cluster_config.yml')))
    #CLUSTER_CONFIG = from_entry_or_env(settings, "CLUSTER_CONFIG") #pathlib.Path(scripts_path, 'cluster_config.yml')
    print('CLUSTER CONFIG PATH ' + repr(str(scripts_path) + 'cluster_config.yml'))
    CLUSTER_CONFIG = str(scripts_path) + 'cluster_config.yml'
    @staticmethod
    def get_cluster_option() -> str:
        cluster = CookieCutter.CLUSTER_NAME
        if cluster != "":
            return f"--cluster={cluster}"
        return ""

    @staticmethod
    def get_cluster_logpath() -> str:
        return "logs/slurm/%r/%j"

    @staticmethod
    def get_cluster_jobname() -> str:
        return "%r_%w"
