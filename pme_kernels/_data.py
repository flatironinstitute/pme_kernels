"""Location of the kernel data (the data/ directory of the repository)."""
import json
import os
from functools import lru_cache
from pathlib import Path


def data_dir():
    """The data directory: $PME_KERNELS_DATA if set, otherwise data/ next to the package."""
    path = Path(os.environ.get('PME_KERNELS_DATA', Path(__file__).resolve().parents[1]/'data'))
    if not path.is_dir():
        raise FileNotFoundError(f'kernel data not found at {path}; set PME_KERNELS_DATA')
    return path


@lru_cache(maxsize=None)
def load(name):
    """A JSON file of the data directory, such as 'si/si_kernels.json'."""
    return json.loads((data_dir()/name).read_text())


def tolerance_key(eps):
    """The tolerances of the data are powers of ten; match them to a relative accuracy of 1e-9."""
    return lambda entry: abs(entry['tolerance']/eps-1) < 1e-9
