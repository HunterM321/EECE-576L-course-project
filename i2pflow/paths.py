"""Project paths and access to the vendored CoFiI2P code."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = PROJECT_ROOT / "third_party"
COFII2P_ROOT = THIRD_PARTY / "CoFiI2P"


def ensure_cofii2p_on_path():
    """Make CoFiI2P's top-level packages (`data`, `model`) importable.

    CoFiI2P uses generic package names, so it is appended (not prepended) to sys.path
    and should only be imported through the helpers in this package.
    """
    root = str(COFII2P_ROOT)
    if root not in sys.path:
        sys.path.append(root)


def resolve(path):
    """Resolve a config path relative to the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_ROOT / p
