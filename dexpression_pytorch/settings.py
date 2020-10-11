from pathlib import Path

ROOT = Path().resolve().parent
DATA_FOLDER = ROOT / "data"
RESULTS_FOLDER = ROOT / "results"


def data(*paths):
    """Return path in the 'data' folder."""
    return DATA_FOLDER.joinpath(*paths)


def results(*paths):
    """Return path in the 'results' folder."""
    return RESULTS_FOLDER.joinpath(*paths)
