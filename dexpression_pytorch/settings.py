from pathlib import Path

ROOT = Path().resolve().parent
DATA_FOLDER = ROOT / "data"
RESULTS_FOLDER = ROOT / "results"


def data(*paths):
    data_path = DATA_FOLDER.joinpath(*paths)
    Path(data_path).mkdir(parents=True, exist_ok=True)

    return data_path


def results(*paths):
    results_path = RESULTS_FOLDER.joinpath(*paths)
    Path(results_path).mkdir(parents=True, exist_ok=True)

    return results_path
