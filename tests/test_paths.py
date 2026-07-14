from pathlib import Path


def test_project_paths_exist():
    root = Path(__file__).resolve().parents[1]
    assert (root / "app.py").exists()
    assert (root / "requirements.txt").exists()
    assert (root / "data" / "airline-passengers.csv").exists()
    assert (root / "models").exists()
