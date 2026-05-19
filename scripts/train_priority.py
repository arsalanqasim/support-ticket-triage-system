import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triage.models.train import train_priority


if __name__ == "__main__":
    result = train_priority()
    print(f"Saved {result.target} model to {result.path}")
    print(result.metrics)
