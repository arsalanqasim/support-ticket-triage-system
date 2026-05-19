import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from triage.models.train import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all baseline triage models.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit per dataset.")
    args = parser.parse_args()

    for result in train_all(limit=args.limit):
        print(f"Saved {result.target} model to {result.path}")
        print(f"Rows: {result.rows}")
        print(f"Labels: {', '.join(result.labels)}")
        print(f"Metrics: {result.metrics}")


if __name__ == "__main__":
    main()
