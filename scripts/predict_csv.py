import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from triage.inference.predictor import TriagePredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict triage labels for a CSV file.")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    predictor = TriagePredictor()
    predictions = predictor.predict_batch(df)
    output = df.copy()
    output["triage_prediction"] = [json.dumps(item) for item in predictions]
    output.to_csv(args.output_csv, index=False)
    print(f"Wrote predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
