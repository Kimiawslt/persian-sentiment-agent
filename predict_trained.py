"""Use the trained sentiment model from the command line."""

import argparse
import json
from pathlib import Path

from train_model import NaiveBayesSentimentModel


def main() -> None:
    parser = argparse.ArgumentParser(description="پیش بینی احساس با مدل آموزش دیده")
    parser.add_argument("text", nargs="+", help="متن نظر فارسی")
    parser.add_argument("--model", type=Path, default=Path("models/sentiment_model.json"))
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit("Model not found. First run: python3 train_model.py")

    model = NaiveBayesSentimentModel.load(args.model)
    text = " ".join(args.text)
    label, confidence = model.predict(text)
    print(json.dumps({"text": text, "label": label, "confidence": confidence}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
