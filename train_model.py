"""Train a small Persian sentiment model using Multinomial Naive Bayes."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from sentiment_agent import normalize_persian


ALLOWED_LABELS = {"مثبت", "منفی", "خنثی"}


def tokenize(text: str) -> list[str]:
    """Turn normalized text into word and character features."""
    normalized = normalize_persian(text)
    words = normalized.split()
    features = [f"word:{word}" for word in words]
    compact = re.sub(r"\s+", "_", normalized)
    features.extend(
        f"char:{compact[index:index + 3]}"
        for index in range(max(0, len(compact) - 2))
    )
    return features


def load_dataset(path: Path) -> list[tuple[str, str]]:
    """Read and validate a UTF-8 CSV containing text,label columns."""
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["text", "label"]:
            raise ValueError("CSV must have exactly these columns: text,label")
        for line_number, row in enumerate(reader, start=2):
            text = (row["text"] or "").strip()
            label = (row["label"] or "").strip()
            if not text:
                raise ValueError(f"Empty text at line {line_number}")
            if label not in ALLOWED_LABELS:
                raise ValueError(f"Invalid label at line {line_number}: {label}")
            rows.append((text, label))
    if len(rows) < 9:
        raise ValueError("At least 9 labeled examples are required")
    return rows


def stratified_split(
    rows: list[tuple[str, str]], test_ratio: float = 0.2, seed: int = 42
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Put examples from every label in both train and test sets."""
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row[1]].append(row)

    train: list[tuple[str, str]] = []
    test: list[tuple[str, str]] = []
    randomizer = random.Random(seed)
    for label in sorted(ALLOWED_LABELS):
        examples = groups[label]
        if len(examples) < 2:
            raise ValueError(f"Label {label!r} needs at least 2 examples")
        randomizer.shuffle(examples)
        test_count = max(1, round(len(examples) * test_ratio))
        test.extend(examples[:test_count])
        train.extend(examples[test_count:])
    randomizer.shuffle(train)
    randomizer.shuffle(test)
    return train, test


@dataclass
class NaiveBayesSentimentModel:
    labels: list[str]
    log_priors: dict[str, float]
    token_counts: dict[str, dict[str, int]]
    total_tokens: dict[str, int]
    vocabulary: list[str]
    alpha: float = 1.0

    @classmethod
    def train(
        cls, rows: list[tuple[str, str]], alpha: float = 1.0
    ) -> "NaiveBayesSentimentModel":
        """Learn label and feature frequencies from labeled examples."""
        document_counts: Counter[str] = Counter()
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        vocabulary: set[str] = set()
        for text, label in rows:
            features = tokenize(text)
            document_counts[label] += 1
            counts[label].update(features)
            vocabulary.update(features)

        labels = sorted(document_counts)
        total_documents = len(rows)
        return cls(
            labels=labels,
            log_priors={
                label: math.log(document_counts[label] / total_documents)
                for label in labels
            },
            token_counts={label: dict(counts[label]) for label in labels},
            total_tokens={label: sum(counts[label].values()) for label in labels},
            vocabulary=sorted(vocabulary),
            alpha=alpha,
        )

    def predict_scores(self, text: str) -> dict[str, float]:
        """Return normalized probabilities for every sentiment label."""
        features = tokenize(text)
        vocabulary_size = max(1, len(self.vocabulary))
        log_scores: dict[str, float] = {}
        for label in self.labels:
            denominator = self.total_tokens[label] + self.alpha * vocabulary_size
            score = self.log_priors[label]
            label_counts = self.token_counts[label]
            for feature in features:
                probability = (label_counts.get(feature, 0) + self.alpha) / denominator
                score += math.log(probability)
            log_scores[label] = score

        highest = max(log_scores.values())
        exponentials = {label: math.exp(score - highest) for label, score in log_scores.items()}
        total = sum(exponentials.values())
        return {label: value / total for label, value in exponentials.items()}

    def predict(self, text: str) -> tuple[str, float]:
        scores = self.predict_scores(text)
        label = max(scores, key=scores.get)
        return label, round(scores[label], 4)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "NaiveBayesSentimentModel":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


def evaluate(
    model: NaiveBayesSentimentModel, rows: list[tuple[str, str]]
) -> dict:
    """Calculate accuracy and a confusion matrix on unseen examples."""
    correct = 0
    matrix = {actual: {predicted: 0 for predicted in model.labels} for actual in model.labels}
    mistakes: list[dict[str, str]] = []
    for text, actual in rows:
        predicted, _ = model.predict(text)
        matrix[actual][predicted] += 1
        if predicted == actual:
            correct += 1
        else:
            mistakes.append({"text": text, "actual": actual, "predicted": predicted})
    return {
        "examples": len(rows),
        "accuracy": round(correct / len(rows), 4),
        "confusion_matrix": matrix,
        "mistakes": mistakes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="آموزش مدل تحلیل احساس فارسی")
    parser.add_argument("--data", type=Path, default=Path("data/reviews.csv"))
    parser.add_argument("--model", type=Path, default=Path("models/sentiment_model.json"))
    parser.add_argument("--report", type=Path, default=Path("models/evaluation.json"))
    args = parser.parse_args()

    rows = load_dataset(args.data)
    train_rows, test_rows = stratified_split(rows)
    model = NaiveBayesSentimentModel.train(train_rows)
    report = evaluate(model, test_rows)
    model.save(args.model)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Training examples: {len(train_rows)}")
    print(f"Test examples: {len(test_rows)}")
    print(f"Accuracy: {report['accuracy']:.1%}")
    print(f"Model saved to: {args.model}")
    print(f"Evaluation saved to: {args.report}")


if __name__ == "__main__":
    main()
