"""A lightweight Persian sentiment classifier.

The first version intentionally has no third-party dependency.  It is useful as a
baseline and as the fallback layer of a future trained model.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Iterable


POSITIVE_WORDS = {
    "عالی": 2.2, "فوق العاده": 2.5, "خوب": 1.3, "محشر": 2.2,
    "راضی": 1.8, "رضایت": 1.6, "دوست داشتم": 1.8, "دوست دارم": 1.8,
    "با کیفیت": 1.7, "باکیفیت": 1.7, "سریع": 1.0, "تمیز": 1.0,
    "مناسب": 1.0, "ارزشمند": 1.4, "خوشم اومد": 1.8, "خوشم آمد": 1.8,
    "پیشنهاد می کنم": 1.8, "پیشنهاد میکنم": 1.8, "ممنون": 1.0,
    "سپاس": 1.0, "بی نظیر": 2.2, "عاشق": 1.8, "کاربردی": 1.2,
}

NEGATIVE_WORDS = {
    "افتضاح": 2.5, "بد": 1.4, "ضعیف": 1.7, "ناراضی": 2.0,
    "بی کیفیت": 2.0, "بیکیفیت": 2.0, "کند": 1.2, "خراب": 2.0,
    "گرون": 1.1, "گران": 1.1, "نامناسب": 1.4, "بی ارزش": 1.7,
    "پشیمان": 2.0, "پشیمون": 2.0, "متنفر": 2.2, "مشکل": 1.1,
    "ایراد": 1.1, "تاخیر": 1.2, "دیر": 1.0, "اصلا پیشنهاد نمی کنم": 2.4,
    "اصلا پیشنهاد نمیکنم": 2.4, "هدر رفت": 1.8, "کلاهبرداری": 2.5,
}

NEGATIONS = {"نه", "نیست", "نبود", "نشد", "نمیشه", "نمی شود", "نمی‌شه", "هرگز"}
INTENSIFIERS = {"خیلی": 1.45, "واقعا": 1.35, "واقعاً": 1.35, "بسیار": 1.4, "اصلا": 1.35, "اصلاً": 1.35}


@dataclass(frozen=True)
class SentimentResult:
    label: str
    confidence: float
    score: float
    positive_score: float
    negative_score: float
    evidence: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_persian(text: str) -> str:
    """Normalize common Arabic/Persian character and spacing variants."""
    translations = str.maketrans({
        "ي": "ی", "ى": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه",
        "ؤ": "و", "إ": "ا", "أ": "ا", "ٱ": "ا",
    })
    text = text.translate(translations).lower().replace("\u200c", " ")
    text = re.sub(r"[ـًٌٍَُِّْ]", "", text)
    text = re.sub(r"[^\w\s!?؟]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


class PersianSentimentAgent:
    labels = {"positive": "مثبت", "negative": "منفی", "neutral": "خنثی"}

    def predict(self, text: str) -> SentimentResult:
        if not isinstance(text, str) or not text.strip():
            return SentimentResult("خنثی", 1.0, 0.0, 0.0, 0.0, [])

        normalized = normalize_persian(text)
        tokens = normalized.split()
        positive = 0.0
        negative = 0.0
        evidence: list[str] = []

        # Long phrases win over their individual words.
        occupied: set[int] = set()
        phrases = sorted(
            [(p, w, 1) for p, w in POSITIVE_WORDS.items()] +
            [(p, w, -1) for p, w in NEGATIVE_WORDS.items()],
            key=lambda item: len(item[0].split()), reverse=True,
        )
        for phrase, weight, polarity in phrases:
            parts = normalize_persian(phrase).split()
            for start in range(len(tokens) - len(parts) + 1):
                indexes = set(range(start, start + len(parts)))
                if indexes & occupied or tokens[start:start + len(parts)] != parts:
                    continue

                before = tokens[max(0, start - 2):start]
                after = tokens[start + len(parts):start + len(parts) + 2]
                context = before + after
                multiplier = max((INTENSIFIERS.get(word, 1.0) for word in before), default=1.0)
                # Persian negation can precede or follow sentiment: «هرگز خوب» / «خوب نیست».
                negated = any(word in NEGATIONS or word.startswith("نمی") for word in context)
                value = weight * multiplier
                effective_polarity = -polarity if negated else polarity
                if effective_polarity > 0:
                    positive += value
                else:
                    negative += value
                evidence.append(("↔ " if negated else "") + phrase)
                occupied.update(indexes)
                break

        score = positive - negative
        # A small dead zone avoids pretending that weak/contradictory evidence is decisive.
        if abs(score) < 0.55:
            label = "خنثی"
        elif score > 0:
            label = "مثبت"
        else:
            label = "منفی"

        strength = abs(score) / (positive + negative + 1.0)
        confidence = 0.5 + 0.48 * (1.0 - math.exp(-2.2 * strength))
        if not evidence:
            confidence = 0.62
        elif label == "خنثی":
            confidence = max(0.55, 1.0 - min(positive, negative) / (positive + negative + 0.1))

        return SentimentResult(
            label=label,
            confidence=round(min(confidence, 0.98), 3),
            score=round(score, 3),
            positive_score=round(positive, 3),
            negative_score=round(negative, 3),
            evidence=evidence,
        )

    def predict_many(self, texts: Iterable[str]) -> list[SentimentResult]:
        return [self.predict(text) for text in texts]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="تشخیص احساس متن فارسی")
    parser.add_argument("text", nargs="*", help="متن فارسی؛ در صورت خالی بودن از ورودی خوانده می‌شود")
    args = parser.parse_args()
    agent = PersianSentimentAgent()
    text = " ".join(args.text) if args.text else input("متن نظر: ")
    print(json.dumps(agent.predict(text).to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
