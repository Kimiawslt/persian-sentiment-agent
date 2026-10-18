import tempfile
import unittest
from collections import Counter
from pathlib import Path

from generate_dataset import generate_rows
from train_model import NaiveBayesSentimentModel, load_dataset, stratified_split


class TrainingPipelineTests(unittest.TestCase):
    def test_dataset_and_split(self):
        rows = load_dataset(Path("data/reviews.csv"))
        self.assertEqual(len(rows), 1000)
        self.assertEqual(len({text for text, _ in rows}), 1000)
        self.assertEqual(Counter(label for _, label in rows), {"مثبت": 334, "منفی": 333, "خنثی": 333})
        train, test = stratified_split(rows)
        self.assertEqual(len(rows), len(train) + len(test))
        self.assertEqual({label for _, label in test}, {"مثبت", "منفی", "خنثی"})

    def test_model_learns_and_round_trips(self):
        rows = load_dataset(Path("data/reviews.csv"))
        model = NaiveBayesSentimentModel.train(rows)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save(path)
            restored = NaiveBayesSentimentModel.load(path)
            self.assertEqual(model.predict("محصول عالی بود"), restored.predict("محصول عالی بود"))

    def test_generator_is_reproducible(self):
        self.assertEqual(generate_rows(), generate_rows())


if __name__ == "__main__":
    unittest.main()
