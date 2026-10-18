import unittest

from sentiment_agent import PersianSentimentAgent, normalize_persian


class SentimentAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = PersianSentimentAgent()

    def test_positive(self):
        self.assertEqual(self.agent.predict("کیفیت محصول واقعاً عالی بود").label, "مثبت")

    def test_negative(self):
        self.assertEqual(self.agent.predict("ارسال خیلی دیر بود و کیفیت افتضاحه").label, "منفی")

    def test_neutral(self):
        self.assertEqual(self.agent.predict("بسته امروز ساعت سه رسید").label, "خنثی")

    def test_negation(self):
        self.assertEqual(self.agent.predict("این محصول خوب نیست").label, "منفی")

    def test_arabic_char_normalization(self):
        self.assertEqual(normalize_persian("كيفيت عالي"), "کیفیت عالی")


if __name__ == "__main__":
    unittest.main()
