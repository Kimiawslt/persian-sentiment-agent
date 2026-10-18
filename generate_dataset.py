"""Generate a reproducible synthetic Persian sentiment training dataset.

The generated comments are educational examples, not real customer data.
"""

from __future__ import annotations

import csv
import random
from itertools import product
from pathlib import Path


OUTPUT = Path("data/reviews.csv")
COUNTS = {"مثبت": 334, "منفی": 333, "خنثی": 333}
SEED = 1403


def positive_candidates() -> set[str]:
    subjects = [
        "کیفیت محصول", "بسته بندی", "سرعت ارسال", "پشتیبانی", "قیمت کالا",
        "طراحی محصول", "عملکرد دستگاه", "جنس کالا", "تجربه خرید", "خدمات فروشگاه",
        "نحوه پاسخگویی", "سلامت بسته",
    ]
    qualities = [
        "عالی", "خیلی خوب", "فوق العاده", "رضایت بخش", "مناسب", "حرفه ای",
        "با کیفیت", "ارزشمند", "تمیز", "سریع", "قابل قبول", "بی نظیر",
    ]
    endings = [
        "کاملا راضی هستم", "دوباره خرید می کنم", "به دیگران پیشنهادش می کنم",
        "ارزش خرید داشت", "به موقع به دستم رسید", "انتظارم را برآورده کرد",
        "از انتخابم خوشحالم", "هیچ ایرادی نداشت",
    ]
    templates = [
        "{subject} {quality} بود و {ending}",
        "از {subject} راضی بودم چون {quality} بود و {ending}",
        "{subject} واقعا {quality} است؛ {ending}",
        "به نظرم {subject} {quality} بود، {ending}",
    ]
    return {
        template.format(subject=subject, quality=quality, ending=ending)
        for subject, quality, ending, template in product(subjects, qualities, endings, templates)
    }


def negative_candidates() -> set[str]:
    subjects = [
        "کیفیت محصول", "بسته بندی", "سرعت ارسال", "پشتیبانی", "قیمت کالا",
        "طراحی محصول", "عملکرد دستگاه", "جنس کالا", "تجربه خرید", "خدمات فروشگاه",
        "نحوه پاسخگویی", "سلامت بسته",
    ]
    qualities = [
        "افتضاح", "خیلی بد", "ضعیف", "ناامید کننده", "نامناسب", "غیر حرفه ای",
        "بی کیفیت", "بی ارزش", "کثیف", "کند", "خراب", "پر از ایراد",
    ]
    endings = [
        "کاملا ناراضی هستم", "دیگر خرید نمی کنم", "به هیچ کس پیشنهادش نمی کنم",
        "ارزش خرید نداشت", "خیلی دیر به دستم رسید", "انتظارم را برآورده نکرد",
        "از انتخابم پشیمانم", "مشکلات زیادی داشت",
    ]
    templates = [
        "{subject} {quality} بود و {ending}",
        "از {subject} ناراضی بودم چون {quality} بود و {ending}",
        "{subject} واقعا {quality} است؛ {ending}",
        "به نظرم {subject} {quality} بود، {ending}",
    ]
    return {
        template.format(subject=subject, quality=quality, ending=ending)
        for subject, quality, ending, template in product(subjects, qualities, endings, templates)
    }


def neutral_candidates() -> set[str]:
    subjects = [
        "محصول", "سفارش", "بسته", "کالا", "فاکتور", "دفترچه راهنما",
        "شماره پیگیری", "رسید خرید", "جعبه", "مدل انتخاب شده", "رنگ کالا", "ضمانت نامه",
    ]
    events = [
        "ثبت شد", "ارسال شد", "تحویل داده شد", "در انبار قرار گرفت",
        "داخل جعبه بود", "در سایت نمایش داده شد", "در فاکتور نوشته شد",
        "از فروشگاه دریافت شد", "توسط پست جابه جا شد", "به سفارش اضافه شد",
    ]
    times = [
        "امروز", "دیروز", "صبح دوشنبه", "ساعت سه", "این هفته", "روز تحویل",
        "پس از ثبت سفارش", "هنگام دریافت",
    ]
    details = [
        "شماره سفارش هم درج شده بود", "رنگ آن سفید بود", "وزن آن یک کیلو بود",
        "دو عدد از آن وجود داشت", "تاریخ روی برچسب نوشته شده بود",
        "مشخصات در رسید ثبت شده بود", "یک نسخه از فاکتور همراهش بود",
        "کد کالا روی بسته قرار داشت",
    ]
    templates = [
        "{subject} {time} {event} و {detail}",
        "{time} {subject} {event}؛ {detail}",
        "طبق اطلاعات سفارش {subject} {event} و {detail}",
        "در گزارش خرید آمده که {subject} {time} {event} و {detail}",
    ]
    return {
        template.format(subject=subject, event=event, time=time, detail=detail)
        for subject, event, time, detail, template in product(subjects, events, times, details, templates)
    }


def generate_rows(seed: int = SEED) -> list[tuple[str, str]]:
    """Return exactly 1,000 unique and approximately balanced examples."""
    randomizer = random.Random(seed)
    sources = {
        "مثبت": sorted(positive_candidates()),
        "منفی": sorted(negative_candidates()),
        "خنثی": sorted(neutral_candidates()),
    }
    rows: list[tuple[str, str]] = []
    for label, count in COUNTS.items():
        candidates = sources[label]
        if len(candidates) < count:
            raise RuntimeError(f"Not enough unique candidates for {label}")
        rows.extend((text, label) for text in randomizer.sample(candidates, count))
    randomizer.shuffle(rows)
    return rows


def write_dataset(path: Path = OUTPUT) -> None:
    rows = generate_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic examples to {path}")


if __name__ == "__main__":
    write_dataset()
