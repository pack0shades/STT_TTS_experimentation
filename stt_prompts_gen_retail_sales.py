"""Generate synthetic STT prompts for a sales/retail voice-bot.

This script outputs *text-only* prompts that plug directly into:
  stt_dataset_gen_melotts.py --prompts <this_output.txt>

Why a separate script?
- You often want to iterate on prompt diversity and distributions without
  regenerating audio.
- You want deterministic generation (seeded), deduplication, and length filters.

Output formats:
- TXT: one utterance per line (default, fastest)
- JSONL: one object per line with {"text": ..., "category": ..., "meta": {...}}

Examples
--------
  /home/pragay/WWAI/.venv/bin/python stt_prompts_gen_retail_sales.py \
    --out data/sales_prompts_generated_en.txt \
    --n 20000 \
    --seed 0 \
    --min-words 4 --max-words 12 \
    --dedupe

  /home/pragay/WWAI/.venv/bin/python stt_prompts_gen_retail_sales.py \
    --out data/sales_prompts_generated_en.jsonl \
    --format jsonl \
    --n 20000 \
    --seed 0 \
    --dedupe

Notes
-----
- These prompts intentionally include some casual speech ("uh", "can you repeat")
  and shorthand ("what's", "I'm") to better match real spoken queries.
- You can tune category ratios and vocab lists via CLI args.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# Defaults (edit or override by CLI)
# -----------------------------

DEFAULT_PRODUCTS = [
    "TV",
    "smart TV",
    "LED TV",
    "QLED TV",
    "OLED TV",
    "4K TV",
    "Android TV",
    "Google TV",
    "speaker",
    "Bluetooth speaker",
    "portable speaker",
    "smart speaker",
    "soundbar",
    "2.1 soundbar",
    "5.1 home theater",
    "home theater",
    "headphones",
    "wireless headphones",
    "wireless earbuds",
    "noise cancelling headphones",
]

DEFAULT_BRANDS = [
    "Sony",
    "Samsung",
    "LG",
    "JBL",
    "Bose",
    "Xiaomi",
    "OnePlus",
    "Panasonic",
    "TCL",
    "Hisense",
    "Philips",
    "Boat",
    "Sennheiser",
    "Marshall",
]

DEFAULT_FEATURES = [
    "Dolby Atmos",
    "Dolby Vision",
    "Bluetooth",
    "WiFi",
    "voice control",
    "HDMI ARC",
    "eARC",
    "screen mirroring",
    "Chromecast",
    "AirPlay",
    "HDR",
    "HDR10+",
    "120Hz",
    "game mode",
    "ALLM",
    "VRR",
    "optical audio",
    "USB playback",
]

DEFAULT_PRICES = [
    "₹15,000",
    "₹20,000",
    "₹25,000",
    "₹30,000",
    "₹35,000",
    "₹40,000",
    "₹45,000",
    "₹50,000",
    "₹55,000",
    "₹60,000",
    "₹70,000",
    "₹75,000",
    "₹80,000",
    "₹90,000",
    "₹1,00,000",
    "₹1,20,000",
]

DEFAULT_SIZES = [
    "32 inch",
    "40 inch",
    "43 inch",
    "50 inch",
    "55 inch",
    "65 inch",
    "75 inch",
    "85 inch",
]

DEFAULT_USE_CASES = [
    "gaming",
    "PS5 gaming",
    "movies",
    "Netflix",
    "sports",
    "music",
    "parties",
    "a small room",
    "a large living room",
    "a bedroom",
    "office use",
]


TEMPLATES: Dict[str, List[str]] = {
    "product": [
        "Do you have {product}?",
        "What {product}s do you have?",
        "Can you show me {product} options?",
        "Do you sell {product}?",
        "What brands do you have for {product}?",
        "Do you have {brand1} {product} in {size}?",
        "Do you have a {product} under {price} for {use_case}?",
        "Show me a {brand1} {product} with {feature}.",
    ],
    "price": [
        "What is the price of this {product}?",
        "How much is this {product}?",
        "Do you have anything under {price} from {brand1}?",
        "What is the cheapest {brand1} {product} you have?",
        "Is there any discount on this {brand1} {product}?",
        "What will be the final price for {brand1} {product} in {size}?",
        "Do you have EMI options for a {product} around {price}?",
    ],
    "comparison": [
        "Which is better {brand1} or {brand2}?",
        "Is the {brand1} {product} better than the {brand2} {product}?",
        "What is the difference between the {brand1} and {brand2} models?",
        "Should I go for {brand1} or {brand2} for {use_case}?",
        "Which one is better for {use_case}?",
        "Between {brand1} and {brand2}, which {product} has better {feature}?",
        "Is {brand1} {product} worth it compared to {brand2} at around {price}?",
    ],
    "features": [
        "Does this support {feature}?",
        "Is this {product} a smart model?",
        "Can I connect my phone to this {product}?",
        "Does it have {feature}?",
        "Is this {product} 4K?",
        "Does this {brand1} {product} support {feature} and WiFi?",
        "Can I use Bluetooth on this {brand1} {product}?",
        "Does the {brand1} {product} have HDMI ARC or eARC?",
    ],
    "availability": [
        "Is the {brand1} {product} in stock right now?",
        "Do you have this {product} in {size}?",
        "Is this {brand1} {product} available right now?",
        "Do you have a bigger size than {size} in {brand1}?",
        "When will this {brand1} {product} be available again?",
        "Do you have {brand2} {product} in {size} available today?",
    ],
    "recommendation": [
        "Which {product} would you recommend?",
        "What is the best {product} under {price}?",
        "Suggest a good {product}.",
        "What should I buy under {price} for {use_case}?",
        "Which {product} is most popular right now?",
        "Recommend a {brand1} {product} for {use_case} with {feature}.",
        "What is a good {product} under {price} in {size}?",
    ],
    "usage": [
        "Is this good for {use_case}?",
        "Will this work for {use_case}?",
        "Is this {product} loud enough for {use_case}?",
        "Can I use this for {use_case}?",
        "Is this good for a small room?",
        "Will a {brand1} {product} be good for {use_case} in {size}?",
        "Is this {brand1} {product} good for {use_case} with {feature}?",
    ],
    "purchase": [
        "I want to buy a {brand1} {product}.",
        "Show me something in my budget under {price}.",
        "I am looking for a {product} around {price}.",
        "Help me choose a {product} for {use_case}.",
        "I need a {product} for my home, maybe {size}.",
        "I want a {brand1} {product} with {feature} under {price}.",
        "I am looking for a {size} {brand1} {product} for {use_case}.",
    ],
    "accessories": [
        "Do you provide installation for this {product}?",
        "Is wall mounting included with the {size} {product}?",
        "Do I need extra cables for this {product}?",
        "Can you set up the {brand1} {product} for me?",
        "Is setup free if I buy it today under {price}?",
        "Is the HDMI cable included with the {brand1} {product}?",
        "Do you provide a warranty on the {brand1} {product}?",
    ],
    "conversation": [
        "Okay, show me the {brand1} {product}.",
        "What about the {brand2} {product} instead?",
        "Is the {brand1} one better for {use_case}?",
        "Can you explain the difference between {brand1} and {brand2} again?",
        "Is it worth it at around {price}?",
        "Wait, sorry, can you repeat the price for the {brand1} {product}?",
        "Uh, what was the price again for the {size} model?",
        "Actually, do you have the same {product} in {size}?",
        "Can you show me a {product} under {price} again?",
        "Sorry, does the {brand1} {product} support {feature}?",
        "Can you recommend a {product} for {use_case} under {price}?",
    ],
}


# -----------------------------
# Generation utilities
# -----------------------------

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def _count_words(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _normalize_text(text: str) -> str:
    # Normalize whitespace and common punctuation spacing.
    text = " ".join(str(text).strip().split())
    # Keep this conservative; Whisper labels can keep punctuation.
    return text


def _choice_distinct(rng: random.Random, xs: List[str], other: str) -> str:
    if not xs:
        return other
    if len(xs) == 1:
        return xs[0]
    # try a few times
    for _ in range(6):
        x = rng.choice(xs)
        if x != other:
            return x
    return rng.choice(xs)


def _add_variation(rng: random.Random, text: str) -> str:
    # Keep variations mild so transcripts stay plausible.
    options = [text]

    # Contractions / casual
    options.append(text.replace("Do you ", "Do you guys "))
    options.append(text.replace("What is", "What's"))
    options.append(text.replace("I am", "I'm"))
    options.append(text.replace("Can you", "Can u"))

    # Add light disfluency sometimes
    if rng.random() < 0.15:
        options.append("Uh, " + text[0].lower() + text[1:])
    if rng.random() < 0.10:
        options.append("Sorry, " + text[0].lower() + text[1:])

    out = rng.choice(options)
    return _normalize_text(out)


@dataclass(frozen=True)
class Vocab:
    products: List[str]
    brands: List[str]
    features: List[str]
    prices: List[str]
    sizes: List[str]
    use_cases: List[str]


def _fill_template(rng: random.Random, template: str, v: Vocab) -> str:
    b1 = rng.choice(v.brands) if v.brands else "BrandA"
    b2 = _choice_distinct(rng, v.brands, b1)
    return template.format(
        product=rng.choice(v.products) if v.products else "product",
        brand1=b1,
        brand2=b2,
        feature=rng.choice(v.features) if v.features else "feature",
        price=rng.choice(v.prices) if v.prices else "price",
        size=rng.choice(v.sizes) if v.sizes else "size",
        use_case=rng.choice(v.use_cases) if v.use_cases else "use case",
    )


def _category_counts(n: int, categories: List[str], rng: random.Random, balance: bool) -> List[str]:
    if not categories:
        return []

    if not balance:
        return [rng.choice(categories) for _ in range(n)]

    # balanced: round-robin then shuffle
    base = [categories[i % len(categories)] for i in range(n)]
    rng.shuffle(base)
    return base


def _pick_balanced_category(
    *,
    rng: random.Random,
    active_categories: List[str],
    cat_added: Dict[str, int],
) -> str:
    """Pick a category with the lowest current count (tie-break randomly)."""

    # Find minimum count among active categories
    min_n = None
    for c in active_categories:
        n = int(cat_added.get(c, 0))
        if min_n is None or n < min_n:
            min_n = n

    # Choose randomly among the min bucket
    mins = [c for c in active_categories if int(cat_added.get(c, 0)) == int(min_n or 0)]
    return rng.choice(mins) if mins else rng.choice(active_categories)


def generate_prompts(
    *,
    n: int,
    seed: int,
    vocab: Vocab,
    min_words: int,
    max_words: int,
    dedupe: bool,
    balance_categories: bool,
    categories: Optional[List[str]] = None,
) -> List[Tuple[str, str, Dict[str, Any]]]:
    rng = random.Random(int(seed))

    cats = categories or sorted(TEMPLATES.keys())
    cats = [c for c in cats if c in TEMPLATES]

    out: List[Tuple[str, str, Dict[str, Any]]] = []
    seen = set()

    cat_added: Dict[str, int] = {c: 0 for c in cats}
    cat_stall: Dict[str, int] = {c: 0 for c in cats}
    active_cats: List[str] = list(cats)

    # Safety cap to avoid infinite loops if filters are too strict.
    attempts = 0
    # Dedupe + balancing can cause many retries; allow a higher ceiling.
    max_attempts = int(max(5000, n * 200))

    # If a category fails to produce a new, valid unique sample too many times
    # in a row, temporarily disable it (soft balancing).
    stall_limit = 5000

    while len(out) < int(n) and attempts < max_attempts:
        attempts += 1
        if not active_cats:
            break

        if balance_categories:
            category = _pick_balanced_category(rng=rng, active_categories=active_cats, cat_added=cat_added)
        else:
            category = rng.choice(cats)

        template = rng.choice(TEMPLATES[category])
        raw = _fill_template(rng, template, vocab)
        text = _add_variation(rng, raw)

        wc = _count_words(text)
        if int(min_words) > 0 and wc < int(min_words):
            cat_stall[category] = int(cat_stall.get(category, 0)) + 1
            if balance_categories and cat_stall[category] >= stall_limit and category in active_cats:
                active_cats = [c for c in active_cats if c != category]
            continue
        if int(max_words) > 0 and wc > int(max_words):
            cat_stall[category] = int(cat_stall.get(category, 0)) + 1
            if balance_categories and cat_stall[category] >= stall_limit and category in active_cats:
                active_cats = [c for c in active_cats if c != category]
            continue

        key = text.lower()
        if dedupe and key in seen:
            cat_stall[category] = int(cat_stall.get(category, 0)) + 1
            if balance_categories and cat_stall[category] >= stall_limit and category in active_cats:
                active_cats = [c for c in active_cats if c != category]
            continue

        seen.add(key)
        cat_added[category] = int(cat_added.get(category, 0)) + 1
        cat_stall[category] = 0
        meta = {
            "words": wc,
        }
        out.append((text, category, meta))

    if len(out) < int(n):
        # Provide a helpful diagnostic summary.
        # (We keep it compact; caller can re-run with different flags.)
        summary = ", ".join([f"{c}:{cat_added.get(c, 0)}" for c in sorted(cats)])
        exhausted = [c for c in cats if c not in active_cats] if balance_categories else []
        raise RuntimeError(
            f"Could not generate enough samples after {attempts} attempts. "
            f"Got {len(out)}/{n}. "
            f"Per-category counts: {summary}. "
            + (f"Exhausted categories (soft-balance): {exhausted}. " if exhausted else "")
            + "Try: reduce --min-words, disable --dedupe, or run without --balance."
        )

    return out


def _write_txt(path: Path, rows: Iterable[Tuple[str, str, Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text, _cat, _meta in rows:
            f.write(text + "\n")


def _write_jsonl(path: Path, rows: Iterable[Tuple[str, str, Dict[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text, cat, meta in rows:
            f.write(json.dumps({"text": text, "category": cat, "meta": meta}, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--format", type=str, default="txt", choices=["txt", "jsonl"])
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--min-words", type=int, default=4)
    p.add_argument("--max-words", type=int, default=12)
    p.add_argument("--dedupe", action="store_true")
    p.add_argument("--balance", action="store_true", help="Balance categories instead of fully random")

    # Optional overrides via comma-separated lists
    p.add_argument("--products", type=str, default="")
    p.add_argument("--brands", type=str, default="")
    p.add_argument("--features", type=str, default="")
    p.add_argument("--prices", type=str, default="")
    p.add_argument("--sizes", type=str, default="")
    p.add_argument("--use-cases", type=str, default="")

    args = p.parse_args(argv)

    def split_csv(s: str) -> List[str]:
        s = str(s).strip()
        if not s:
            return []
        return [x.strip() for x in s.split(",") if x.strip()]

    v = Vocab(
        products=split_csv(args.products) or list(DEFAULT_PRODUCTS),
        brands=split_csv(args.brands) or list(DEFAULT_BRANDS),
        features=split_csv(args.features) or list(DEFAULT_FEATURES),
        prices=split_csv(args.prices) or list(DEFAULT_PRICES),
        sizes=split_csv(args.sizes) or list(DEFAULT_SIZES),
        use_cases=split_csv(args.use_cases) or list(DEFAULT_USE_CASES),
    )

    rows = generate_prompts(
        n=int(args.n),
        seed=int(args.seed),
        vocab=v,
        min_words=int(args.min_words),
        max_words=int(args.max_words),
        dedupe=bool(args.dedupe),
        balance_categories=bool(args.balance),
    )

    out_path = Path(args.out)
    if str(args.format).lower() == "jsonl":
        _write_jsonl(out_path, rows)
    else:
        _write_txt(out_path, rows)

    print(f"wrote {len(rows)} prompts -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
