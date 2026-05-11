"""
inference.py
============
Run the fine-tuned fake-review detector on new review text.

Usage:
    # Interactive mode (prompts for input):
    python inference.py

    # Single review from command line:
    python inference.py --review "This product is absolutely amazing!!!"

    # Batch mode from a text file (one review per line):
    python inference.py --file my_reviews.txt
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─────────────────────────────────────────
# CONFIG  — must match fine_tune.py paths
# ─────────────────────────────────────────
MODEL_DIR = "../results/model"
MAX_LEN   = 128

ID2LABEL  = {0: "✅ REAL", 1: "🚩 FAKE"}

# ─────────────────────────────────────────
# PREDICTOR
# ─────────────────────────────────────────
class ReviewClassifier:
    def __init__(self, model_dir: str = MODEL_DIR):
        print(f"Loading model from {model_dir} …")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()
        self.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print("Model ready.\n")

    def predict(self, text: str) -> dict:
        """Return label and confidence for a single review string."""
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits

        probs      = torch.softmax(logits, dim=-1).squeeze()
        pred_id    = int(torch.argmax(probs))
        confidence = float(probs[pred_id]) * 100

        return {
            "label":      ID2LABEL[pred_id],
            "confidence": confidence,
            "prob_real":  float(probs[0]) * 100,
            "prob_fake":  float(probs[1]) * 100,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]


# ─────────────────────────────────────────
# PRETTY PRINTER
# ─────────────────────────────────────────
def print_result(review: str, result: dict, idx: int | None = None):
    header = f"Review #{idx}" if idx is not None else "Review"
    bar_fake = int(result["prob_fake"] / 5)   # scale to 20 chars
    bar_real = 20 - bar_fake

    print(f"\n{'─'*55}")
    print(f"  {header}: {review[:80]}{'…' if len(review) > 80 else ''}")
    print(f"{'─'*55}")
    print(f"  Prediction : {result['label']}  ({result['confidence']:.1f}% confident)")
    print(f"  Real  [{('█' * bar_real):<20}] {result['prob_real']:5.1f}%")
    print(f"  Fake  [{('█' * bar_fake):<20}] {result['prob_fake']:5.1f}%")


# ─────────────────────────────────────────
# DEMO REVIEWS  (shown in interactive mode)
# ─────────────────────────────────────────
DEMO_REVIEWS = [
    # Should predict FAKE
    "BEST RESTAURANT ON EARTH!!! Every dish was a miracle!! I cried tears of pure joy!! "
    "The chef is literally a genius!!! Tell EVERYONE you know!!! 5 stars isn't enough!!!",

    # Should predict FAKE
    "This product CHANGED MY LIFE FOREVER!!! I lost 30 pounds overnight!! "
    "My doctor was shocked!!! Everyone I know is now buying this!! MIRACLE!!!",

    # Should predict REAL
    "The pasta was slightly overcooked but the sauce had great depth. "
    "Service was attentive and the wine list was reasonably priced. Would return.",

    # Should predict REAL
    "Decent noise-canceling headphones for the price. The battery lasts about "
    "18 hours and the fit is comfortable. Microphone quality is average for calls.",

    # Ambiguous / tricky
    "Really enjoyed my meal here. The portions were generous and everything "
    "tasted fresh. Definitely one of the best burgers I've had in this city!",
]


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Fake Review Detector — Inference")
    parser.add_argument("--review", type=str,  help="Single review text to classify")
    parser.add_argument("--file",   type=str,  help="Path to text file (one review per line)")
    args = parser.parse_args()

    clf = ReviewClassifier()

    if args.review:
        # ── Single review from CLI arg ─────────────────────
        result = clf.predict(args.review)
        print_result(args.review, result)

    elif args.file:
        # ── Batch from file ────────────────────────────────
        with open(args.file) as f:
            reviews = [line.strip() for line in f if line.strip()]
        results = clf.predict_batch(reviews)
        for i, (rev, res) in enumerate(zip(reviews, results), 1):
            print_result(rev, res, idx=i)

    else:
        # ── Interactive demo ───────────────────────────────
        print("=" * 55)
        print("  Fake Review Detector — Demo Mode")
        print("  Running built-in example reviews …")
        print("=" * 55)

        for i, review in enumerate(DEMO_REVIEWS, 1):
            result = clf.predict(review)
            print_result(review, result, idx=i)

        print(f"\n{'─'*55}")
        print("  Tip: python inference.py --review \"Your review here\"")
        print("       python inference.py --file   my_reviews.txt")
        print(f"{'─'*55}\n")


if __name__ == "__main__":
    main()
