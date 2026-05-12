# Fake Review Detector

Fine-tuning DistilBERT to classify product and restaurant reviews as real or fake; a defensive approach to the cyber deception tactic of fake review manipulation.

## Files
- `dataset/reviews.csv` — 174 labeled reviews (real/fake), fake ones generated using Claude AI
- `src/fine_tune.py` — fine-tunes the model on the dataset
- `src/inference.py` — run predictions on new reviews
- `src/evaluate.py` — generates accuracy metrics and plots

## How to run
```bash
pip install -r requirements.txt
python src/fine_tune.py
python src/inference.py
python src/evaluate.py
```

## Model
Pretrained `distilbert-base-uncased` from Hugging Face, fine-tuned for binary text classification (real vs. fake).
