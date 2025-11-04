# English Toxic Comment Classifier (English‑only) — model_training.py

Purpose
- Train a high‑precision binary classifier that labels English comments as toxic or non‑toxic for content moderation use cases.  
- Single‑file training script with imbalance handling, evaluation, threshold tuning, and artifact export.

What this script does
- Loads English Jigsaw‑style data (comment_text, toxic).  
- Creates stratified train/dev/test splits for honest evaluation.  
- Tokenizes text with a transformer tokenizer (DistilBERT, uncased).  
- Handles class imbalance via class weights and a WeightedRandomSampler.  
- Trains a transformer classifier with mixed precision when CUDA is available.  
- Evaluates during training (step‑based or epoch‑based depending on config), uses early stopping, and saves the best model.  
- Tunes a decision threshold on the dev set to target high precision with competitive recall.  
- Exports the trained model, tuned threshold, logs, and plots.

Representative results (precision‑leaning operating point)
- Accuracy ≈ 0.956  
- Precision ≈ 0.853  
- Recall ≈ 0.620  
- F1 ≈ 0.718  
- AUC ≈ 0.972

Quick start
1) Setup
```sh
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

2) Data
- Expected CSV columns:
  - comment_text: the input text  
  - toxic: 0/1 target label  
- Optional: unintended‑bias CSV for extra English examples (script caps size for runtime).

3) Run training
```sh
python model_training.py
```
- Artifacts saved under outputs_.../:
  - final_model/ (config.json, weights, tokenizer files)  
  - threshold.json (tuned decision threshold)  
  - final_metrics.json, trainer_state.json, metrics_log.csv  
  - train_loss.png, eval_loss.png, eval_metrics.png

Configuration
- Edit constants at the top of model_training.py:
  - model_name, max_len, epochs, learning rate, batch sizes  
  - eval mode (step‑based vs epoch‑based) and early stopping patience  
  - paths to Jigsaw‑style CSVs  
- Threshold tuning objective:
  - Precision‑leaning: prioritize fewer false positives  
  - Balanced: equalize precision and recall

Notes
- CUDA is used automatically if available (mixed precision enabled).  
- All splits are stratified; thresholds are tuned on the dev set and reported on the held‑out test set.  
- For faster runs, switch to epoch‑based eval and reduce epochs; for finer control, use step‑based eval with moderate eval_steps.

License
- Add an OSS license (e.g., MIT/Apache‑2.0).  
- Acknowledge Jigsaw datasets and Hugging Face Transformers.

Contact
- Open an issue in this repository for questions or suggestions.
