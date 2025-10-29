# Deep Learning Project — Content Moderation (English-only training)

Purpose
- Multi-label toxicity detection pipeline using a multilingual DistilBERT backbone fine-tuned on English Jigsaw data.
- Designed for training, evaluation, threshold tuning, exporting, and lightweight serving.

Highlights
- Entrypoint: `model_training.main` in `model_training.py`
- Key utilities:
  - Data loading: `model_training.load_english_only`
  - Dataset conversion: `model_training.df_to_hfds`
  - Balanced sampler & pos-weight: `model_training.build_balanced_sampler_and_pos_weight`
  - Trainer with BCE pos-weight support: `model_training.PosWeightTrainer`
  - Freeze helper for speed: `model_training.freeze_for_speed`
  - Metrics, threshold tuning & eval: `model_training.compute_metrics`, `model_training.tune_thresholds`, `model_training.eval_with_thresholds`
  - Logs & plots exporter: `model_training.save_logs_and_plots`

Quick start (recommended)
1. Create virtual environment and install dependencies:
   ```sh
   python -m venv .venv
   # Linux / macOS
   source .venv/bin/activate
   # Windows (PowerShell)
   .venv\Scripts\Activate.ps1

   pip install -r requirements.txt
   ```

2. Prepare data
   - This repository expects English-only Jigsaw-like CSVs with at least:
     - `comment_text` (text to classify)
     - one or more binary label columns (e.g., `toxic`, `severe_toxic`, ...)
   - If using the provided loader, follow its expected column names or adapt `load_english_only`.

3. Train (example)
   ```sh
   python -m model_training \
     --data-path /path/to/english_train.csv \
     --output-dir ./runs/exp1 \
     --epochs 3 \
     --batch-size 16 \
     --lr 2e-5
   ```
   - The script exposes flags for data path, hyperparameters, checkpointing, and logging. See `model_training.py` for full list.

4. Tune thresholds (example)
   ```sh
   python -m model_training --tune-thresholds --preds ./runs/exp1/preds_val.npy --labels ./runs/exp1/labels_val.npy
   ```

5. Evaluate with tuned thresholds
   ```sh
   python -m model_training --eval --checkpoint ./runs/exp1/checkpoint_best.pt --thresholds ./runs/exp1/thresholds.json
   ```

Data format and preprocessing
- Input: CSV/Parquet with text column and binary label columns.
- Tokenization: uses Hugging Face transformers (DistilBERT-based tokenizer). Text is lowercased/padded/truncated according to the model's config inside the training script.
- Class imbalance: script computes pos-weight and can use a balanced sampler to mitigate severe imbalance.

Training details & recommendations
- Use reproducible seeds: set random seeds in `model_training.load_english_only` or main entry.
- For faster experiments: freeze backbone layers with `freeze_for_speed` helper before training.
- Use mixed precision (if supported) to reduce memory and speed up training.

Checkpoints & logs
- Checkpoints are saved to `--output-dir` (by default `runs/<exp>`).
- Logs include training/validation losses, per-label metrics, and plots exported by `save_logs_and_plots`.
- Keep best checkpoint (by validation metric) and final checkpoint for reproducibility.

Exporting & serving (production)
- Export model weights and tokenizer together (recommended folder structure):
  ```
  /model-export
    - config.json
    - pytorch_model.bin
    - tokenizer.json (or tokenizer files)
  ```
- Serving options:
  - FastAPI + transformers pipeline: load tokenizer + model, expose a predict endpoint that applies the tuned thresholds.
  - TorchServe or Triton for scalable deployment (export to TorchScript if needed).
- When serving, apply the same preprocessing pipeline and thresholds used during evaluation.

Evaluation & thresholds
- This repo supports threshold tuning per label to maximize F1 or other metrics — see `tune_thresholds`.
- Use holdout validation for threshold selection; do not tune on test data.

Reproducibility & experiment tracking
- Log hyperparameters, seed, dataset version, and git commit hash in the run metadata.
- Use MLFlow/Weights & Biases (optional) for richer experiment tracking; hooks can be added around the training loop.

Best practices
- Validate data quality (no NaNs in text, correct label encoding).
- Start with small batch/epoch to validate pipeline, then scale.
- Monitor class-wise metrics — macro averages can hide poor per-class performance.

Developer notes
- The repository is English-only training focused — multilingual backbone is fine-tuned on English labels.
- See function docstrings in `model_training.py` for detailed behavior and configurable options.

Contributing
- Fork, make changes, add tests for new utilities, and open a pull request.
- Keep changes modular and document CLI options for new features.

License & attribution
- Include appropriate license in root (e.g., MIT or Apache-2.0). Add license file if missing.
- Acknowledge Jigsaw datasets and Hugging Face Transformers where applicable.

Contact / Support
- For issues or feature requests, open an issue in the repository.