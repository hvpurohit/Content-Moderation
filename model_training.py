# train_english_only_balanced.py
# Goal: High English-only multilabel performance (aiming for 8–9/10), reliable graphs, fast training
# - English-only (Jigsaw English data only)
# - pos_weight in BCEWithLogitsLoss (neg/pos per label)
# - WeightedRandomSampler to expose positives more often
# - Two-phase fine-tune: freeze lower layers then optionally unfreeze top block for polishing
# - Per-label threshold tuning on English dev set
# - Guaranteed trainer_state.json + CSV + PNG plots
# - Windows-safe main guard; Transformers 4.57.x

import os
os.environ["USE_TF"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, average_precision_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from torch.utils.data import WeightedRandomSampler

# ------------- Config -------------
seed = 42
model_name = "distilbert/distilbert-base-multilingual-cased"  # multilingual DistilBERT works well for English too
max_len = 160  # modest length for better context than 128
epochs_phase1 = 2
epochs_phase2 = 1  # short polish with a bit more trainable capacity
lr = 2e-5
train_bs = 16
eval_bs = 64
grad_accum = 2
logging_steps = 150
out_dir = Path("outputs_english_only_balanced")
out_dir.mkdir(parents=True, exist_ok=True)

# Jigsaw English files (replace with your exact paths)
base = Path(r"C:\Users\hvpur\OneDrive\Desktop\sem 7\Deep Learning")
english_files = {
    "tox":  base / "jigsaw-toxic-comment-train.csv",
    "bias": base / "jigsaw-unintended-bias-train.csv",
}

CATS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def set_seeds(sd):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)

def pick_text_col(df):
    for c in ["comment_text", "text", "content"]:
        if c in df.columns:
            return c
    raise ValueError(f"No text column found; got: {list(df.columns)}")

def load_english_only():
    df_tox = pd.read_csv(english_files["tox"])
    tcol = pick_text_col(df_tox)
    present = [c for c in CATS if c in df_tox.columns]
    if not present and "toxic" in df_tox.columns:
        present = ["toxic"]
    df_tox = df_tox[[tcol] + present].dropna().rename(columns={tcol: "text"})
    for c in present:
        df_tox[c] = df_tox[c].astype(int)

    # Optional: use unintended-bias as extra English data by binarizing toxicity score if present
    df_bias = pd.read_csv(english_files["bias"])
    btxt = pick_text_col(df_bias)
    score_col = next((c for c in ["toxicity", "target", "toxic"] if c in df_bias.columns), None)
    if score_col is None:
        # fall back to only df_tox
        df_train_all = df_tox.copy()
        cats = sorted(list(set(present) | set(CATS)))
        for c in cats:
            if c not in df_train_all.columns:
                df_train_all[c] = 0
        return df_train_all[["text"] + cats], cats

    df_bias_bin = df_bias[[btxt, score_col]].dropna().rename(columns={btxt: "text", score_col: "score"})
    df_bias_bin["toxic"] = (df_bias_bin["score"] >= 0.5).astype(int)
    df_bias_bin = df_bias_bin[["text", "toxic"]]

    cats = sorted(list(set(present) | {"toxic"}))
    for c in cats:
        if c not in df_tox.columns:
            df_tox[c] = 0
        if c not in df_bias_bin.columns:
            df_bias_bin[c] = 0

    df_train_all = pd.concat([df_tox[["text"] + cats], df_bias_bin[["text"] + cats]], ignore_index=True)
    return df_train_all, cats

def df_to_hfds(df, cats):
    out = df.copy()
    out[cats] = out[cats].astype(np.float32)
    out["labels"] = out[cats].values.tolist()
    out = out[["text", "labels"]]
    features = Features({"text": Value("string"), "labels": Sequence(Value("float32"))})
    return Dataset.from_pandas(out.reset_index(drop=True), features=features, preserve_index=False)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    labels = np.array(labels)
    preds = (probs >= 0.5).astype(int)
    _, _, f_micro, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    _, _, f_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    out = {"f1_micro": f_micro, "f1_macro": f_macro}
    pr_aucs, roc_aucs = [], []
    for j in range(probs.shape[1]):
        if len(np.unique(labels[:, j])) > 1:
            pr_aucs.append(average_precision_score(labels[:, j], probs[:, j]))
            try:
                roc_aucs.append(roc_auc_score(labels[:, j], probs[:, j]))
            except ValueError:
                pass
    if pr_aucs:
        out["pr_auc_macro"] = float(np.mean(pr_aucs))
    if roc_aucs:
        out["roc_auc_macro"] = float(np.mean(roc_aucs))
    return out

def freeze_for_speed(model, num_frozen=3):
    if hasattr(model, "distilbert"):
        for p in model.distilbert.embeddings.parameters():
            p.requires_grad = False
        for i, layer in enumerate(model.distilbert.transformer.layer):
            if i < num_frozen:
                for p in layer.parameters():
                    p.requires_grad = False
    return model

def build_balanced_sampler_and_pos_weight(train_labels_np):
    pos_counts = train_labels_np.sum(axis=0)
    neg_counts = train_labels_np.shape[0] - pos_counts
    # pos_weight = neg/pos per label (clamped for stability)
    pos_weight_cpu = torch.tensor((neg_counts / np.clip(pos_counts, 1.0, None)).astype(np.float32))
    pos_weight_cpu = torch.clamp(pos_weight_cpu, 1.0, 10.0)  # avoid extremes [web:218][web:230][web:219]
    # sample weights per example: sum of inverse class frequency for its positive labels
    class_weights = 1.0 / np.clip(pos_counts + 1e-6, 1e-6, None)
    sample_weights = (train_labels_np * class_weights).sum(axis=1)
    sample_weights = sample_weights + (sample_weights == 0) * (class_weights.mean() * 0.1)
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler, pos_weight_cpu

class PosWeightTrainer(Trainer):
    def __init__(self, *args, pos_weight_cpu=None, **kwargs):
        self._pos_weight_cpu = pos_weight_cpu
        super().__init__(*args, **kwargs)
        self._bce = None
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        device = model.device
        labels = labels.to(device)
        # Build/refresh BCEWithLogitsLoss on correct device with pos_weight [web:230]
        if (self._bce is None) or (getattr(self._bce, 'pos_weight', None) is not None and self._bce.pos_weight.device != device):
            pos_weight = None
            if self._pos_weight_cpu is not None:
                pos_weight = self._pos_weight_cpu.to(device)
            self._bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self._bce(logits, labels)
        return (loss, outputs) if return_outputs else loss
    def get_train_dataloader(self):
        dl = super().get_train_dataloader()
        if hasattr(self, "_sampler") and self._sampler is not None:
            dl.sampler = self._sampler
        return dl

def tune_thresholds(dev_logits, dev_labels, grid=None):
    if grid is None:
        grid = np.linspace(0.1, 0.9, 33)
    probs = torch.sigmoid(torch.tensor(dev_logits)).numpy()
    labels = np.array(dev_labels)
    best_thr = [0.5] * probs.shape[1]
    for j in range(probs.shape[1]):
        if len(np.unique(labels[:, j])) < 2:
            best_thr[j] = 0.5
            continue
        best_f1, best_t = -1.0, 0.5
        for t in grid:
            preds = (probs[:, j] >= t).astype(int)
            f1 = f1_score(labels[:, j], preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thr[j] = float(best_t)
    return best_thr

def eval_with_thresholds(trainer, dataset, thresholds):
    out = trainer.predict(dataset)
    probs = torch.sigmoid(torch.tensor(out.predictions)).numpy()
    labels = np.array(out.label_ids)
    thr = np.array(thresholds, dtype=float)
    preds = (probs >= thr.reshape(1, -1)).astype(int)
    _, _, f_micro, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    _, _, f_macro, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    pr_aucs, roc_aucs = [], []
    for j in range(probs.shape[1]):
        if len(np.unique(labels[:, j])) > 1:
            pr_aucs.append(average_precision_score(labels[:, j], probs[:, j]))
            try:
                roc_aucs.append(roc_auc_score(labels[:, j], probs[:, j]))
            except ValueError:
                pass
    out = {"f1_micro": f_micro, "f1_macro": f_macro}
    if pr_aucs:
        out["pr_auc_macro"] = float(np.mean(pr_aucs))
    if roc_aucs:
        out["roc_auc_macro"] = float(np.mean(roc_aucs))
    return out

def save_logs_and_plots(out_dir: Path):
    state_path = out_dir / "trainer_state.json"
    csv_path = out_dir / "metrics_log.csv"
    if not state_path.exists():
        print("trainer_state.json not found; but trainer.save_state() was called – check permissions or output_dir.")
        return
    st = json.loads(state_path.read_text(encoding="utf-8"))
    logs = st.get("log_history", [])
    df_logs = pd.json_normalize(logs)
    if "step" not in df_logs.columns and "global_step" in df_logs.columns:
        df_logs["step"] = df_logs["global_step"]
    df_logs.to_csv(csv_path, index=False)
    print("Wrote metrics CSV:", csv_path.resolve().as_posix())

    def plot_and_save(df_logs: pd.DataFrame, out_png: Path, title: str, x_col: str, y_cols: list):
        if df_logs.empty:
            return
        plt.figure(figsize=(9, 5))
        for y in y_cols:
            if y in df_logs.columns:
                plt.plot(df_logs[x_col], df_logs[y], label=y)
        plt.title(title); plt.xlabel(x_col); plt.ylabel("value")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(out_png, dpi=150); plt.close()
        print("Saved graph:", out_png.resolve().as_posix())

    if "loss" in df_logs:
        plot_and_save(df_logs.dropna(subset=["loss"]), out_dir / "train_loss.png", "Training Loss vs Step", "step", ["loss"])
    if "eval_loss" in df_logs:
        plot_and_save(df_logs.dropna(subset=["eval_loss"]), out_dir / "eval_loss.png", "Eval Loss vs Step", "step", ["eval_loss"])
    cols = [c for c in ["eval_f1_micro", "eval_f1_macro", "eval_pr_auc_macro", "eval_roc_auc_macro"] if c in df_logs.columns]
    if cols:
        plot_and_save(df_logs.dropna(subset=cols), out_dir / "eval_metrics.png", "Eval Metrics vs Step", "step", cols)

def main():
    set_seeds(seed)

    # 1) Data: English only
    df_all, cats = load_english_only()
    # Train/dev/test split within English (stratify by any-positive)
    pos_flag = (df_all[cats].sum(axis=1) > 0).astype(int)
    train_df, temp_df = train_test_split(df_all, test_size=0.2, random_state=seed, stratify=pos_flag)
    pos_flag_temp = (temp_df[cats].sum(axis=1) > 0).astype(int)
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=pos_flag_temp)

    raw = DatasetDict({
        "train": df_to_hfds(train_df, cats),
        "dev":   df_to_hfds(dev_df,   cats),
        "val":   df_to_hfds(dev_df,   cats),   # use dev as validation for reporting
        "test":  df_to_hfds(test_df,  cats),
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_len)

    tokenized = raw.map(tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Sampler + pos_weight from train labels
    train_labels_np = np.array(tokenized["train"]["labels"])
    sampler, pos_weight_cpu = build_balanced_sampler_and_pos_weight(train_labels_np)

    # 2) Phase 1: freeze lower layers for speed
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(cats),
        problem_type="multi_label_classification",
    )
    model = freeze_for_speed(model, num_frozen=3)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("CUDA available:", torch.cuda.is_available(), "Device:", next(model.parameters()).device)

    args1 = TrainingArguments(
        output_dir=str(out_dir / "phase1"),
        do_train=True,
        do_eval=True,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs_phase1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=logging_steps,
        fp16=torch.cuda.is_available(),
        seed=seed,
        dataloader_num_workers=0,
        save_total_limit=2,
        report_to=[],
    )

    trainer1 = PosWeightTrainer(
        model=model,
        args=args1,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        pos_weight_cpu=pos_weight_cpu,
    )
    # attach sampler
    trainer1._sampler = sampler

    trainer1.train()
    trainer1.save_state()
    trainer1.save_model(str(out_dir / "phase1_final"))

    # 3) Phase 2: unfreeze one more top block for polishing (short)
    if hasattr(model, "distilbert"):
        layers = list(model.distilbert.transformer.layer)
        # Unfreeze the topmost block
        for p in layers[-1].parameters():
            p.requires_grad = True

    # Ensure model remains on device after unfreezing
    model.to(device)

    args2 = TrainingArguments(
        output_dir=str(out_dir / "phase2"),
        do_train=True,
        do_eval=True,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs_phase2,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=logging_steps,
        fp16=torch.cuda.is_available(),
        seed=seed,
        dataloader_num_workers=0,
        save_total_limit=2,
        report_to=[],
    )

    trainer2 = PosWeightTrainer(
        model=model,
        args=args2,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        pos_weight_cpu=pos_weight_cpu,
    )
    trainer2._sampler = sampler

    trainer2.train()
    trainer2.save_state()
    trainer2.save_model(str(out_dir / "phase2_final"))

    # 4) Evaluate and threshold tuning on English dev
    val_default = trainer2.evaluate(eval_dataset=tokenized["val"])
    test_default = trainer2.evaluate(eval_dataset=tokenized["test"])
    print("Validation (default thr=0.5):", val_default)
    print("Test (default thr=0.5):", test_default)

    dev_out = trainer2.predict(tokenized["dev"])
    best_thresholds = tune_thresholds(dev_out.predictions, dev_out.label_ids)
    thr_map = dict(zip(cats, best_thresholds))
    print("Per-label thresholds (dev-tuned):", thr_map)

    val_tuned = eval_with_thresholds(trainer2, tokenized["val"], best_thresholds)
    test_tuned = eval_with_thresholds(trainer2, tokenized["test"], best_thresholds)
    print("Validation (tuned):", val_tuned)
    print("Test (tuned):", test_tuned)

    # 5) Save export with thresholds for the English model
    export_dir = out_dir / "best_english_only"
    trainer2.save_model(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))
    with open(export_dir / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump({"categories": cats, "thresholds": best_thresholds}, f, ensure_ascii=False, indent=2)
    print("Saved to:", export_dir.resolve().as_posix())

    # 6) Save logs + plots for both phases
    save_logs_and_plots(out_dir / "phase1")
    save_logs_and_plots(out_dir / "phase2")

if __name__ == "__main__":
    main()
