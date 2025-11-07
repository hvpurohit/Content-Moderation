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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset, DatasetDict, Features, Value
from torch.utils.data import WeightedRandomSampler

# ---------------- Config ----------------
seed = 42
model_name = "distilbert/distilbert-base-uncased"
max_len = 192                 # slightly higher than 160 to capture more cues
epochs = 3                    # allow a bit more learning; early stopping will cap
lr = 2e-5
weight_decay = 0.02
train_bs = 24                 # raise if VRAM allows
eval_bs = 128
grad_accum = 1
target_evals_per_epoch = 9    # aim for 8-10 evals per epoch
logging_steps = 150           # frequent enough to see loss trends
out_dir = Path("outputs_english_binary_toxic_stepwise")
out_dir.mkdir(parents=True, exist_ok=True)

# Data paths (adjust)
base = Path(r"C:\Users\hvpur\OneDrive\Desktop\sem 7\Deep Learning")
tox_csv = base / "jigsaw-toxic-comment-train.csv"
bias_csv = base / "jigsaw-unintended-bias-train.csv"   # optional English extra

def set_seeds(sd):
    random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)

def pick_text_col(df):
    for c in ["comment_text", "text", "content"]:
        if c in df.columns:
            return c
    raise ValueError(f"No text column found in columns {list(df.columns)}")

def load_binary_english():
    df_tox = pd.read_csv(tox_csv)
    tcol = pick_text_col(df_tox)
    if "toxic" not in df_tox.columns:
        raise ValueError("Expected 'toxic' column in jigsaw-toxic-comment-train.csv")
    df_tox = df_tox[[tcol, "toxic"]].dropna().rename(columns={tcol: "text"})
    df_tox["label"] = df_tox["toxic"].astype(int)
    df_tox = df_tox[["text", "label"]]

    # Optional: modest English boost from unintended-bias
    extra = []
    if bias_csv.exists():
        df_bias = pd.read_csv(bias_csv)
        btxt = pick_text_col(df_bias)
        score_col = next((c for c in ["toxicity", "target", "toxic"] if c in df_bias.columns), None)
        if score_col:
            df_b = df_bias[[btxt, score_col]].dropna().rename(columns={btxt: "text", score_col: "score"})
            df_b["label"] = (df_b["score"] >= 0.5).astype(int)
            # cap size to control runtime
            df_b = df_b.sample(n=min(len(df_b), 100_000), random_state=seed)
            extra.append(df_b[["text", "label"]])
    df_all = pd.concat([df_tox] + extra, ignore_index=True) if extra else df_tox.copy()
    df_all = df_all.dropna(subset=["text"]).reset_index(drop=True)
    df_all["text"] = df_all["text"].astype(str).str.strip()
    df_all = df_all[df_all["text"] != ""]
    return df_all

def df_to_hfds(df):
    feats = Features({"text": Value("string"), "label": Value("int64")})
    return Dataset.from_pandas(df.reset_index(drop=True), features=feats, preserve_index=False)

def make_splits(df):
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, stratify=df["label"])
    dev_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, stratify=temp_df["label"])
    return train_df, dev_df, test_df

def compute_class_weights(labels_np):
    pos = labels_np.sum()
    neg = labels_np.shape[0] - pos
    total = len(labels_np)
    w0 = total / (2.0 * max(1, neg))
    w1 = total / (2.0 * max(1, pos))
    return torch.tensor([w0, w1], dtype=torch.float32)

def build_sampler(labels_np):
    class_counts = np.bincount(labels_np, minlength=2)
    class_weights = 1.0 / np.clip(class_counts, 1, None)
    sample_weights = class_weights[labels_np]
    return WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

def bin_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds = probs.argmax(axis=1)
    labels = np.array(labels)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except Exception:
        auc = float("nan")
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "roc_auc": auc}

def save_logs_and_plots(out_dir: Path):
    state_path = out_dir / "trainer_state.json"
    if not state_path.exists():
        return
    st = json.loads(state_path.read_text(encoding="utf-8"))
    logs = st.get("log_history", [])
    df_logs = pd.json_normalize(logs)
    if "step" not in df_logs and "global_step" in df_logs.columns:
        df_logs["step"] = df_logs["global_step"]
    df_logs.to_csv(out_dir / "metrics_log.csv", index=False)

    def plot(df, out_png, title, ys):
        if df.empty:
            return
        plt.figure(figsize=(9,5))
        for y in ys:
            if y in df.columns:
                plt.plot(df["step"], df[y], label=y)
        plt.title(title); plt.xlabel("step"); plt.ylabel("value")
        plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
        plt.savefig(out_png, dpi=150); plt.close()

    plot(df_logs.dropna(subset=["loss"]), out_dir / "train_loss.png", "Training Loss vs Step", ["loss"])
    plot(df_logs.dropna(subset=["eval_loss"]), out_dir / "eval_loss.png", "Eval Loss vs Step", ["eval_loss"])
    plot(df_logs.dropna(subset=["eval_accuracy","eval_precision","eval_recall","eval_f1"]),
         out_dir / "eval_metrics.png", "Eval Metrics vs Step", ["eval_accuracy","eval_precision","eval_recall","eval_f1"])

def choose_eval_steps(train_size, eff_batch_per_step, target_evals=9):
    steps_per_epoch = max(1, int(np.ceil(train_size / eff_batch_per_step)))
    eval_steps = max(200, steps_per_epoch // target_evals)
    return eval_steps, steps_per_epoch

def tune_threshold_custom(probs_pos, labels, w_precision=0.6, w_recall=0.4):
    y = np.array(labels)
    candidates = np.unique(np.concatenate([[0.0, 1.0], probs_pos]))
    best_val, best_t = -1.0, 0.5
    for t in candidates:
        preds = (probs_pos >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y, preds, average="binary", zero_division=0)
        val = w_precision * p + w_recall * r
        if val > best_val:
            best_val, best_t = val, t
    return float(best_t), float(best_val)

class ClassWeightedTrainer(Trainer):
    def __init__(self, *args, class_weights_cpu=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights_cpu = class_weights_cpu
        self._ce = None
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        device = model.device
        if self._ce is None or (getattr(self._ce, "weight", None) is not None and self._ce.weight.device != device):
            w = None if self._class_weights_cpu is None else self._class_weights_cpu.to(device)
            self._ce = torch.nn.CrossEntropyLoss(weight=w)
        loss = self._ce(logits, labels.to(device))
        return (loss, outputs) if return_outputs else loss
    def get_train_dataloader(self):
        dl = super().get_train_dataloader()
        if hasattr(self, "_sampler") and self._sampler is not None:
            dl.sampler = self._sampler
        return dl

def main():
    set_seeds(seed)

    # 1) Load/split
    df_all = load_binary_english()
    train_df, dev_df, test_df = make_splits(df_all)

    ds = DatasetDict({
        "train": df_to_hfds(train_df),
        "dev":   df_to_hfds(dev_df),
        "test":  df_to_hfds(test_df),
    })

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    def tok(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=max_len)
    ds_tok = ds.map(tok, batched=True, remove_columns=["text"])
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # 2) Sampler/weights
    y_train = np.array(ds_tok["train"]["label"])
    class_weights_cpu = compute_class_weights(y_train)
    sampler = build_sampler(y_train)

    # 3) Determine eval_steps adaptively for ~9 evals per epoch
    eff_batch = train_bs * max(1, torch.cuda.device_count()) * grad_accum
    eval_steps, steps_per_epoch = choose_eval_steps(len(ds_tok["train"]), eff_batch, target_evals_per_epoch)
    print(f"steps_per_epoch={steps_per_epoch}, eval_steps={eval_steps}")

    # 4) Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, problem_type="single_label_classification")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("CUDA available:", torch.cuda.is_available(), "Device:", next(model.parameters()).device)

    # 5) TrainingArguments: step-based eval, warmup+cosine, early stopping on F1
    total_train_steps = steps_per_epoch * epochs
    warmup_steps = max(100, int(0.06 * total_train_steps))

    args = TrainingArguments(
        output_dir=str(out_dir),
        do_train=True, do_eval=True,
        learning_rate=lr,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        logging_strategy="steps",
        logging_steps=logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        fp16=torch.cuda.is_available(),
        seed=seed,
        dataloader_num_workers=0,
        save_total_limit=3,
        report_to=[],
    )

    trainer = ClassWeightedTrainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=ds_tok["dev"],
        tokenizer=tokenizer,
        compute_metrics=bin_metrics,
        class_weights_cpu=class_weights_cpu,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=4, early_stopping_threshold=1e-4)]
    )
    trainer._sampler = sampler

    # 6) Train
    trainer.train()
    trainer.save_state()
    trainer.save_model(str(out_dir / "final_model"))
    tokenizer.save_pretrained(str(out_dir / "final_model"))

    # 7) Precision-leaning threshold tuning on dev
    dev_out = trainer.predict(ds_tok["dev"])
    probs_dev = torch.softmax(torch.tensor(dev_out.predictions), dim=-1).numpy()[:, 1]
    best_thr, best_custom = tune_threshold_custom(probs_dev, np.array(dev_out.label_ids),
                                                  w_precision=0.6, w_recall=0.4)

    # 8) Evaluate on test with tuned threshold
    test_out = trainer.predict(ds_tok["test"])
    probs_test = torch.softmax(torch.tensor(test_out.predictions), dim=-1).numpy()[:, 1]
    preds_test = (probs_test >= best_thr).astype(int)
    y_test = np.array(test_out.label_ids)

    acc = accuracy_score(y_test, preds_test)
    p, r, f1, _ = precision_recall_fscore_support(y_test, preds_test, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_test, probs_test)
    except Exception:
        auc = float("nan")
    metrics = {"test_accuracy": acc, "test_precision": p, "test_recall": r, "test_f1": f1, "test_auc": auc,
               "threshold": best_thr}
    print(metrics)

    with open(out_dir / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_thr}, f, indent=2)
    with open(out_dir / "final_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # 9) Plots
    save_logs_and_plots(out_dir)

if __name__ == "__main__":
    main()
