#!/usr/bin/env python3
"""
Dynamic LoRA with Fisher Information-based Rank Adaptation
==========================================================
Fine-tunes DeBERTa-v3-base on GLUE RTE.
Baseline: lora.ipynb  (fixed r=8 on query_proj + value_proj)
This file: per-layer rank adapts dynamically using diagonal Fisher trace.

Phases
------
  1. Warmup  (steps 0 .. WARMUP_STEPS-1)
       All layers fixed at r=R_INIT.  Fisher EMA accumulates but no action.
  2. Dynamic (step >= WARMUP_STEPS, every UPDATE_INTERVAL steps)
       Ranks reallocated from Fisher budget shares.
  3. Convergence
       Ranks stabilise; training continues to end of EPOCHS.

Rank formula
------------
  F_l  = sum( grad_i^2 )  for every LoRA param in layer l   <- raw trace
  F̂_l  = beta*F̂_prev + (1-beta)*F_l                         <- EMA
  F̃_l  = F̂_l / sum_j(F̂_j)                                   <- budget share
  r_l  = clip( round(r_min + (r_max - r_min) * F̃_l), r_min, r_max )

Install
-------
  pip install transformers datasets evaluate peft accelerate sentencepiece protobuf matplotlib scikit-learn scipy
"""
import os
import time
import json
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import evaluate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME      = "microsoft/deberta-v3-base"
TASK            = "rte"
MAX_LENGTH      = 256
LORA_ALPHA      = 16
LORA_DROPOUT    = 0.05
LR              = 5e-4
EPOCHS          = 55
BATCH_SIZE      = 16
GRAD_ACCUM      = 1
WEIGHT_DECAY    = 0.01
WARMUP_RATIO    = 0.06
SEED            = 42
TARGET_MODULES  = ["query_proj", "value_proj"]

R_INIT          = 4      # Rank used during warmup phase
R_MIN           = 2      # Minimum rank any layer may receive
R_MAX           = 8      # Maximum rank (equal to baseline budget)
EMA_BETA        = 0.97    # Fisher EMA decay
WARMUP_STEPS    = 200    # Backward passes before first rank update
UPDATE_INTERVAL = 200    # Backward passes between rank updates

OUTPUT_DIR = "./dynamic_lora_rte_output"
if torch.cuda.is_available():
    DEVICE = "cuda:4"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

ID2LABEL = {0: "entailment", 1: "not_entailment"}
LABEL2ID = {"entailment": 0, "not_entailment": 1}

torch.manual_seed(SEED)
np.random.seed(SEED)


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Dynamic LoRA Linear Layer
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicLoRALinear(nn.Module):
    """
    Frozen base Linear augmented with trainable low-rank adapters A, B.

    Forward: out = W·x + (alpha/r) · B·A·x

    set_rank(new_r) resizes A and B in-place:
      expand  → pad with small-Gaussian A rows and zero B columns
      shrink  → keep top-new_r components by ||B[:,i]||·||A[i,:]|| importance
    """

    def __init__(self, base: nn.Linear, r: int, lora_alpha: int, dropout: float):
        super().__init__()
        self.base         = base
        self.in_features  = base.in_features
        self.out_features = base.out_features
        self.r            = r
        self.lora_alpha   = lora_alpha

        for p in self.base.parameters():
            p.requires_grad_(False)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.lora_A  = nn.Parameter(torch.randn(r, self.in_features) * 0.02)
        self.lora_B  = nn.Parameter(torch.zeros(self.out_features, r))

    @property
    def scaling(self) -> float:
        return self.lora_alpha / self.r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return self.base(x) + lora_out

    def set_rank(self, new_r: int) -> bool:
        """Resize adapters; returns True if rank actually changed."""
        if new_r == self.r:
            return False
        dev, dt = self.lora_A.device, self.lora_A.dtype
        if new_r > self.r:
            extra = new_r - self.r
            new_A = torch.cat(
                [self.lora_A.data,
                 torch.randn(extra, self.in_features, device=dev, dtype=dt) * 0.02],
                dim=0,
            )
            new_B = torch.cat(
                [self.lora_B.data,
                 torch.zeros(self.out_features, extra, device=dev, dtype=dt)],
                dim=1,
            )
        else:
            # Keep the most expressive rank-1 components
            importance = self.lora_B.data.norm(dim=0) * self.lora_A.data.norm(dim=1)
            keep  = importance.topk(new_r).indices.sort().values
            new_A = self.lora_A.data[keep]
            new_B = self.lora_B.data[:, keep]
        self.lora_A = nn.Parameter(new_A)
        self.lora_B = nn.Parameter(new_B)
        self.r      = new_r
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Model Surgery
# ═══════════════════════════════════════════════════════════════════════════════

def inject_dynamic_lora(
    model: nn.Module,
    target_modules: list,
    r: int,
    lora_alpha: int,
    dropout: float,
) -> dict:
    """
    Walk the model tree and replace every nn.Linear whose local name
    is in target_modules with a DynamicLoRALinear.

    Returns dict: full_dotted_name -> DynamicLoRALinear
    """
    replaced: dict = {}

    def _walk(parent: nn.Module, prefix: str):
        for child_name, child_mod in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child_mod, nn.Linear) and child_name in target_modules:
                new_layer = DynamicLoRALinear(child_mod, r, lora_alpha, dropout)
                setattr(parent, child_name, new_layer)
                replaced[full_name] = new_layer
            else:
                _walk(child_mod, full_name)

    _walk(model, "")
    return replaced


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Fisher Scheduler
# ═══════════════════════════════════════════════════════════════════════════════

"""
FisherScheduler — Fixed Version
================================
Fixes applied vs original:

  Fix 1 — Per-projection-type normalization
    Query scores compete only against other query scores; value against value.
    Prevents value's 10-100x larger Fisher scores from collapsing query to r_min.

  Fix 2 — Optional combined lora_A + lora_B signal (alpha parameter)
    alpha=1.0  → pure lora_A  (original behaviour, Theorem-2 grounded)
    alpha=0.0  → pure lora_B
    alpha=0.5  → equal weight after per-matrix per-proj normalisation
    Each matrix is normalised independently before blending, avoiding the
    scale-conflict that would arise from raw value combination.

  Fix 3 — Rank-change dampening (vote-to-change)
    A rank change is only committed if the *same* proposed rank is seen for
    `patience` consecutive update steps.  Eliminates the ±1 oscillation
    visible in the training log (e.g. L8.value cycling 6→7→6→7).
"""

import numpy as np
from collections import defaultdict


class FisherScheduler:
    def __init__(
        self,
        lora_layers,
        r_min,
        r_max,
        ema_beta,
        warmup_steps,
        update_interval,
        alpha=0.5,          # Fix 2: weight of lora_A vs lora_B (1.0 = A only)
        patience=2,         # Fix 3: consecutive votes needed before rank change
    ):
        self.lora_layers     = lora_layers
        self.r_min           = r_min
        self.r_max           = r_max
        self.ema_beta        = ema_beta
        self.warmup_steps    = warmup_steps
        self.update_interval = update_interval
        self.alpha           = alpha
        self.patience        = patience

        # ── EMA state ────────────────────────────────────────────────────────
        self.fisher_ema_A  = {n: 0.0 for n in lora_layers}
        self.fisher_ema_B  = {n: 0.0 for n in lora_layers}
        self.step_count    = {n: 0   for n in lora_layers}

        # ── Fix 3: vote buffer ────────────────────────────────────────────────
        # Maps layer_name → list of last `patience` proposed ranks
        self._vote_buffer  = {n: [] for n in lora_layers}

        # ── Logging ───────────────────────────────────────────────────────────
        self.update_steps     = []
        self.rank_snapshots   = defaultdict(list)
        self.fisher_snapshots = defaultdict(list)  # stores combined score

    # ═════════════════════════════════════════════════════════════════════════
    # EMA helpers
    # ═════════════════════════════════════════════════════════════════════════

    def _update_ema(self):
        for name, layer in self.lora_layers.items():
            grad_A = layer.lora_A.grad
            grad_B = layer.lora_B.grad

            if grad_A is None and grad_B is None:
                continue

            self.step_count[name] += 1
            t    = self.step_count[name]
            beta = self.ema_beta

            if grad_A is not None:
                trace_A = grad_A.data.pow(2).sum().item()
                self.fisher_ema_A[name] = (
                    beta * self.fisher_ema_A[name] + (1.0 - beta) * trace_A
                )

            if grad_B is not None:
                trace_B = grad_B.data.pow(2).sum().item()
                self.fisher_ema_B[name] = (
                    beta * self.fisher_ema_B[name] + (1.0 - beta) * trace_B
                )

    def _bias_corrected(self, ema_dict, name):
        t = self.step_count[name]
        if t == 0:
            return 0.0
        correction = 1.0 - self.ema_beta ** t
        return ema_dict[name] / correction

    # ═════════════════════════════════════════════════════════════════════════
    # Fix 1 + 2: per-projection normalisation with optional A+B blend
    # ═════════════════════════════════════════════════════════════════════════

    def _projection_type(self, name):
        """Return the local module name, e.g. 'query_proj' or 'value_proj'."""
        return name.split(".")[-1]

    def _log_compress(self, s_norm, stretch=10.0):
        """Map [0,1] → [0,1] with log compression to reduce dynamic-range skew."""
        return np.log1p(s_norm * stretch) / np.log1p(stretch)

    def _normalise_scores(self, raw_scores):
        """
        Per-projection-type min-max normalisation followed by log compression.

        raw_scores : dict  name → float
        returns    : dict  name → float in [0, 1]
        """
        # Group by projection type
        by_proj = defaultdict(dict)
        for name, score in raw_scores.items():
            by_proj[self._projection_type(name)][name] = score

        normed = {}
        for proj, group in by_proj.items():
            vals   = list(group.values())
            lo, hi = min(vals), max(vals) + 1e-12
            rng    = hi - lo + 1e-12
            for name, score in group.items():
                s_norm        = (score - lo) / rng          # [0, 1]
                normed[name]  = self._log_compress(s_norm)  # still [0, 1]

        return normed

    def _compute_proposed_ranks(self):
        """
        1. Compute bias-corrected EMA for A and B separately.
        2. Normalise each signal per projection type (Fix 1).
        3. Blend with alpha (Fix 2).
        4. Map blended score → integer rank.
        """
        raw_A = {n: self._bias_corrected(self.fisher_ema_A, n)
                 for n in self.lora_layers}
        raw_B = {n: self._bias_corrected(self.fisher_ema_B, n)
                 for n in self.lora_layers}

        normed_A = self._normalise_scores(raw_A)
        normed_B = self._normalise_scores(raw_B)

        ranks = {}
        for name in self.lora_layers:
            # Blended score in [0, 1]
            blended = (
                self.alpha       * normed_A[name]
                + (1 - self.alpha) * normed_B[name]
            )
            r = int(np.clip(
                np.round(self.r_min + (self.r_max - self.r_min) * blended),
                self.r_min,
                self.r_max,
            ))
            ranks[name] = r

        return ranks

    # ═════════════════════════════════════════════════════════════════════════
    # Fix 3: vote-to-change dampening
    # ═════════════════════════════════════════════════════════════════════════

    def _apply_dampening(self, proposed):
        """
        Only return a rank for a layer if the same value has been proposed
        for `patience` consecutive update steps.

        Returns a dict of ranks that are *committed* (may be subset of proposed,
        and committed rank may differ from proposed if patience not yet met).
        """
        committed = {}
        for name, new_r in proposed.items():
            buf = self._vote_buffer[name]
            buf.append(new_r)

            # Keep only the last `patience` votes
            if len(buf) > self.patience:
                buf.pop(0)

            if len(buf) == self.patience and len(set(buf)) == 1:
                # All votes agree — commit
                committed[name] = new_r
                buf.clear()           # reset so next change needs patience again
            else:
                # Not yet stable — keep current rank
                committed[name] = self.lora_layers[name].r

        return committed

    # ═════════════════════════════════════════════════════════════════════════
    # Public API (unchanged from original)
    # ═════════════════════════════════════════════════════════════════════════

    def step(self, global_step):
        self._update_ema()
        if global_step < self.warmup_steps:
            return None
        if (global_step - self.warmup_steps) % self.update_interval != 0:
            return None

        proposed  = self._compute_proposed_ranks()
        committed = self._apply_dampening(proposed)   # Fix 3
        return committed

    def record_update(self, step, applied_ranks):
        self.update_steps.append(step)

        # Log the *combined* normalised score for plotting
        raw_A    = {n: self._bias_corrected(self.fisher_ema_A, n)
                    for n in self.lora_layers}
        raw_B    = {n: self._bias_corrected(self.fisher_ema_B, n)
                    for n in self.lora_layers}
        normed_A = self._normalise_scores(raw_A)
        normed_B = self._normalise_scores(raw_B)

        print(f"    Fisher scores at step {step}:")
        for name in sorted(self.lora_layers,
                           key=lambda n: (self.layer_index(n), n)):
            idx    = self.layer_index(name)
            proj   = name.split(".")[-1].replace("_proj", "")
            score  = (self.alpha * normed_A[name]
                      + (1 - self.alpha) * normed_B[name])
            raw_a  = self._bias_corrected(self.fisher_ema_A, name)
            raw_b  = self._bias_corrected(self.fisher_ema_B, name)
            r      = applied_ranks[name]
            print(
                f"      L{idx:2d}.{proj:6s}: "
                f"A={raw_a:.6f}  B={raw_b:.6f}  "
                f"blend={score:.4f}  → r={r}"
            )
            self.rank_snapshots[name].append(r)
            self.fisher_snapshots[name].append(score)

    def current_ranks(self):
        return {n: layer.r for n, layer in self.lora_layers.items()}

    @staticmethod
    def layer_index(name):
        parts = name.split(".")
        for i, p in enumerate(parts):
            if p == "layer" and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    pass
        return -1


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Data
# ═══════════════════════════════════════════════════════════════════════════════

def build_datasets(tokenizer):
    raw = load_dataset("glue", TASK)

    def tokenize(batch):
        return tokenizer(
            batch["sentence1"], batch["sentence2"],
            truncation=True, max_length=MAX_LENGTH,
        )

    keep = ["input_ids", "attention_mask", "labels"]

    def prep(split):
        ds = raw[split].map(tokenize, batched=True)
        ds = ds.rename_column("label", "labels")
        return ds.select_columns([c for c in keep if c in ds.column_names])

    return prep("train"), prep("validation"), raw["validation"]


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_model(model, loader, metric, device):
    model.eval()
    preds, refs, loss_sum = [], [], 0.0
    for batch in loader:
        labels = batch.pop("labels").to(device)
        inputs = {k: v.to(device) for k, v in batch.items()}
        out    = model(**inputs, labels=labels)
        loss_sum += out.loss.item()
        preds    += torch.argmax(out.logits, -1).tolist()
        refs     += labels.tolist()
    model.train()
    res       = metric.compute(predictions=preds, references=refs)
    res["loss"] = loss_sum / len(loader)
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Optimizer factory
# ═══════════════════════════════════════════════════════════════════════════════

def make_optimizer(model, current_opt_step: int, total_opt_steps: int, warmup_lr_steps: int):
    """
    Creates a fresh AdamW + linear-warmup scheduler.
    Called once at start, and again whenever parameter shapes change
    (rank update) because the old optimizer state is stale.
    Note: recreating loses accumulated momentum — an acceptable limitation
    for the first-pass research implementation.
    """
    params    = [p for p in model.parameters() if p.requires_grad]
    # print(params)
    optimizer = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = AdamW(
    params,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),   # beta1, beta2
    eps=1e-6              # epsilon
)
#     optimizer = AdamW([
#     {"params": lora_params,  "lr": LR, "weight_decay": 0.0},
#     {"params": other_params, "lr": LR, "weight_decay": WEIGHT_DECAY},
# ])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(0, warmup_lr_steps - current_opt_step),
        num_training_steps=max(1, total_opt_steps - current_opt_step),
    )
    return optimizer, scheduler


def update_optimizer_after_rank_change(
    model, optimizer, lr_sched, changed_layer_names, 
    lora_layers, opt_step, total_opt_steps, warmup_lr_steps
):
    """
    Rebuild optimizer preserving Adam state for unchanged params.
    Only changed LoRA params lose their momentum (unavoidable).
    """
    # Save state for all current params
    old_state = optimizer.state
    old_param_id_to_state = {id(p): s for p, s in old_state.items()}
    
    # Build new optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    new_optimizer = AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Restore state for params whose shape didn't change
    changed_param_ids = set()
    for name in changed_layer_names:
        layer = lora_layers[name]
        changed_param_ids.add(id(layer.lora_A))
        changed_param_ids.add(id(layer.lora_B))
    
    for p in params:
        old_id = id(p)
        if old_id in old_param_id_to_state and old_id not in changed_param_ids:
            new_optimizer.state[p] = old_param_id_to_state[old_id]
    
    # Rebuild scheduler from current position
    new_scheduler = get_linear_schedule_with_warmup(
        new_optimizer,
        num_warmup_steps=max(0, warmup_lr_steps - opt_step),
        num_training_steps=max(1, total_opt_steps - opt_step),
    )
    
    return new_optimizer, new_scheduler    

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_results(sched: FisherScheduler, epoch_results: list, out_dir: str) -> None:
    names = sorted(sched.rank_snapshots.keys())
    steps = sched.update_steps

    # ── Rank trajectories + depth pattern ──────────────────────────────────
    if steps:
        _, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for name in names:
            ranks = sched.rank_snapshots[name]
            idx   = FisherScheduler.layer_index(name)
            proj  = name.split(".")[-1].replace("_proj", "")
            ax.plot(steps[:len(ranks)], ranks,
                    label=f"L{idx}-{proj}", alpha=0.8, linewidth=1.2)
        ax.axvline(WARMUP_STEPS, color="gray", linestyle="--", alpha=0.5, label="warmup end")
        ax.set(xlabel="Training Step", ylabel="Rank",
               title="LoRA Rank Trajectories (Fisher-Adaptive)", ylim=(0, R_MAX + 1))
        ax.legend(ncol=3, fontsize=7)
        ax.grid(alpha=0.3)

        # Expected depth pattern: lower layers → lower rank, upper → higher
        ax2 = axes[1]
        depth: dict = defaultdict(dict)
        for name in names:
            if sched.rank_snapshots[name]:
                idx  = FisherScheduler.layer_index(name)
                proj = name.split(".")[-1]
                depth[idx][proj] = sched.rank_snapshots[name][-1]
        layer_ids = sorted(depth.keys())
        q_ranks   = [depth[i].get("query_proj") for i in layer_ids]
        v_ranks   = [depth[i].get("value_proj")  for i in layer_ids]
        ax2.plot(layer_ids, q_ranks, "o-", label="query", color="steelblue")
        ax2.plot(layer_ids, v_ranks, "s-", label="value",  color="darkorange")
        ax2.set(xlabel="Transformer Layer Index", ylabel="Final Rank",
                title="Depth Pattern: Final Rank per Layer\n"
                      "(lower layers expected lower rank)",
                ylim=(0, R_MAX + 1), xticks=layer_ids)
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "rank_analysis.png"), dpi=150)
        plt.close()
        print(f"  rank_analysis.png saved")

    # ── Training curves ─────────────────────────────────────────────────────
    epochs     = [r["epoch"]        for r in epoch_results]
    val_acc    = [r["val_accuracy"] * 100 for r in epoch_results]
    val_loss   = [r["val_loss"]     for r in epoch_results]
    train_loss = [r["train_loss"]   for r in epoch_results]

    _, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, val_acc, "o-", color="green")
    axes[0].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Validation Accuracy")
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_loss, "o-", label="Train",      color="steelblue")
    axes[1].plot(epochs, val_loss,   "s-", label="Validation", color="crimson")
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Training & Validation Loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  training_curves.png saved")

    # ── Fisher heatmap ──────────────────────────────────────────────────────
    if steps and len(names) > 0:
        data = np.array(
            [[sched.fisher_snapshots[n][t] for n in names]
             for t in range(len(steps))]
        ).T   # shape: [n_layers, n_updates]

        _, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(data, aspect="auto", origin="lower",
                       extent=[steps[0], steps[-1], 0, len(names)])
        plt.colorbar(im, ax=ax, label="Fisher EMA (F̂)")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Layer")
        short_names = [n.split(".")[-2] + "." + n.split(".")[-1] for n in names]
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(short_names, fontsize=6)
        ax.set_title("Fisher Information EMA per LoRA Layer over Training")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "fisher_heatmap.png"), dpi=150)
        plt.close()
        print(f"  fisher_heatmap.png saved")


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  Main Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    # ── Model setup ──────────────────────────────────────────────────────────
    print("\nLoading DeBERTa-v3 tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32,
    )

    print(f"Injecting Dynamic LoRA  (r_init={R_INIT}, r_min={R_MIN}, r_max={R_MAX})...")
    # lora_layers = inject_dynamic_lora(
    #     model, TARGET_MODULES, r=R_INIT, lora_alpha=LORA_ALPHA, dropout=LORA_DROPOUT
    # )
    for param in model.parameters():
        param.requires_grad_(False)
    
        # Then unfreeze only LoRA params + classifier + pooler
    lora_layers = inject_dynamic_lora(
            model, TARGET_MODULES, r=R_INIT, 
            lora_alpha=LORA_ALPHA, dropout=LORA_DROPOUT
        )
    for name, param in model.named_parameters():
            if "classifier" in name or "pooler" in name:
                param.requires_grad_(True)
    print(f"  Replaced {len(lora_layers)} Linear layers with DynamicLoRALinear")

    # Keep classifier + pooler trainable (mirrors baseline modules_to_save)
    for name, param in model.named_parameters():
        if "classifier" in name or "pooler" in name:
            param.requires_grad_(True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable : {n_trainable:,} / {n_total:,}  ({100*n_trainable/n_total:.2f}%)")

    assert all(p.dtype == torch.float32 for p in model.parameters()), \
        "Mixed dtypes detected — set torch_dtype=torch.float32"

    model = model.to(DEVICE)

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading RTE dataset...")
    train_ds, val_ds, _ = build_datasets(tokenizer)
    collator    = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collator)
    val_loader   = DataLoader(val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collator)
    print(f"  Train : {len(train_ds):,}  |  Val : {len(val_ds):,}")

    # ── Optimizer & LR scheduler ─────────────────────────────────────────────
    total_opt_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_lr_steps = int(WARMUP_RATIO * total_opt_steps)
    print(f"\n  Total optimizer steps : {total_opt_steps}")
    print(f"  LR warmup steps       : {warmup_lr_steps}")
    print(f"  Fisher warmup steps   : {WARMUP_STEPS}")

    optimizer, lr_sched = make_optimizer(model, 0, total_opt_steps, warmup_lr_steps)

    fisher_sched = FisherScheduler(
        lora_layers, R_MIN, R_MAX, EMA_BETA, WARMUP_STEPS, UPDATE_INTERVAL
    )
    metric = evaluate.load("glue", TASK)

    # ── Training ─────────────────────────────────────────────────────────────
    print("\n>>> Starting Dynamic LoRA training on RTE...\n")
    t0              = time.time()
    global_step     = 0   # increments every backward (every batch)
    opt_step        = 0   # increments every optimizer.step()
    accum_count     = 0   # batches accumulated since last optimizer.step()
    best_acc        = 0.0
    epoch_results   = []
    rank_update_log = []

    model.train()
    optimizer.zero_grad()

    for epoch in range(EPOCHS):
        ep_loss   = 0.0
        n_batches = 0

        for batch in train_loader:
            labels = batch.pop("labels").to(DEVICE)
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}

            # ── Forward + backward ──────────────────────────────────────────
            out  = model(**inputs, labels=labels)
            loss = out.loss / GRAD_ACCUM
            loss.backward()

            ep_loss   += out.loss.item()
            n_batches += 1
            global_step += 1
            accum_count += 1

            # ── Fisher EMA update; get rank proposal if update step ─────────
            proposed = fisher_sched.step(global_step)

            if proposed is not None:
                # Identify layers whose rank will actually change
                changed = {
                    name: new_r
                    for name, new_r in proposed.items()
                    if new_r != lora_layers[name].r
                }

                # Apply all proposed ranks (even unchanged ones are recorded)
                for name, new_r in proposed.items():
                    lora_layers[name].set_rank(new_r)

                fisher_sched.record_update(global_step, proposed)
                rank_update_log.append({"step": global_step, "ranks": proposed})

                if changed:
                    rank_str = "  ".join(
                        f"L{FisherScheduler.layer_index(k)}"
                        f".{k.split('.')[-1].replace('_proj','')}={v}"
                        for k, v in sorted(
                            changed.items(),
                            key=lambda x: FisherScheduler.layer_index(x[0]),
                        )
                    )
                    print(f"  [step {global_step:5d}] rank update → {rank_str}")
                
                    # Flush accumulated gradients before rebuilding
                    if accum_count > 0:
                        nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], 1.0
                        )
                        optimizer.step()
                        optimizer.zero_grad()
                        opt_step   += 1
                        accum_count = 0
                
                    optimizer, lr_sched = update_optimizer_after_rank_change(
                        model, optimizer, lr_sched, list(changed.keys()),
                        lora_layers, opt_step, total_opt_steps, warmup_lr_steps
                    )
                    optimizer.zero_grad()
                    continue

            # ── Optimizer step when gradient accumulation window is full ────
            if accum_count == GRAD_ACCUM:
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                lr_sched.step()
                optimizer.zero_grad()
                opt_step    += 1
                accum_count  = 0

        # Flush any remaining accumulated gradients at epoch end
        if accum_count > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad()
            opt_step    += 1
            accum_count  = 0

        # ── Epoch-end evaluation ────────────────────────────────────────────
        val  = eval_model(model, val_loader, metric, DEVICE)
        hms  = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
        avg_loss = ep_loss / n_batches

        print(
            f"Ep {epoch+1:2d}/{EPOCHS}  "
            f"train_loss={avg_loss:.4f}  "
            f"val_acc={val['accuracy']*100:.2f}%  "
            f"val_loss={val['loss']:.4f}  "
            f"elapsed={hms}"
        )

        cur_ranks = fisher_sched.current_ranks()
        epoch_results.append({
            "epoch":        epoch + 1,
            "train_loss":   avg_loss,
            "val_accuracy": val["accuracy"],
            "val_loss":     val["loss"],
            "ranks":        {k: int(v) for k, v in cur_ranks.items()},
        })

        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))
            print(f"  ✓ New best: {best_acc*100:.2f}%")

    total_time = time.time() - t0
    train_hms  = time.strftime("%H:%M:%S", time.gmtime(total_time))

    # ── Final report ─────────────────────────────────────────────────────────
    final_ranks = fisher_sched.current_ranks()
    n_trainable_final = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 62)
    print("  FINAL RESULTS — RTE  (DeBERTa-v3 + Dynamic LoRA)")
    print("=" * 62)
    print(f"  Best Accuracy      : {best_acc:.4f}  ({best_acc*100:.2f}%)")
    print(f"  Training time      : {train_hms}  ({total_time:.1f}s)")
    print(f"  Final trainable    : {n_trainable_final:,}  "
          f"({100*n_trainable_final/n_total:.2f}% of total)")
    if DEVICE == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"  Peak GPU memory    : {peak:.1f} MB  ({peak/1024:.2f} GB)")
    elif DEVICE == "mps":
        peak = torch.mps.current_allocated_memory() / 1024 ** 2
        print(f"  Peak MPS memory    : {peak:.1f} MB")
    print("\n  Final rank per layer:")
    for name in sorted(final_ranks.keys()):
        idx  = FisherScheduler.layer_index(name)
        proj = name.split(".")[-1]
        print(f"    L{idx:2d}  {proj:12s}  r = {final_ranks[name]}")
    print("=" * 62)

    # ── Save artifacts ────────────────────────────────────────────────────────
    print("\nSaving artifacts...")

    with open(os.path.join(OUTPUT_DIR, "epoch_results.json"), "w") as f:
        json.dump(epoch_results, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "rank_update_log.json"), "w") as f:
        json.dump(rank_update_log, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "final_summary.json"), "w") as f:
        json.dump({
            "best_accuracy":      best_acc,
            "training_time_s":    total_time,
            "final_ranks":        {k: int(v) for k, v in final_ranks.items()},
            "n_trainable_final":  n_trainable_final,
            "n_total":            n_total,
            "config": {
                "R_INIT": R_INIT, "R_MIN": R_MIN, "R_MAX": R_MAX,
                "EMA_BETA": EMA_BETA, "WARMUP_STEPS": WARMUP_STEPS,
                "UPDATE_INTERVAL": UPDATE_INTERVAL,
            },
        }, f, indent=2)

    # LoRA adapter checkpoint (per-layer)
    torch.save(
        {
            name: {
                "lora_A": layer.lora_A.data.cpu(),
                "lora_B": layer.lora_B.data.cpu(),
                "r":      layer.r,
            }
            for name, layer in lora_layers.items()
        },
        os.path.join(OUTPUT_DIR, "lora_adapters.pt"),
    )
    tokenizer.save_pretrained(OUTPUT_DIR)

    plot_results(fisher_sched, epoch_results, OUTPUT_DIR)
    print(f"\nAll artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
