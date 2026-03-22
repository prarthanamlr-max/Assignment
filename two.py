"""
=============================================================================
RNN Sequence Learning: Text Classification (Sentiment Analysis)
PyTorch implementation
=============================================================================
Requirements covered:
  REQ 1 → Plain RNN baseline          (class VanillaRNN)
  REQ 2 → LSTM and GRU versions       (class LSTMModel, class GRUModel)
  REQ 3 → Comparison on:
            - Accuracy                (evaluate() returns accuracy %)
            - Training stability      (loss curves saved in report PNG)

Install dependencies:
    pip install torch matplotlib numpy
=============================================================================
"""

import re
import time
import random
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 – TINY DATASET
#   100 hand-crafted sentences (50 positive, 50 negative).
#   No downloads required — everything is embedded in the script.
#   Swap this section with real IMDB data for production use.
# ─────────────────────────────────────────────────────────────────────────────

POSITIVE = [
    "this film was absolutely wonderful and I loved every moment",
    "a masterpiece of storytelling with stunning visuals and great acting",
    "the best movie I have seen in years truly breathtaking",
    "incredible performances from the entire cast brilliant direction",
    "a heartwarming story that left me smiling for hours",
    "outstanding cinematography combined with a compelling narrative",
    "I was completely captivated from start to finish amazing work",
    "the script is witty and the characters are deeply likable",
    "a delightful experience that the whole family can enjoy",
    "one of the most moving films I have ever watched beautiful",
    "funny clever and surprisingly touching highly recommend this film",
    "the director has outdone themselves this is a true gem",
    "superb acting and a gripping plot kept me on the edge of my seat",
    "a refreshing and original take on a classic theme loved it",
    "emotional powerful and beautifully shot this is cinema at its best",
    "uplifting and inspirational a must watch for everyone",
    "the chemistry between the leads is electric wonderful film",
    "flawless pacing great music and an unforgettable ending",
    "a triumph of independent filmmaking genuinely moving",
    "charming funny and full of heart this exceeded all my expectations",
    "the acting is phenomenal and the story is deeply touching indeed",
    "a joyful cinematic experience I will remember for a very long time",
    "witty dialogue and superb performances make this a modern classic",
    "visually stunning and emotionally resonant highly recommended film",
    "a perfect blend of humor and heart truly a very special film",
    "remarkable storytelling that stays with you long after watching",
    "the best performance of the year from the lead actor brilliant",
    "a beautifully crafted film with genuine emotional depth throughout",
    "smart funny and wonderfully acted this is top tier cinema",
    "an absolute delight from opening scene to closing credits spectacular",
    "this movie restored my faith in modern storytelling fantastic work",
    "gripping tense and ultimately very rewarding great film indeed",
    "one of those rare films that is both entertaining and meaningful",
    "the soundtrack perfectly complements the gorgeous imagery wonderful",
    "a bravura piece of filmmaking that demands to be seen immediately",
    "warm funny and full of life exactly what cinema should be like",
    "the ensemble cast delivers career best work truly extraordinary",
    "a generous and big hearted film that earns every single tear",
    "clever inventive and genuinely surprising in the very best way",
    "touching story brilliantly told this is exactly what movies are for",
    "loved every second of it a genuine crowd pleaser for everyone",
    "the director balances comedy and drama with real impressive skill",
    "a deeply satisfying film that fires on all cylinders brilliantly",
    "beautiful performances and a script full of sharp genuine insights",
    "this is exactly the kind of film the world needs right now",
    "the kind of movie that reminds you why you love cinema so much",
    "pitch perfect casting and a wonderfully imaginative original story",
    "an emotionally rich film that resonates long after viewing it",
    "this was a thoroughly enjoyable and genuinely uplifting experience",
    "a confident and moving film that deserves the widest recognition",
]

NEGATIVE = [
    "this film was absolutely terrible and I hated every moment",
    "a complete waste of time with poor acting and a dull script",
    "the worst movie I have seen in years truly disappointing indeed",
    "dreadful performances from the cast and no direction whatsoever",
    "a boring story that left me falling asleep within mere minutes",
    "terrible cinematography combined with a completely nonsensical plot",
    "I was completely bored from start to finish truly awful work",
    "the script is lazy and the characters are deeply unlikable indeed",
    "a painful experience that I do not recommend to anyone at all",
    "one of the most tedious films I have ever had to sit through",
    "unfunny awkward and surprisingly dull avoid at all costs always",
    "the director has failed completely this is a total disaster film",
    "poor acting and a confusing plot made this completely unbearable",
    "a tired and unoriginal take on a classic theme hated every bit",
    "flat lifeless and poorly shot this is cinema at its very worst",
    "depressing and uninspiring a film to skip at all costs forever",
    "the leads have no chemistry whatsoever truly terrible dull film",
    "clunky pacing forgettable music and a completely nonsensical ending",
    "a failure of independent filmmaking that is genuinely awful indeed",
    "joyless dull and empty this fell far short of all my expectations",
    "the acting is wooden and the story is painfully predictable indeed",
    "a miserable cinematic experience I truly want my two hours back",
    "stilted dialogue and poor performances make this entirely forgettable",
    "visually ugly and emotionally hollow please do not bother watching",
    "a terrible blend of failed humor and completely false sentiment",
    "lazy storytelling that disappears from memory almost immediately",
    "the worst performance of the year from the lead actor terrible",
    "a sloppily made film with absolutely no emotional depth whatsoever",
    "dumb unfunny and poorly acted this is bottom tier bad cinema",
    "a complete disappointment from the opening to the closing credits",
    "this movie destroyed my faith in modern filmmaking truly awful",
    "slow tedious and ultimately very unrewarding dull terrible film",
    "one of those rare films that manages to be both boring and bad",
    "the soundtrack is jarring and all the imagery is really ugly",
    "a clumsy piece of filmmaking that should never have been made",
    "cold unfunny and lifeless exactly what cinema should never be",
    "the ensemble cast delivers career worst work truly atrocious bad",
    "a mean spirited and small minded film that earns absolutely no praise",
    "derivative predictable and genuinely boring in the absolute worst way",
    "tedious story badly told this is exactly what bad movies look like",
    "hated every second of it a genuine crowd repeller for everyone",
    "the director fumbles both comedy and drama very badly throughout",
    "a deeply unsatisfying film that misfires on absolutely all cylinders",
    "wooden performances and a script full of tired overused cliches",
    "this is exactly the kind of film the world does not need ever",
    "the kind of movie that reminds you why you hate bad cinema",
    "miscast characters and a completely unimaginative and boring story",
    "an emotionally hollow film that fades from memory almost instantly",
    "this was a thoroughly unpleasant and deeply depressing experience",
    "an uncertain and unconvincing film that deserves no recognition",
]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 – TOKENISATION & VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────

def simple_tokenise(text):
    """Lower-case, strip punctuation, split on whitespace."""
    return re.sub(r"[^a-z\s]", "", text.lower()).split()

def build_vocab(sentences):
    """
    Build word-to-index mapping from training sentences.
    Index 0 = <pad>  (used for padding shorter sequences)
    Index 1 = <unk>  (used for words not seen during training)
    """
    counter = collections.Counter()
    for s in sentences:
        counter.update(simple_tokenise(s))
    vocab = {"<pad>": 0, "<unk>": 1}
    for word, _ in counter.most_common():
        vocab[word] = len(vocab)
    return vocab

def encode(sentence, vocab, max_len=20):
    """Convert a sentence string to a padded/truncated integer list."""
    tokens = simple_tokenise(sentence)[:max_len]
    ids    = [vocab.get(t, 1) for t in tokens]          # 1 = <unk>
    ids   += [0] * (max_len - len(ids))                  # 0 = <pad>
    return ids


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 – PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SentimentDataset(Dataset):
    """
    Wraps encoded sentences and binary labels into a PyTorch Dataset.
    Label 1 = positive, 0 = negative.
    """
    def __init__(self, texts, labels, vocab, max_len=20):
        self.X = torch.tensor(
            [encode(t, vocab, max_len) for t in texts], dtype=torch.long
        )
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 – MODEL DEFINITIONS
#   REQ 1: VanillaRNN  – plain recurrent network (baseline)
#   REQ 2: LSTMModel   – Long Short-Term Memory
#   REQ 2: GRUModel    – Gated Recurrent Unit
#
#   Architecture for all three:
#     Embedding → Recurrent Layer → Last Hidden State → Dropout → Linear(1)
# ─────────────────────────────────────────────────────────────────────────────

class VanillaRNN(nn.Module):
    """
    REQ 1 – RNN Baseline.
    Uses nn.RNN: h_t = tanh(Wxh·xₜ + Whh·h_{t-1} + b)
    No gating; prone to vanishing gradients on long sequences.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # REQ 1: plain RNN cell
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded         = self.dropout(self.embedding(x))   # (B, T, E)
        _, hidden        = self.rnn(embedded)                # hidden: (layers, B, H)
        last_hidden      = self.dropout(hidden[-1])          # take top layer
        return self.fc(last_hidden).squeeze(1)               # (B,)


class LSTMModel(nn.Module):
    """
    REQ 2a – LSTM.
    Uses nn.LSTM: adds a cell state c_t and three gates (input, forget, output).
    Resolves the vanishing gradient problem of plain RNN.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # REQ 2a: LSTM cell
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded            = self.dropout(self.embedding(x))
        _, (hidden, _cell)  = self.lstm(embedded)     # unpack (h_n, c_n)
        last_hidden         = self.dropout(hidden[-1])
        return self.fc(last_hidden).squeeze(1)


class GRUModel(nn.Module):
    """
    REQ 2b – GRU.
    Uses nn.GRU: merges cell state into hidden state via update & reset gates.
    Fewer parameters than LSTM; usually trains faster.
    """
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # REQ 2b: GRU cell
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        embedded    = self.dropout(self.embedding(x))
        _, hidden   = self.gru(embedded)
        last_hidden = self.dropout(hidden[-1])
        return self.fc(last_hidden).squeeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 – TRAINING & EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimiser, criterion, device):
    """One full pass over training data. Returns average loss."""
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimiser.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        # Gradient clipping — critical for training stability (REQ 3b)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    """
    REQ 3a – Performance metric: Accuracy.
    Returns (avg_loss, accuracy_percent).
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds     = model(X_batch)
            loss      = criterion(preds, y_batch)
            total_loss += loss.item()
            predicted  = (torch.sigmoid(preds) >= 0.5).float()
            correct   += (predicted == y_batch).sum().item()
            total     += y_batch.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def train_model(model, train_loader, val_loader, epochs, lr, device, name):
    """
    Full training loop.
    REQ 3b – records per-epoch train/val loss for stability comparison.
    Returns history dict and elapsed time.
    """
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    # Reduce LR if val loss plateaus — improves stability
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss        = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_loss, acc  = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{name:8s}] Epoch {epoch:3d}/{epochs} | "
                  f"Train: {tr_loss:.4f} | Val: {val_loss:.4f} | Acc: {acc:.1f}%")

    elapsed = time.time() - t0
    _, final_acc = evaluate(model, val_loader, criterion, device)
    print(f"  [{name:8s}] Finished in {elapsed:.1f}s — Final Acc: {final_acc:.2f}%")
    return history, elapsed, final_acc


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 – REPORT GENERATION
#   REQ 3 – 4-panel PNG comparing all three models
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(results, epochs, output_path="rnn_comparison_report.png"):
    """
    Saves a 4-panel comparison figure:
      Top-left  → Training loss curves     (REQ 3b: stability)
      Top-right → Validation loss curves   (REQ 3b: stability)
      Bot-left  → Validation accuracy      (REQ 3a: performance)
      Bot-right → Final accuracy bar chart + summary table
    """
    colours = {"RNN": "#e74c3c", "LSTM": "#2ecc71", "GRU": "#3498db"}
    x = list(range(1, epochs + 1))

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)
    axs = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    def style_ax(ax, title, xlabel, ylabel):
        ax.set_facecolor("#16213e")
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=9)
        ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=9)
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444466")
        ax.grid(color="#2a2a4a", linestyle="--", linewidth=0.5)

    # Top-left — training loss (REQ 3b)
    style_ax(axs[0], "REQ 3b  Training Loss per Epoch", "Epoch", "BCEWithLogits Loss")
    for name, r in results.items():
        axs[0].plot(x, r["history"]["train_loss"],
                    color=colours[name], label=name, linewidth=2)
    axs[0].legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # Top-right — validation loss (REQ 3b)
    style_ax(axs[1], "REQ 3b  Validation Loss per Epoch", "Epoch", "BCEWithLogits Loss")
    for name, r in results.items():
        axs[1].plot(x, r["history"]["val_loss"],
                    color=colours[name], label=name, linewidth=2)
    axs[1].legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # Bot-left — accuracy per epoch (REQ 3a)
    style_ax(axs[2], "REQ 3a  Validation Accuracy per Epoch", "Epoch", "Accuracy (%)")
    for name, r in results.items():
        axs[2].plot(x, r["history"]["val_acc"],
                    color=colours[name], label=name, linewidth=2)
    axs[2].set_ylim(0, 108)
    axs[2].legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    # Bot-right — final accuracy bar chart (REQ 3a)
    axs[3].set_facecolor("#16213e")
    axs[3].set_title("REQ 3a  Final Accuracy & Summary", color="white", fontsize=11, pad=8)
    names      = list(results.keys())
    final_accs = [results[n]["final_acc"] for n in names]

    bars = axs[3].bar(names, final_accs,
                      color=[colours[n] for n in names],
                      width=0.45, edgecolor="white", linewidth=0.7)
    axs[3].set_ylim(0, 120)
    axs[3].set_ylabel("Test Accuracy (%)", color="#aaaaaa", fontsize=9)
    axs[3].tick_params(colors="white", labelsize=10)
    for sp in axs[3].spines.values():
        sp.set_edgecolor("#444466")
    axs[3].grid(axis="y", color="#2a2a4a", linestyle="--", linewidth=0.5)

    for bar, acc in zip(bars, final_accs):
        axs[3].text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.5, f"{acc:.1f}%",
                    ha="center", va="bottom",
                    color="white", fontsize=11, fontweight="bold")

    # Summary table below bars
    rows = ["Model    Acc(%)  Time(s)  Params  LossVar",
            "-" * 44]
    for n in names:
        r  = results[n]
        lv = float(np.var(r["history"]["train_loss"]))
        rows.append(f"{n:<8} {r['final_acc']:>5.1f}  "
                    f"{r['time']:>6.1f}  {r['params']:>6}  {lv:.5f}")
    axs[3].text(0.5, -0.30, "\n".join(rows),
                transform=axs[3].transAxes, ha="center", va="top",
                fontsize=8, color="white", fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.4",
                          facecolor="#0f3460", edgecolor="#444466"))

    fig.suptitle(
        "RNN vs LSTM vs GRU  —  Sentiment Text Classification (PyTorch)",
        color="white", fontsize=13, fontweight="bold", y=1.01
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Report saved  →  {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 – MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── hyperparameters ──────────────────────────────────────────────────────
    EMBED_DIM  = 32
    HIDDEN_DIM = 64
    N_LAYERS   = 1
    DROPOUT    = 0.3
    BATCH_SIZE = 16
    EPOCHS     = 50
    LR         = 1e-3
    MAX_LEN    = 20
    DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 62)
    print("  RNN Sequence Learning — Sentiment Text Classification")
    print("  PyTorch  |  RNN vs LSTM vs GRU")
    print("=" * 62)
    print(f"  Device  : {DEVICE}")
    print(f"  Embed   : {EMBED_DIM}   Hidden : {HIDDEN_DIM}   "
          f"Layers : {N_LAYERS}")
    print(f"  Epochs  : {EPOCHS}   Batch : {BATCH_SIZE}   LR : {LR}")

    # ── dataset ──────────────────────────────────────────────────────────────
    all_texts  = POSITIVE + NEGATIVE
    all_labels = [1] * len(POSITIVE) + [0] * len(NEGATIVE)

    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)

    split = int(0.8 * len(all_texts))
    tr_texts,  val_texts  = all_texts[:split],  all_texts[split:]
    tr_labels, val_labels = all_labels[:split], all_labels[split:]

    vocab      = build_vocab(tr_texts)
    VOCAB_SIZE = len(vocab)

    print(f"\n  Dataset : {len(all_texts)} samples  "
          f"(train={len(tr_texts)}, val={len(val_texts)})")
    print(f"  Vocab   : {VOCAB_SIZE} unique tokens")

    # ── data loaders ─────────────────────────────────────────────────────────
    train_ds     = SentimentDataset(tr_texts,  tr_labels,  vocab, MAX_LEN)
    val_ds       = SentimentDataset(val_texts, val_labels, vocab, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # ── model configs ─────────────────────────────────────────────────────────
    model_kwargs = dict(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    )

    configs = [
        ("RNN",  VanillaRNN),    # REQ 1 — baseline
        ("LSTM", LSTMModel),     # REQ 2a
        ("GRU",  GRUModel),      # REQ 2b
    ]

    # ── train ─────────────────────────────────────────────────────────────────
    results = {}
    for name, ModelClass in configs:
        print(f"\n{'─'*62}\n  Training  {name}\n{'─'*62}")
        model    = ModelClass(**model_kwargs).to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable parameters: {n_params:,}")

        history, elapsed, final_acc = train_model(
            model, train_loader, val_loader, EPOCHS, LR, DEVICE, name
        )
        results[name] = {
            "history":   history,
            "final_acc": final_acc,
            "time":      elapsed,
            "params":    n_params,
        }

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  FINAL COMPARISON  (REQ 3)")
    print(f"{'='*62}")
    print(f"  {'Model':<8} {'Val Acc%':>9} {'Time(s)':>9} "
          f"{'Params':>8} {'Loss Variance':>14}")
    print(f"  {'─'*8} {'─'*9} {'─'*9} {'─'*8} {'─'*14}")
    for name, r in results.items():
        lv = float(np.var(r["history"]["train_loss"]))
        print(f"  {name:<8} {r['final_acc']:>9.2f} "
              f"{r['time']:>9.1f} {r['params']:>8,} {lv:>14.6f}")

    # REQ 3b — training stability note
    print(f"\n  Training-stability note (REQ 3b):")
    print(f"  Lower loss variance = smoother, more stable training.")
    for name, r in results.items():
        lv = float(np.var(r["history"]["train_loss"]))
        print(f"    {name:<8}  train-loss variance = {lv:.6f}")

    # ── generate visual report ────────────────────────────────────────────────
    generate_report(results, EPOCHS, "rnn_comparison_report.png")
    print("\n  Done. Open rnn_comparison_report.png for the full chart.")


if __name__ == "__main__":
    main()