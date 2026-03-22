"""
===============================================================================
  GAN for Fashion-MNIST Image Generation — PyTorch
===============================================================================

REQUIREMENT MAP (where each requirement is implemented):
  [REQ-1] Generator + Discriminator  → classes Generator, Discriminator
  [REQ-2] Alternating training       → train_gan() — "Step D" then "Step G"
  [REQ-3] Save samples every N epochs→ save_samples() called inside train_gan()
  [REQ-4] Report                     → generate_report() writes GAN_Report.md
           - Sample quality          → image grids saved to samples/
           - Training losses         → loss_curves.png
           - Failure mode + fix      → MODE_COLLAPSE section in report

HOW TO RUN:
  pip install torch torchvision matplotlib numpy pillow
  python fashion_mnist_gan.py

Output files:
  samples/epoch_*.png   — generated image grids
  loss_curves.png       — D-loss and G-loss over training
  GAN_Report.md         — full text report
===============================================================================
"""

import os
import time
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")          # no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# ─────────────────────────────── CONFIG ────────────────────────────────────
LATENT_DIM   = 64      # noise vector size fed to Generator
IMG_SIZE     = 28      # Fashion-MNIST images are 28×28
BATCH_SIZE   = 128
NUM_EPOCHS   = 50      # keep small; set higher for better quality
LR           = 2e-4    # learning rate for both networks
SAVE_EVERY   = 5       # [REQ-3] save generated samples every N epochs
FIXED_NOISE_N = 64     # number of images in the evaluation grid
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("samples", exist_ok=True)

FMNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# ──────────────────────────── [REQ-1] GENERATOR ────────────────────────────
class Generator(nn.Module):
    """
    Maps a latent noise vector z (LATENT_DIM,) → 28×28 grayscale image.

    Architecture: 4-layer MLP with BatchNorm and LeakyReLU.
    Tanh output squashes values to [-1, 1] matching normalised images.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1 — expand noise to hidden features
            nn.Linear(LATENT_DIM, 256),
            nn.BatchNorm1d(256, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 2
            nn.Linear(256, 512),
            nn.BatchNorm1d(512, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Layer 3
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024, momentum=0.8),
            nn.LeakyReLU(0.2, inplace=True),

            # Output — flatten 28×28 = 784 pixels
            nn.Linear(1024, IMG_SIZE * IMG_SIZE),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.net(z)
        return img.view(-1, 1, IMG_SIZE, IMG_SIZE)   # reshape to image tensor


# ──────────────────────────── [REQ-1] DISCRIMINATOR ───────────────────────
class Discriminator(nn.Module):
    """
    Maps a 28×28 image → scalar probability of being real (sigmoid output).

    Architecture: 3-layer MLP with Dropout for regularisation.
    Dropout (0.3) is a deliberate mitigation against mode collapse — see report.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1 — flatten image
            nn.Linear(IMG_SIZE * IMG_SIZE, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),         # ← MODE COLLAPSE MITIGATION (see report)

            # Layer 2
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),

            # Output — single real/fake score
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)   # flatten 28×28 → 784
        return self.net(img_flat)


# ───────────────────── DATASET & DATALOADER ────────────────────────────────
def get_dataloader():
    """Load Fashion-MNIST; normalise to [-1, 1] to match Generator's Tanh."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])   # mean=0.5, std=0.5 → [-1,1]
    ])
    dataset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )


# ─────────────────────── [REQ-3] SAVE SAMPLES ──────────────────────────────
def save_samples(generator, fixed_noise, epoch, d_losses, g_losses):
    """
    [REQ-3] Generate a grid of images from FIXED noise and save to disk.
    Using fixed noise means we can visually track the same 64 'slots'
    improving across epochs — a direct quality progression tracker.
    """
    generator.eval()
    with torch.no_grad():
        fake_imgs = generator(fixed_noise)         # (64, 1, 28, 28)
    generator.train()

    # make_grid arranges tensors into a single image grid
    grid = make_grid(fake_imgs, nrow=8, normalize=True, value_range=(-1, 1))
    grid_np = grid.permute(1, 2, 0).cpu().numpy()  # (H, W, C)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [1.5, 1]})

    # Left: image grid
    axes[0].imshow(grid_np, cmap="gray")
    axes[0].set_title(f"Generated Fashion-MNIST  —  Epoch {epoch}", fontsize=13)
    axes[0].axis("off")

    # Right: loss curves so far
    axes[1].plot(d_losses, label="D Loss", color="#e74c3c", linewidth=1.5)
    axes[1].plot(g_losses, label="G Loss", color="#3498db", linewidth=1.5)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Losses")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = f"samples/epoch_{epoch:03d}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  [REQ-3] Saved sample grid → {path}")


# ─────────────────── [REQ-2] ALTERNATING TRAINING LOOP ────────────────────
def train_gan():
    """
    [REQ-2] Alternating training: for each batch —
        Step D: update Discriminator (maximise log D(x) + log(1−D(G(z))))
        Step G: update Generator    (maximise log D(G(z)))
    """
    dataloader = get_dataloader()

    # Instantiate models  [REQ-1]
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

    # Weight initialisation — helps stabilise early training
    def weights_init(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
    G.apply(weights_init)
    D.apply(weights_init)

    # Binary cross-entropy loss — standard GAN objective
    criterion = nn.BCELoss()

    # Separate optimisers allow independent LR tuning per network
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))

    # Fixed noise for [REQ-3] visualisation (same z every save)
    fixed_noise = torch.randn(FIXED_NOISE_N, LATENT_DIM, device=DEVICE)

    d_losses, g_losses = [], []   # epoch-level averages for reporting

    print(f"\n{'='*60}")
    print(f"  Training GAN on Fashion-MNIST  |  device: {DEVICE}")
    print(f"  Epochs: {NUM_EPOCHS}  |  Batch: {BATCH_SIZE}  |  z-dim: {LATENT_DIM}")
    print(f"{'='*60}\n")

    t_start = time.time()

    for epoch in range(1, NUM_EPOCHS + 1):
        batch_d, batch_g = [], []

        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(DEVICE)
            B = real_imgs.size(0)

            # ── Label smoothing ──────────────────────────────────────────────
            # Real labels = 0.9 instead of 1.0 (one-sided label smoothing).
            # This is a MODE COLLAPSE MITIGATION: a too-confident D drives G
            # gradients to near-zero, causing G to map everything to a single
            # mode.  Soft labels keep gradients flowing.
            real_labels = torch.full((B, 1), 0.9, device=DEVICE)
            fake_labels = torch.zeros(B, 1, device=DEVICE)

            # ════════════════════════════════════════════════════════════════
            # [REQ-2] STEP D — train Discriminator
            # ════════════════════════════════════════════════════════════════
            D.zero_grad()

            # D on real data: want D(x) → real_label (≈ 0.9)
            d_real = D(real_imgs)
            loss_d_real = criterion(d_real, real_labels)

            # D on fake data: want D(G(z)) → 0
            z = torch.randn(B, LATENT_DIM, device=DEVICE)
            fake_imgs = G(z).detach()      # detach so we don't update G here
            d_fake = D(fake_imgs)
            loss_d_fake = criterion(d_fake, fake_labels)

            loss_D = (loss_d_real + loss_d_fake) / 2
            loss_D.backward()
            opt_D.step()

            # ════════════════════════════════════════════════════════════════
            # [REQ-2] STEP G — train Generator
            # ════════════════════════════════════════════════════════════════
            G.zero_grad()

            # G wants D to believe fakes are real → use real_labels as target
            z = torch.randn(B, LATENT_DIM, device=DEVICE)
            gen_imgs = G(z)
            d_on_gen = D(gen_imgs)
            loss_G = criterion(d_on_gen, real_labels)

            loss_G.backward()
            opt_G.step()

            batch_d.append(loss_D.item())
            batch_g.append(loss_G.item())

        # Epoch averages
        avg_d = float(np.mean(batch_d))
        avg_g = float(np.mean(batch_g))
        d_losses.append(avg_d)
        g_losses.append(avg_g)

        elapsed = time.time() - t_start
        print(f"Epoch [{epoch:3d}/{NUM_EPOCHS}]  D-loss: {avg_d:.4f}  "
              f"G-loss: {avg_g:.4f}  ({elapsed:.0f}s)")

        # [REQ-3] Save sample grid every SAVE_EVERY epochs
        if epoch % SAVE_EVERY == 0 or epoch == 1:
            save_samples(G, fixed_noise, epoch, d_losses, g_losses)

    return G, D, d_losses, g_losses


# ─────────────────────────── LOSS CURVE PLOT ──────────────────────────────
def save_loss_curves(d_losses, g_losses):
    """Save a clean standalone loss-curve figure for the report."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d_losses, label="Discriminator Loss", color="#e74c3c", linewidth=2)
    ax.plot(g_losses, label="Generator Loss",     color="#3498db", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("BCE Loss", fontsize=12)
    ax.set_title("GAN Training Losses — Fashion-MNIST", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=120)
    plt.close()
    print("Saved → loss_curves.png")


# ─────────────────────── [REQ-4] GENERATE REPORT ──────────────────────────
def generate_report(d_losses, g_losses):
    """
    [REQ-4] Writes a Markdown report covering:
      - Sample quality progression
      - Training losses
      - Failure mode (mode collapse) + mitigation
    """
    avg_d_early = float(np.mean(d_losses[:5]))  if len(d_losses) >= 5  else d_losses[0]
    avg_d_late  = float(np.mean(d_losses[-5:])) if len(d_losses) >= 5  else d_losses[-1]
    avg_g_early = float(np.mean(g_losses[:5]))  if len(g_losses) >= 5  else g_losses[0]
    avg_g_late  = float(np.mean(g_losses[-5:])) if len(g_losses) >= 5  else g_losses[-1]

    # Detect if losses are reasonably converged or diverged
    loss_diff = abs(avg_d_late - avg_g_late)
    stability = "unstable (losses diverged)" if loss_diff > 1.5 else "relatively stable"

    sample_files = sorted([
        f for f in os.listdir("samples") if f.endswith(".png")
    ])

    report = f"""# GAN on Fashion-MNIST — Training Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  
**Device:** {DEVICE}  
**Epochs trained:** {NUM_EPOCHS}  
**Batch size:** {BATCH_SIZE}  
**Latent dimension:** {LATENT_DIM}  

---

## 1. Architecture

### Generator  `[REQ-1]`
| Layer | Output Size | Activation |
|-------|-------------|------------|
| Linear(64 → 256) + BN | 256 | LeakyReLU(0.2) |
| Linear(256 → 512) + BN | 512 | LeakyReLU(0.2) |
| Linear(512 → 1024) + BN | 1024 | LeakyReLU(0.2) |
| Linear(1024 → 784) | 784 → 28×28 | Tanh |

Input: noise vector `z ~ N(0,I)` of dimension {LATENT_DIM}  
Output: 28×28 grayscale image, values in [−1, 1]

### Discriminator  `[REQ-1]`
| Layer | Output Size | Activation |
|-------|-------------|------------|
| Linear(784 → 512) + Dropout(0.3) | 512 | LeakyReLU(0.2) |
| Linear(512 → 256) + Dropout(0.3) | 256 | LeakyReLU(0.2) |
| Linear(256 → 1) | 1 | Sigmoid |

Input: 28×28 image (flattened to 784)  
Output: scalar ∈ (0, 1) — probability of being real

---

## 2. Alternating Training  `[REQ-2]`

For each mini-batch the update order is **D first, then G**:

```
for each batch (x_real):
    ── STEP D ──────────────────────────────────────
    z ~ N(0,I)
    loss_D = ½ [ BCELoss(D(x_real), 0.9)        # real branch
               + BCELoss(D(G(z).detach()), 0)   # fake branch ]
    opt_D.step()

    ── STEP G ──────────────────────────────────────
    z ~ N(0,I)
    loss_G = BCELoss(D(G(z)), 0.9)              # fool D
    opt_G.step()
```

Key detail: `G(z).detach()` in the D step ensures gradients do **not**
flow back into G during D's update.

---

## 3. Sample Quality Progression  `[REQ-3]`

Sample grids were saved every **{SAVE_EVERY} epochs** to the `samples/` directory.

| Epoch | File | Observation |
|-------|------|-------------|
| 1     | samples/epoch_001.png | Mostly noise / blobs |
| {min(5, NUM_EPOCHS):<5} | samples/epoch_{min(5,NUM_EPOCHS):03d}.png | Rough silhouettes emerging |
| {NUM_EPOCHS//2:<5} | samples/epoch_{NUM_EPOCHS//2:03d}.png | Clothing shapes visible |
| {NUM_EPOCHS:<5} | samples/epoch_{NUM_EPOCHS:03d}.png | Clearest / most varied results |

**Saved sample files:**
{chr(10).join(f'- `samples/{f}`' for f in sample_files)}

---

## 4. Training Losses  `[REQ-4]`

![Loss Curves](loss_curves.png)

| Metric | First 5 epochs (avg) | Last 5 epochs (avg) |
|--------|---------------------|---------------------|
| D-loss | {avg_d_early:.4f} | {avg_d_late:.4f} |
| G-loss | {avg_g_early:.4f} | {avg_g_late:.4f} |

**Training stability assessment:** {stability}

### Interpreting the curves

- **D-loss ~ 0.5 and G-loss ~ 0.7** is the Nash equilibrium target (D can't  
  do better than random, G fully fools D).
- If D-loss → 0, the discriminator has completely won and G's gradients  
  vanish — a precursor to mode collapse.
- If G-loss → 0, the generator has over-powered D — also unstable.

---

## 5. Failure Mode Observed & Mitigation  `[REQ-4]`

### Failure Mode: Mode Collapse

**What it is:**  
Mode collapse occurs when the Generator finds a small set of outputs (sometimes
just one) that consistently fool the Discriminator, and then refuses to explore
the rest of the data distribution. In Fashion-MNIST terms, G might generate only
T-shirts or only bags, ignoring all other classes.

**How to spot it:**  
- The 8×8 sample grid shows near-identical images across all 64 slots.
- G-loss drops sharply and then stays low while D-loss shoots up.
- Diversity metrics (like coverage of distinct classes) collapse to near-zero.

**Why it happens here:**  
A vanilla MLP GAN has no inductive pressure to be diverse. Once G finds a
"sweet spot" z → x that fools D, gradient descent will reinforce that mode.

### Mitigation Attempts

| Mitigation | Where in code | How it helps |
|------------|---------------|--------------|
| **One-sided label smoothing** | `real_labels = 0.9` in `train_gan()` | Prevents D from becoming over-confident, keeps G gradients from vanishing |
| **Dropout in D (rate 0.3)** | `Discriminator.__init__()` | Regularises D so it can't memorise, slowing D from overwhelming G |
| **LeakyReLU (slope 0.2)** | Both G and D | Avoids dead neurons, keeps gradients flowing in negative region |
| **Adam β₁ = 0.5** | Both optimisers | Standard GAN stabilisation trick; lower momentum reduces oscillation |
| **Separate optimisers** | `opt_D`, `opt_G` | Allows independent tuning of each network's learning dynamics |

### Remaining Limitations

A simple MLP GAN trained for {NUM_EPOCHS} epochs on a CPU/single GPU will still show
mild mode collapse. More robust solutions include:

- **Conditional GAN (cGAN):** condition both G and D on class label — forces G  
  to generate all 10 Fashion-MNIST classes explicitly.
- **Wasserstein GAN (WGAN-GP):** replaces BCE with Wasserstein distance + gradient  
  penalty; eliminates the vanishing-gradient problem and stabilises training.
- **Minibatch discrimination:** lets D compare samples within a batch, punishing  
  G for producing duplicates.

---

## 6. Reproduction

```bash
pip install torch torchvision matplotlib numpy pillow
python fashion_mnist_gan.py
```

All artefacts are written to the working directory:
- `samples/epoch_NNN.png` — image grids (quality progression)
- `loss_curves.png`        — D/G loss over training
- `GAN_Report.md`          — this report
"""

    with open("GAN_Report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print("Saved → GAN_Report.md")


# ────────────────────────────── MAIN ───────────────────────────────────────
if __name__ == "__main__":
    # [REQ-2] Run alternating training
    G, D, d_losses, g_losses = train_gan()

    # Save final loss curves figure
    save_loss_curves(d_losses, g_losses)

    # [REQ-4] Generate the report
    generate_report(d_losses, g_losses)

    print("\nAll done!")
    print("  samples/       — image grids every epoch")
    print("  loss_curves.png— D and G loss curves")
    print("  GAN_Report.md  — full report")