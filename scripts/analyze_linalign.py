# src/analyze_linalign.py
"""
Produce paper figures:
  (1) controller_dynamics.pdf   -- from a LinAlign run's history JSON
  (2) linearity_vs_accuracy.pdf -- from a sweep summary JSON

Usage:
  python src/analyze_linalign.py \
      --history results/linalign_history_seed42.json \
      --sweep results/sweep_summary.json \
      --outdir figures/
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_controller_dynamics(history_path, outdir):
    with open(history_path) as f:
        hist = json.load(f)["history"]
    steps = [r["step"] for r in hist]
    lam = [r["lambda"] for r in hist]
    sbar = [r.get("S_bar") for r in hist]
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    ax1.plot(steps, lam, color="tab:blue", label=r"$\lambda_t$")
    ax1.set_xlabel("training step"); ax1.set_ylabel(r"$\lambda_t$", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(steps, sbar, color="tab:red", alpha=0.8, label=r"$\bar{S}_t$")
    ax2.set_ylabel(r"$\bar{S}_t$", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    fig.tight_layout()
    out = os.path.join(outdir, "controller_dynamics.pdf")
    fig.savefig(out, bbox_inches="tight"); print(f"[saved] {out}")


def plot_linearity_vs_accuracy(sweep_path, outdir):
    with open(sweep_path) as f:
        rows = json.load(f)
    r2 = [r["R2"] for r in rows]; acc = [r["accuracy"] for r in rows]
    labels = [r.get("lambda") for r in rows]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.scatter(r2, acc, color="tab:purple")
    for x, y, l in zip(r2, acc, labels):
        ax.annotate(f"$\\lambda={l}$", (x, y), textcoords="offset points",
                    xytext=(4, 4), fontsize=8)
    ax.set_xlabel(r"linearity proxy $R^2$"); ax.set_ylabel("test accuracy (%)")
    try:
        from scipy.stats import spearmanr, pearsonr
        rho, _ = spearmanr(r2, acc); r, _ = pearsonr(r2, acc)
        ax.set_title(f"Spearman $\\rho$={rho:.2f}, Pearson $r$={r:.2f}", fontsize=9)
    except Exception:
        pass
    fig.tight_layout()
    out = os.path.join(outdir, "linearity_vs_accuracy.pdf")
    fig.savefig(out, bbox_inches="tight"); print(f"[saved] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default=None)
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--outdir", default="figures")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    if a.history:
        plot_controller_dynamics(a.history, a.outdir)
    if a.sweep:
        plot_linearity_vs_accuracy(a.sweep, a.outdir)


if __name__ == "__main__":
    main()
