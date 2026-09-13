from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(r"D:\3 Study 学习资料\2E 金融研二(上)资料\BMF5360 Machine Learning in Investments\Group project_2.0")
OUT = ROOT / "_artifact_work" / "figures"
OUT.mkdir(exist_ok=True)
RUN = ROOT / "data/model_runs/ai_pool_daily_state_models_v1_1/20260910T070000000000Z"
BT = ROOT / "data/backtests/ai_portfolio_v1/20260910T133000000000Z"

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
SURFACE, BOXFILL, BOXEDGE = "#ffffff", "#f3f2ef", "#c9c8c2"
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.edgecolor": BOXEDGE,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
})


def box(ax, x, y, w, h, title, body, fill=BOXFILL, edge=BOXEDGE):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                linewidth=0.8, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h - 0.05, title, ha="center", va="top", fontsize=7, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h / 2 - 0.06, body, ha="center", va="center", fontsize=6.3, color=INK2, linespacing=1.3)


def arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8, linewidth=0.8, color=INK2))


def figure_pipeline():
    fig, ax = plt.subplots(figsize=(6.27, 1.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    w, gap, y, h = 0.176, 0.03, 0.42, 0.5
    xs = [i * (w + gap) for i in range(5)]
    box(ax, xs[0], y, w, h, "Point-in-time universe", "781 S&P 500 RICs\n154 delisted kept\n49-company pilot pool")
    box(ax, xs[1], y, w, h, "Factor library", "Market, macro,\nindustry state\nall through F-1")
    box(ax, xs[2], y, w, h, "Stock-day panel", "110,829 company-days\nH21 label vs SPY\nnon-overlapping anchors")
    box(ax, xs[3], y, w, h, "Models", "Logistic vs\nRandom Forest\npurged training folds")
    box(ax, xs[4], y, w, h, "Portfolio rule", "Pre-registered caps,\nhedge and costs\nindependent audit")
    for i in range(4):
        arrow(ax, xs[i] + w + 0.003, xs[i + 1] - 0.003, y + h / 2)
    ty, th = 0.06, 0.2
    spans = [(0.0, 0.5, "Training 2015–2020", BOXFILL, None),
             (0.5, 0.7, "Validation 2021–22", "#e4ecf8", None),
             (0.7, 1.0, "Test 2023-01 to 2026-06, sealed", "#ffffff", "////")]
    for x0, x1, label, fill, hatch in spans:
        ax.add_patch(Rectangle((x0, ty), x1 - x0, th, linewidth=0.8, edgecolor=BOXEDGE, facecolor=fill, hatch=hatch))
        ax.text((x0 + x1) / 2, ty + th / 2, label, ha="center", va="center", fontsize=6.5, color=INK)
    ax.text(0.0, ty + th + 0.03, "Samples never shuffled; every feature dated before the decision session", ha="left", va="bottom", fontsize=6.5, color=MUTED)
    fig.savefig(OUT / "fig1_pipeline.png", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def figure_quintiles():
    vp = pd.read_csv(RUN / "validation_predictions.csv")
    col = "p_up_technical_plus_ai_state_logistic"
    q = pd.qcut(vp[col].rank(method="first"), 5, labels=False)
    means = vp.groupby(q)["forward_excess_return"].mean().values * 100
    counts = vp.groupby(q).size().values
    fig, ax = plt.subplots(figsize=(6.27, 1.8))
    xs = np.arange(5)
    ax.grid(axis="y", color="#ececea", linewidth=0.6, zorder=0)
    bars = ax.bar(xs, means, 0.55, color=BLUE, linewidth=0, zorder=3)
    for b, v, n in zip(bars, means, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + (0.12 if v >= 0 else -0.12), f"{v:+.2f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7, color=INK2)
    ax.axhline(0, color=BOXEDGE, linewidth=0.8, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Q1 (lowest p)\nn={counts[0]}", f"Q2\nn={counts[1]}", f"Q3\nn={counts[2]}", f"Q4\nn={counts[3]}", f"Q5 (highest p)\nn={counts[4]}"])
    ax.set_ylabel("Mean 21-session excess\nreturn vs SPY (%)")
    ax.set_ylim(-3.3, 2.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(OUT / "fig2_quintiles.png", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("primary quintile means %", np.round(means, 2).tolist())


def figure_nav():
    nav = pd.read_csv(BT / "nav_daily.csv", parse_dates=["Date"])
    nav["spy_index"] = (1 + nav["spy_return"]).cumprod()
    pt = pd.read_csv(BT / "portfolio_targets.csv")
    held_dates = sorted(pd.to_datetime(pt.loc[pt["selected_for_target"] == True, "formation_session"].unique()))
    all_dates = sorted(pd.to_datetime(pt["formation_session"].unique()))
    skipped = [d for d in all_dates if d not in held_dates]
    fig, ax = plt.subplots(figsize=(6.27, 2.1))
    # shade extension windows: from a skipped rebalance date to the next executed rebalance
    first = True
    for d in skipped:
        nxt = min([h for h in held_dates if h > d], default=nav["Date"].max())
        ax.axvspan(d, nxt, color="#f3e2dc", linewidth=0, label="Skipped rebalance, positions held on" if first else None)
        first = False
    ax.plot(nav["Date"], nav["net_nav"], color=BLUE, linewidth=1.4, label="Hedged portfolio, net of spread costs")
    ax.plot(nav["Date"], nav["spy_index"], color=ORANGE, linewidth=1.4, label="SPY")
    ax.axhline(1.0, color=BOXEDGE, linewidth=0.8)
    ax.vlines(held_dates, 0.83, 0.85, color=MUTED, linewidth=0.6)
    ax.text(held_dates[0], 0.858, "executed rebalances", fontsize=6.5, color=MUTED, va="bottom")
    for col, lab, colr in (("net_nav", "0.97", BLUE), ("spy_index", "1.13", ORANGE)):
        ax.text(nav["Date"].iloc[-1], nav[col].iloc[-1], "  " + lab, fontsize=7.5, color=colr, va="center")
    ax.set_ylim(0.82, 1.32)
    ax.set_ylabel("Growth of 1")
    ax.grid(axis="y", color="#ececea", linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax.margins(x=0.01)
    fig.savefig(OUT / "fig3_nav.png", dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("skipped", [d.date() for d in skipped])


if __name__ == "__main__":
    figure_pipeline()
    figure_quintiles()
    figure_nav()
    print("figures written to", OUT)
