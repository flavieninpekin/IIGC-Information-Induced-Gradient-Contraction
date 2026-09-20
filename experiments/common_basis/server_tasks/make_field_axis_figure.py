"""Figure: field-axis comparison across Toy / Overcooked / 510K.

Scatter: x = kappa_mean(reinforce field), y = kappa_mean(value/soft field).
y=x diagonal separates "value survives" (above) from "fields agree" (on line).
Points: filled = hidden condition, hollow = visible condition; color = env.
The figure plots reinforce against value; the differentiable-baseline awr
field is reported in the paper table, not here.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RES = os.path.join(ROOT, "data", "kappa", "server_tasks", "results")
OUTDIR = os.path.join(ROOT, "paper", "figures")
os.makedirs(OUTDIR, exist_ok=True)


def oc_field(mode, field, key="kappa_mix"):
    o = json.load(open(os.path.join(RES, "oc_field_axis.json")))
    ks = []
    for s in (41, 44, 48):
        entry = o[f"{mode}_s{s}_{field}"]
        if key in entry:
            ks.append(entry[key])
        else:
            mu2, vc = entry["E_shared"], entry["E_contrast"]
            ks.append(mu2 / (mu2 + vc) if (mu2 + vc) > 0 else 0.0)
    return float(np.mean(ks)), float(np.std(ks))


def s510k(p, field, key="kappa_mix"):
    d = json.load(open(os.path.join(RES, "510k_field_axis.json")))
    ks = []
    for s in range(41, 47):
        level = d.get(f"{p:.1f}", {})
        entry = level.get(f"s{s}_paired")
        if entry is None:
            entry = level.get(f"s{s}", {})
        r = entry.get(field, {})
        if key in r:
            ks.append(r[key])
    return float(np.mean(ks)), float(np.std(ks))


def toy():
    """Canonical Toy anchors.

    Hidden: exact reinforce cancellation (0.0); the soft field anchor is the
    canonical softq closed form at alpha=10 (0.903). Revealed: all fields
    degenerate to ~0.44 under the controlled same-policy protocol
    (`toy_fields/visibility_control.json`).
    """
    return (0.0, 0.903), (0.44, 0.44)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)

    # diagonal
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.2, zorder=1,
            label="fields equal (y=x)")

    colors = {"Toy": "#2ca02c", "Overcooked": "#1f77b4", "510K": "#ff7f0e"}
    hidden_marker, visible_marker = "o", "s"

    # ---- Toy (canonical anchors) ----
    (rf_h, val_h), (rf_r, val_r) = toy()
    ax.scatter([rf_h], [val_h], marker=hidden_marker, s=90, color=colors["Toy"],
               edgecolor="k", zorder=5)
    ax.scatter([rf_r], [val_r], marker=visible_marker, s=80, facecolor="none",
               edgecolor=colors["Toy"], linewidths=2, zorder=5)
    ax.annotate("Toy (mirror anchor)", xy=(rf_r + 0.015, val_r - 0.01),
                fontsize=9, color=colors["Toy"])

    # ---- Overcooked ----
    for mode, lbl, dx, dy in [("static", "Overcooked static", 0.015, -0.045),
                              ("dynamic", "Overcooked dynamic", -0.30, 0.005)]:
        rx, rs = oc_field(mode, "reinforce")
        vx, vs = oc_field(mode, "value")
        m = hidden_marker if mode == "dynamic" else visible_marker
        if mode == "dynamic":
            ax.scatter([rx], [vx], marker=m, s=90, color=colors["Overcooked"],
                       edgecolor="k", zorder=5)
            ax.errorbar([rx], [vx], xerr=[rs], yerr=[vs], fmt="none",
                        ecolor=colors["Overcooked"], zorder=4)
        else:
            ax.scatter([rx], [vx], marker=m, s=80, facecolor="none",
                       edgecolor=colors["Overcooked"], linewidths=2, zorder=5)
        ax.annotate(lbl, xy=(rx + dx, vx + dy), fontsize=8,
                    color=colors["Overcooked"])

    # ---- 510K ----
    for p, lbl, dy in [
        (0.0, "510K p=0 (hidden)", 0.012),
        (1.0, "510K p=1 (visible)", -0.025),
    ]:
        rx, rs = s510k(p, "reinforce")
        vx, vs = s510k(p, "value")
        m = hidden_marker if p == 0.0 else visible_marker
        if p == 0.0:
            ax.scatter([rx], [vx], marker=m, s=90, color=colors["510K"],
                       edgecolor="k", zorder=5)
            ax.errorbar([rx], [vx], xerr=[rs], yerr=[vs], fmt="none",
                        ecolor=colors["510K"], zorder=4)
        else:
            ax.scatter([rx], [vx], marker=m, s=80, facecolor="none",
                       edgecolor=colors["510K"], linewidths=2, zorder=5)
        ax.annotate(lbl, xy=(rx + 0.02, vx + dy), fontsize=8,
                    color=colors["510K"], va="bottom" if dy > 0 else "top",
                    ha="left")

    ax.set_xlabel(r"$\kappa_{\mathrm{mix}}$ (reinforce field)", fontsize=11)
    ax.set_ylabel(r"$\kappa_{\mathrm{mix}}$ (value / soft field)", fontsize=11)
    ax.set_title("Structural field axis: value fields align with hidden conditions,\n"
                 "policy-gradient fields stay near-orthogonal (kappa_mix, condition means)",
                 fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.25, zorder=0)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="gray", linestyle="--", lw=1.2, label="fields equal (y=x)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="k", markersize=8,
               label="hidden"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none",
               markeredgecolor="k", markersize=8, label="visible"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "field_axis_comparison.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "field_axis_comparison.pdf"), bbox_inches="tight")
    print("saved", os.path.join(OUTDIR, "field_axis_comparison.png"))
    print("points:")
    print("  Toy hidden:   (%.3f, %.3f)" % toy()[0])
    print("  Toy revealed: (%.3f, %.3f)" % toy()[1])
    print("  OC static:    (%.3f, %.3f)" % (oc_field("static", "reinforce")[0],
                                            oc_field("static", "value")[0]))
    print("  OC dynamic:   (%.3f, %.3f)" % (oc_field("dynamic", "reinforce")[0],
                                            oc_field("dynamic", "value")[0]))
    print("  510K p=0:     (%.3f, %.3f)" % (s510k(0.0, "reinforce")[0], s510k(0.0, "value")[0]))
    print("  510K p=1:     (%.3f, %.3f)" % (s510k(1.0, "reinforce")[0], s510k(1.0, "value")[0]))


if __name__ == "__main__":
    main()
