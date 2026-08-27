"""Figure: field-axis comparison across Toy / Overcooked / 510K.

Scatter: x = kappa_mean(reinforce field), y = kappa_mean(value/soft field).
y=x diagonal separates "value survives" (above) from "fields agree" (on line).
Points: filled = hidden condition, hollow = visible condition; color = env.
Arrow from visible to hidden within each env shows hidden strengthens the gap.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = r"C:\Users\Flavi\opencode\IIGC\data\kappa\server_tasks\results"
OUTDIR = r"C:\Users\Flavi\opencode\IIGC\paper\figures"
os.makedirs(OUTDIR, exist_ok=True)


def oc_field(mode, field, key):
    o = json.load(open(os.path.join(RES, "oc_field_axis.json")))
    ks = [o[f"{mode}_s{s}_{field}"][key] for s in (41, 44, 48)]
    return float(np.mean(ks)), float(np.std(ks))


def s510k(p, field, key="kappa_mean"):
    d = json.load(open(os.path.join(RES, "510k_field_axis.json")))
    ks = []
    for s in range(41, 47):
        r = d[f"{p:.1f}"][f"s{s}"][field]
        if key == "kappa_mean":
            mu2, vb = r["mu2"], r["var_between"]
            ks.append(mu2 / (mu2 + vb) if (mu2 + vb) > 0 else 0.0)
        else:
            ks.append(r[key])
    return float(np.mean(ks)), float(np.std(ks))


def toy():
    t = json.load(open(r"C:\Users\Flavi\opencode\IIGC\data\kappa\toy_fields\theory_toy.json"))
    # exact anchors: reinforce hidden -> 0 (g_B=-g_A); softmaxq tau=10 -> 0.898
    rf_h, val_h = 0.0, 0.898
    rf_r = t["REVEALED"]["reinforce"]["kappa_mean"]
    val_r = t["REVEALED"]["softq"]["kappa_mean"]
    return (rf_h, val_h), (rf_r, val_r)


def main():
    fig, ax = plt.subplots(figsize=(7.2, 6.0), dpi=150)

    # diagonal
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.2, zorder=1,
            label="fields equal (y=x)")

    colors = {"Toy": "#2ca02c", "Overcooked": "#1f77b4", "510K": "#ff7f0e"}
    hidden_marker, visible_marker = "o", "s"

    # ---- Toy (exact anchors) ----
    (rf_h, val_h), (rf_r, val_r) = toy()
    ax.scatter([rf_h], [val_h], marker=hidden_marker, s=90, color=colors["Toy"],
               edgecolor="k", zorder=5)
    ax.scatter([rf_r], [val_r], marker=visible_marker, s=80, facecolor="none",
               edgecolor=colors["Toy"], linewidths=2, zorder=5)
    ax.annotate("Toy", xy=(rf_r - 0.02, val_r + 0.03), color=colors["Toy"], fontsize=10)
    ax.annotate("hidden", xy=(rf_h + 0.01, val_h + 0.02), color=colors["Toy"], fontsize=8)

    # ---- Overcooked ----
    for mode, lbl in [("static", "Overcooked static"), ("dynamic", "Overcooked dynamic")]:
        rx, rs = oc_field(mode, "reinforce", "kappa_mean")
        vx, vs = oc_field(mode, "value", "kappa_mean")
        m = hidden_marker if mode == "dynamic" else visible_marker
        if mode == "dynamic":
            ax.scatter([rx], [vx], marker=m, s=90, color=colors["Overcooked"],
                       edgecolor="k", zorder=5)
            ax.errorbar([rx], [vx], xerr=[rs], yerr=[vs], fmt="none", ecolor=colors["Overcooked"], zorder=4)
        else:
            ax.scatter([rx], [vx], marker=m, s=80, facecolor="none",
                       edgecolor=colors["Overcooked"], linewidths=2, zorder=5)
        ax.annotate(lbl, xy=(rx, vx), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color=colors["Overcooked"])

    # ---- 510K ----
    for p, lbl, txtpos, arrow in [
        (0.0, "510K p=0 (hidden)", (0.615, 1.035), (0.545, 0.978)),
        (1.0, "510K p=1 (visible)", (0.40, 1.035), (0.468, 0.990)),
    ]:
        rx, rs = s510k(p, "reinforce")
        vx, vs = s510k(p, "value")
        m = hidden_marker if p == 0.0 else visible_marker
        if p == 0.0:
            ax.scatter([rx], [vx], marker=m, s=90, color=colors["510K"],
                       edgecolor="k", zorder=5)
            ax.errorbar([rx], [vx], xerr=[rs], yerr=[vs], fmt="none", ecolor=colors["510K"], zorder=4)
        else:
            ax.scatter([rx], [vx], marker=m, s=80, facecolor="none",
                       edgecolor=colors["510K"], linewidths=2, zorder=5)
        ax.annotate(lbl, xy=arrow, xytext=txtpos, fontsize=8, color=colors["510K"],
                    arrowprops=dict(arrowstyle="-", color=colors["510K"], lw=1.0),
                    ha="center")

    ax.set_xlabel(r"$\kappa$ (reinforce / hard policy-gradient field)", fontsize=11)
    ax.set_ylabel(r"$\kappa$ (value / mean-seeking field)", fontsize=11)
    ax.set_title("Field axis: value survives where policy gradient contracts\n"
                 "(same network & data, only the objective changes)", fontsize=11)
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
    print("  OC static:    (%.3f, %.3f)" % (oc_field("static", "reinforce", "kappa_mean")[0],
                                            oc_field("static", "value", "kappa_mean")[0]))
    print("  OC dynamic:   (%.3f, %.3f)" % (oc_field("dynamic", "reinforce", "kappa_mean")[0],
                                            oc_field("dynamic", "value", "kappa_mean")[0]))
    print("  510K p=0:     (%.3f, %.3f)" % (s510k(0.0, "reinforce")[0], s510k(0.0, "value")[0]))
    print("  510K p=1:     (%.3f, %.3f)" % (s510k(1.0, "reinforce")[0], s510k(1.0, "value")[0]))


if __name__ == "__main__":
    main()
