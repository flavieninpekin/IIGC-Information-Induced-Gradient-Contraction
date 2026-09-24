"""Figure: retention alone vs the joint diagnostic (Toy / Overcooked / 510K).

Left panel: scatter of kappa_mix (reinforce) against kappa_mix (value/soft).
Retention alone separates fields but says nothing about how much of the field
is condition-dependent.

Right panel: kappa_mix against the condition contrast over the
mean-estimation noise floor (log scale). A point with high retention and a
negligible/near-floor contrast is condition-blind, not aligned; points left of
the dashed line have an unresolvable condition contrast. Marker shape encodes
the field (circle = reinforce, triangle = awr, square = value); color encodes
the environment.
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


def _load(name):
    return json.load(open(os.path.join(RES, name)))


def oc_field(mode, field, key="kappa_mix"):
    o = _load("oc_field_axis.json")
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
    d = _load("510k_field_axis.json")
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


def oc_joint(mode, field):
    o = _load("oc_field_axis.json")
    km, cn = [], []
    for s in (41, 44, 48):
        e = o[f"{mode}_s{s}_{field}"]
        km.append(e["kappa_mix"])
        cn.append(e["E_contrast"] / e["mean_noise_energy"])
    return float(np.mean(km)), float(np.mean(np.log10(cn)))


def s510k_joint(p, field):
    d = _load("510k_field_axis.json")
    km, cn = [], []
    for s in range(41, 47):
        level = d.get(f"{p:.1f}", {})
        e = level.get(f"s{s}_paired", level.get(f"s{s}", {}))[field]
        km.append(e["kappa_mix"])
        cn.append(e["E_contrast"] / e["mean_noise_energy"])
    return float(np.mean(km)), float(np.mean(np.log10(cn)))


def toy():
    return (0.0, 0.903), (0.44, 0.44)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.9), dpi=150)

    colors = {"Toy": "#2ca02c", "Overcooked": "#1f77b4", "510K": "#ff7f0e"}
    hidden_marker, visible_marker = "o", "s"
    field_marker = {"reinforce": "o", "awr": "^", "value": "s"}

    # ---------------- left: retention scatter ----------------
    ax1.plot([0, 1], [0, 1], "--", color="gray", lw=1.2, zorder=1)

    (rf_h, val_h), (rf_r, val_r) = toy()
    ax1.scatter([rf_h], [val_h], marker=hidden_marker, s=90, color=colors["Toy"],
                edgecolor="k", zorder=5)
    ax1.scatter([rf_r], [val_r], marker=visible_marker, s=80, facecolor="none",
                edgecolor=colors["Toy"], linewidths=2, zorder=5)
    ax1.annotate("Toy (mirror anchor)", xy=(rf_r + 0.015, val_r - 0.01),
                 fontsize=9, color=colors["Toy"])

    for mode, lbl, dx, dy in [("static", "Overcooked static", 0.015, -0.045),
                              ("dynamic", "Overcooked dynamic", -0.30, 0.005)]:
        rx, rs = oc_field(mode, "reinforce")
        vx, vs = oc_field(mode, "value")
        m = hidden_marker if mode == "dynamic" else visible_marker
        if mode == "dynamic":
            ax1.scatter([rx], [vx], marker=m, s=90, color=colors["Overcooked"],
                        edgecolor="k", zorder=5)
            ax1.errorbar([rx], [vx], xerr=[rs], yerr=[vs], fmt="none",
                         ecolor=colors["Overcooked"], zorder=4)
        else:
            ax1.scatter([rx], [vx], marker=m, s=80, facecolor="none",
                        edgecolor=colors["Overcooked"], linewidths=2, zorder=5)
        ax1.annotate(lbl, xy=(rx + dx, vx + dy), fontsize=8,
                     color=colors["Overcooked"])

    for p, lbl, dy in [(0.0, "510K p=0 (hidden)", 0.012),
                       (1.0, "510K p=1 (visible)", -0.025)]:
        rx, rs = s510k(p, "reinforce")
        vx, vs = s510k(p, "value")
        m = hidden_marker if p == 0.0 else visible_marker
        if p == 0.0:
            ax1.scatter([rx], [vx], marker=m, s=90, color=colors["510K"],
                        edgecolor="k", zorder=5)
            ax1.errorbar([rx], [vx], xerr=[rs], yerr=[vs], fmt="none",
                         ecolor=colors["510K"], zorder=4)
        else:
            ax1.scatter([rx], [vx], marker=m, s=80, facecolor="none",
                        edgecolor=colors["510K"], linewidths=2, zorder=5)
        ax1.annotate(lbl, xy=(rx + 0.02, vx + dy), fontsize=8,
                     color=colors["510K"], va="bottom" if dy > 0 else "top",
                     ha="left")

    ax1.set_xlabel(r"$\kappa_{\mathrm{mix}}$ (reinforce field)", fontsize=11)
    ax1.set_ylabel(r"$\kappa_{\mathrm{mix}}$ (value / soft field)", fontsize=11)
    ax1.set_title("Retention alone: no condition-contrast information",
                  fontsize=10)
    ax1.set_xlim(0, 1.05)
    ax1.set_ylim(0, 1.08)
    ax1.grid(alpha=0.25, zorder=0)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="gray", linestyle="--", lw=1.2, label="fields equal (y=x)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="k", markersize=8,
               label="hidden"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="none",
               markeredgecolor="k", markersize=8, label="visible"),
    ]
    ax1.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.9)

    # ---------------- right: joint diagnostic ----------------
    ax2.axvspan(-0.6, 0.0, color="gray", alpha=0.12, zorder=0)
    ax2.axvline(0.0, color="gray", ls="--", lw=1.2, zorder=1)
    ax2.text(-0.58, 0.045,
             "contrast at/below\nthe noise floor\n(unresolvable)",
             fontsize=8, color="dimgray", va="bottom")

    pts = []
    for mode, off in [("static", (0.0, 0.0)), ("dynamic", (0.0, 0.0))]:
        for field in ("reinforce", "awr", "value"):
            km, x = oc_joint(mode, field)
            pts.append((km, x, colors["Overcooked"], field,
                        f"OC {mode} {field}"))
    for p in (0.0, 0.5, 1.0):
        for field in ("reinforce", "value"):
            km, x = s510k_joint(p, field)
            pts.append((km, x, colors["510K"], field,
                        f"510K p={p} {field}"))

    for km, x, c, field, _ in pts:
        ax2.scatter([x], [km], marker=field_marker[field], s=95, color=c,
                    edgecolor="k", zorder=5)

    ann = [
        ("OC dyn value", oc_joint("dynamic", "value")),
        ("OC dyn awr", oc_joint("dynamic", "awr")),
        ("OC static value", oc_joint("static", "value")),
    ]
    for label, (km, x) in ann:
        dy = 0.065 if label != "OC static value" else -0.09
        ax2.annotate(label, xy=(x, km), xytext=(x - 0.55, km + dy),
                     fontsize=8, color="k", ha="left",
                     arrowprops=dict(arrowstyle="-", color="gray", lw=0.7))
    ax2.annotate("510K value (p=0, 0.5, 1)",
                 xy=(s510k_joint(0.5, "value")[1], s510k_joint(0.5, "value")[0]),
                 xytext=(0.16, 0.86), fontsize=8, color=colors["510K"],
                 ha="left", arrowprops=dict(arrowstyle="-", color="gray", lw=0.7))

    ax2.set_xlabel(r"$\log_{10}\left(E_{\mathrm{contrast}}\,/\,\mathrm{mean\ noise}\right)$",
                   fontsize=11)
    ax2.set_ylabel(r"$\kappa_{\mathrm{mix}}$ (retention)", fontsize=11)
    ax2.set_title("Joint read: retention vs resolvable contrast", fontsize=10)
    ax2.set_xlim(-0.6, 3.9)
    ax2.set_ylim(0, 1.13)
    ax2.grid(alpha=0.25, zorder=0)

    field_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="gray",
               markeredgecolor="k", markersize=8, label="reinforce"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="gray",
               markeredgecolor="k", markersize=8, label="awr"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="gray",
               markeredgecolor="k", markersize=8, label="value"),
    ]
    env_handles = [Line2D([0], [0], marker="o", color="none",
                          markerfacecolor=c, markeredgecolor="k", markersize=8,
                          label=k) for k, c in colors.items() if k != "Toy"]
    leg1 = ax2.legend(handles=field_handles, loc="upper left", fontsize=8,
                      framealpha=0.9, title="field")
    ax2.add_artist(leg1)
    ax2.legend(handles=env_handles, loc="lower right", fontsize=8,
               framealpha=0.9, title="env")

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, "field_axis_comparison.png"), bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, "field_axis_comparison.pdf"), bbox_inches="tight")
    print("saved", os.path.join(OUTDIR, "field_axis_comparison.png"))
    print("joint points (kappa, log10 contrast/noise):")
    for km, x, _, field, name in pts:
        print(f"  {name:28s} kappa={km:.3f} log10(c/n)={x:+.2f}")


if __name__ == "__main__":
    main()
