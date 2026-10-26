"""Chart style for the research notebooks, and figure export for the README.

One categorical palette in a fixed, colour-blind-checked order; P&L components keep the same
colour in every chart (net blue, price orange, funding aqua, fee yellow). Thin lines, hairline
grid, no dual axes; lines are labelled at their end as well as in the legend, because two of
the colours are low-contrast on the light background.
"""

import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from cycler import cycler

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PNL = {"net": SERIES[0], "price": SERIES[1], "funding": SERIES[2], "fee": SERIES[3], "cost": SERIES[3]}
LEG = {"long": SERIES[6], "short": SERIES[4]}      # violet / magenta, distinct from the P&L colours
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
FIG_DIR = "research/figures"


def apply():
    mpl.rcParams.update({
        "axes.prop_cycle": cycler(color=SERIES),
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.axisbelow": True,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelcolor": INK2, "axes.labelsize": 9,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "text.color": INK, "font.size": 9, "lines.linewidth": 1.6,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "date.converter": "concise",           # short, non-overlapping date ticks
    })


def zero(ax):
    ax.axhline(0, color=MUTED, lw=0.8, zorder=1)


def end_label(ax, series, text, color):
    """Name a line at its last point. The text is placed when the figure is saved, once every
    label on the axes is known, so neighbouring labels can be spread apart."""
    s = series.dropna()
    if len(s):
        ax.plot([s.index[-1]], [s.iloc[-1]], "o", ms=3.5, color=color)
        ax.__dict__.setdefault("_end_labels", []).append((s.index[-1], s.iloc[-1], text))


def _place_end_labels(fig, gap_pt=11):
    """Spread each axes' end labels vertically so neighbours keep at least gap_pt apart."""
    fig.canvas.draw()
    px_per_pt = fig.dpi / 72
    for ax in fig.axes:
        items = ax.__dict__.pop("_end_labels", [])
        if not items:
            continue
        pos = []
        for x, y, text in items:
            xd, yd = ax.transData.transform((ax.convert_xunits(x), y))
            pos.append([yd, yd, x, y, text])
        pos.sort(key=lambda p: p[0])
        gap = gap_pt * px_per_pt
        for i in range(1, len(pos)):                       # push up from the bottom ...
            pos[i][1] = max(pos[i][1], pos[i - 1][1] + gap)
        shift = (sum(p[1] for p in pos) - sum(p[0] for p in pos)) / len(pos)
        for p in pos:                                      # ... then re-centre on the data
            p[1] -= shift
        for i in range(1, len(pos)):
            pos[i][1] = max(pos[i][1], pos[i - 1][1] + gap)
        for y0, y1, x, y, text in pos:
            ax.annotate(text, (x, y), xytext=(5, (y1 - y0) / px_per_pt), textcoords="offset points",
                        va="center", fontsize=8, color=INK2)


def pnl_lines(ax, daily, cols=("funding", "price", "fee", "net"), title=None, sign_fee=True):
    """Cumulative P&L components, % of starting equity, in their fixed colours."""
    for c in cols:
        s = daily[c].cumsum() * 100 * (-1 if (c in ("fee", "cost") and sign_fee) else 1)
        ax.plot(s.index, s, color=PNL[c], lw=2.2 if c == "net" else 1.4,
                label=c if c not in ("fee", "cost") else "fees")
        end_label(ax, s, c if c not in ("fee", "cost") else "fees", PNL[c])
    zero(ax)
    ax.set_ylabel("cumulative, % of equity")
    if title:
        ax.set_title(title)
    ax.legend(loc="upper left")


def legend_below(ax, ncol=3):
    """Legend under the plot, for charts whose data fill every corner."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=ncol)


def bars(ax, values, labels=None, color=SERIES[0], horizontal=False, fmt="{:.0f}"):
    """Single-series bars with a surface gap between them and selective value labels."""
    labels = list(values.index) if labels is None else labels
    kw = dict(color=color, edgecolor=SURFACE, linewidth=1.5)
    rects = ax.barh(labels, values, **kw) if horizontal else ax.bar(labels, values, **kw)
    ax.bar_label(rects, labels=[fmt.format(v) for v in values], fontsize=8, color=INK2, padding=2)
    if horizontal:
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.margins(x=0.18)                     # room for the value labels past the bar ends
    else:
        zero(ax)
        ax.margins(y=0.15)
    return rects


def save(fig, name):
    _place_end_labels(fig)
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"))
