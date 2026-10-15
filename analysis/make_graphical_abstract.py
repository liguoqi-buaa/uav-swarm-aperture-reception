"""生成 CEP 要求的 Graphical abstract（矢量 PDF + 位图 PNG）。

画布 13.0 × 5.31 cm（≈ Elsevier 1328×531 px 最小比例），不做 tight 裁剪，
输出宽高比稳定 ~2.45:1。

A 三个站点的孔径增益 G（柱）
B 实测 A_m → 等长度 A_L0 的斜率图：排序发生翻转
底部一句话结论
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

SITES = ["0°N\n174.6°W", "30°N\n100.0°W", "45°N\n8.0°E"]
G_P3 = [0.261, 0.275, 0.273]
G_ST = [0.281, 0.272, 0.296]
G_B2 = [0.569, 0.554, 0.613]
AM_ST, AL0_ST = [0.739, 0.729, 0.767], [0.524, 0.516, 0.578]
AM_B2, AL0_B2 = [0.638, 0.606, 0.642], [0.615, 0.593, 0.645]

C_ST, C_B2, C_P3 = "#8c8c8c", "#1f3b57", "#d9d9d9"
EDGE = "#4d4d4d"
MK = ["o", "s", "^"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.0,
    "axes.linewidth": 0.5,
    "axes.edgecolor": "#5a5a5a",
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 1.5,
    "ytick.major.size": 1.5,
    "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5,
})

fig = plt.figure(figsize=(5.118, 2.091), dpi=300)
fig.patch.set_facecolor("white")
gs = fig.add_gridspec(1, 2, left=0.082, right=0.985, top=0.865, bottom=0.375,
                      wspace=0.26)

# ---------------- A: aperture gain ----------------
axA = fig.add_subplot(gs[0, 0])
x, w = np.arange(3.0), 0.25
axA.bar(x - w, G_P3, w, color=C_P3, edgecolor=EDGE, linewidth=0.4, hatch="..",
        label="$P_3$  (2.5 ms)")
axA.bar(x, G_ST, w, color=C_ST, edgecolor=EDGE, linewidth=0.4, hatch="//",
        label="predictive controller  (30.6 ms)")
axA.bar(x + w, G_B2, w, color=C_B2, edgecolor=C_B2, linewidth=0.4,
        label="geometric $B_2$  (0.26 ms)")
axA.set_xticks(x)
axA.set_xticklabels(SITES)
axA.set_ylabel("Aperture gain $G$", fontsize=6.2)
axA.set_ylim(0, 0.99)
axA.set_yticks([0, 0.2, 0.4, 0.6])
axA.set_title("A.  Prediction loses on its own metric", fontsize=6.4,
              loc="left", pad=4, color="#111111")
axA.legend(frameon=False, fontsize=4.9, loc="upper left", handlelength=1.3,
           borderpad=0.1, labelspacing=0.3)
for xi, (a, b) in enumerate(zip(G_ST, G_B2)):
    axA.text(xi + w, b + 0.015, f"+{100*(b-a)/a:.0f}%", ha="center",
             fontsize=5.6, color=C_B2, fontweight="bold")
axA.spines[["top", "right"]].set_visible(False)
axA.grid(axis="y", color="#ececec", linewidth=0.45)
axA.set_axisbelow(True)

# ---------------- B: slope chart, the ranking reverses ----------------
axB = fig.add_subplot(gs[0, 1])
xs = [0.0, 1.0]
for i in range(3):
    axB.plot(xs, [AM_ST[i], AL0_ST[i]], color=C_ST, lw=1.0, marker=MK[i],
             markersize=3.0, markerfacecolor="white", markeredgewidth=0.8,
             label="predictive controller" if i == 0 else None)
    axB.plot(xs, [AM_B2[i], AL0_B2[i]], color=C_B2, lw=1.0, marker=MK[i],
             markersize=3.0, markerfacecolor=C_B2, markeredgewidth=0.8,
             label="geometric $B_2$" if i == 0 else None)
axB.set_xlim(-0.28, 1.28)
axB.set_ylim(0.44, 0.96)
axB.set_xticks(xs)
axB.set_xticklabels(["measured\n$A_m$", "length-controlled\n$A_{L_0}$"])
axB.set_ylabel("Alignment quality", fontsize=6.2)
axB.set_title("B.  Equal length reverses the ranking", fontsize=6.4,
              loc="left", pad=4, color="#111111")
axB.legend(frameon=False, fontsize=4.9, loc="lower left", handlelength=1.3,
           borderpad=0.1, labelspacing=0.3)
axB.text(0.50, 0.885, "the order swaps", ha="center", va="center", fontsize=5.0,
         color="#7a7a7a", style="italic")
axB.spines[["top", "right"]].set_visible(False)
axB.grid(axis="y", color="#ececec", linewidth=0.45)
axB.set_axisbelow(True)

# ---------------- bottom takeaway ----------------
fig.lines.append(plt.Line2D([0.082, 0.985], [0.305, 0.305], color="#d0d0d0",
                            linewidth=0.6, transform=fig.transFigure))
fig.text(0.082, 0.155,
         "Prediction is not the binding constraint: a prediction-free, learning-free geometric controller reaches a\n"
         "higher aperture gain at $1/119$ of the computation, and adding foresight lowers it. The same conclusion\n"
         "holds at three ground sites built from the same TLE archive.",
         fontsize=5.6, color="#111111", va="center", linespacing=1.5)

fig.savefig(OUT / "graphical_abstract.pdf", format="pdf", facecolor="white")
fig.savefig(OUT / "graphical_abstract.png", format="png", dpi=300, facecolor="white")

from PIL import Image  # noqa: E402
im = Image.open(OUT / "graphical_abstract.png")
print(f"PNG {im.size[0]}x{im.size[1]} px  aspect={im.size[0]/im.size[1]:.2f} "
      f"(Elsevier min 1328x531, aspect ~2.5)")
print(f"-> {OUT / 'graphical_abstract.pdf'}")
