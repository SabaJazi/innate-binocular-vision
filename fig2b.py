"""
Publication-quality figure: Single Layer Simulated LGN Activity with Varying P/T.

Grid of panels sweeping T (columns) and P (rows) with fixed r and a.
"""

import matplotlib.pyplot as plt
import numpy as np
from lgn_ibv import LGN, calculate_optimal_p

# ---------------------------------------------------------------------------
# Sweep parameters  
# ---------------------------------------------------------------------------

R           = 4
A           = 0.5
LGN_WIDTH   = 128
RANDOM_SEED = 42

T_VALUES    = [3, 4, 5, 6, 7, 8]
P_VALUES    = [0.30, 0.26, 0.22, 0.18, 0.14, 0.10]   # top to bottom

# ---------------------------------------------------------------------------
# Typography  (points)
# ---------------------------------------------------------------------------

FONT_FAMILY = "DejaVu Sans"
TITLE_SIZE  = 15
LABEL_SIZE  = 13
TICK_SIZE   = 15

# ---------------------------------------------------------------------------
# Layout (inches)  — square panels since the array is square
# ---------------------------------------------------------------------------

PANEL_SIZE    = 1.2
LEFT_MARGIN   = 0.80
RIGHT_MARGIN  = 0.15
TOP_MARGIN    = 0.65
BOTTOM_MARGIN = 0.45
COL_GAP       = 0.06
ROW_GAP       = 0.06

N_ROWS = len(P_VALUES)
N_COLS = len(T_VALUES)

FIG_W = LEFT_MARGIN + N_COLS * PANEL_SIZE + (N_COLS - 1) * COL_GAP + RIGHT_MARGIN
FIG_H = TOP_MARGIN  + N_ROWS * PANEL_SIZE + (N_ROWS - 1) * ROW_GAP  + BOTTOM_MARGIN

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":    FONT_FAMILY,
    "font.size":      LABEL_SIZE,
    "figure.dpi":     300,
    "savefig.dpi":    300,
    "savefig.bbox":   "tight",
    "pdf.fonttype":   42,
    "ps.fonttype":    42,
})

# ---------------------------------------------------------------------------
# Build figure and axes
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(FIG_W, FIG_H))

def to_fx(inch): return inch / FIG_W
def to_fy(inch): return inch / FIG_H

panel_w = to_fx(PANEL_SIZE)
panel_h = to_fy(PANEL_SIZE)
col_gap = to_fx(COL_GAP)
row_gap = to_fy(ROW_GAP)
grid_left   = to_fx(LEFT_MARGIN)
grid_bottom = to_fy(BOTTOM_MARGIN)

axes = {}
for row in range(N_ROWS):
    for col in range(N_COLS):
        x = grid_left + col * (panel_w + col_gap)
        y = 1.0 - to_fy(TOP_MARGIN) - (row + 1) * panel_h - row * row_gap
        axes[(row, col)] = fig.add_axes([x, y, panel_w, panel_h])

# ---------------------------------------------------------------------------
# Generate LGN activity and fill panels
# ---------------------------------------------------------------------------

for row, p in enumerate(P_VALUES):
    for col, t in enumerate(T_VALUES):
        ax = axes[(row, col)]

        np.random.seed(RANDOM_SEED + row * N_COLS + col)
        lgn = LGN(
            width=LGN_WIDTH, p=p, r=R, t=t,
            trans=A, num_layers=2, make_wave=True,
        )
        mat = lgn.make_img_mat()[0]

        ax.imshow(mat, cmap="gray", vmin=0, vmax=1,
                  aspect="equal", interpolation="nearest")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(axis="both", which="both", labelsize=TICK_SIZE, length=3, width=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#666666")

        # T value labels along the bottom row
        if row == N_ROWS - 1:
            ax.set_xlabel(str(t), fontsize=LABEL_SIZE + 2, labelpad=3)

        # Per-row p values are kept as labels, distinct from the global P axis title.
        if col == 0:
            ax.text(-0.1, 0.5, f"{p:.2f}", transform=ax.transAxes,
                    fontsize=LABEL_SIZE + 2, ha="right", va="center",
                    rotation=0, color="black")

# ---------------------------------------------------------------------------
# Shared axis titles and figure title
# ---------------------------------------------------------------------------

# 'T' label centred below the bottom row
fig.text(
    grid_left + (N_COLS * panel_w + (N_COLS - 1) * col_gap) / 2,
    to_fy(BOTTOM_MARGIN) - to_fy(0.28),
    "T",
    ha="center", va="top",
    fontsize=TITLE_SIZE + 2, fontweight="bold",
)

# Global vertical axis label kept far away from the row-wise p values.
fig.text(
    to_fx(0.02),
    grid_bottom + (N_ROWS * panel_h + (N_ROWS - 1) * row_gap) / 2,
    "P",
    ha="center", va="center",
    fontsize=TITLE_SIZE + 2, fontweight="bold",
    rotation=90,
)

# Subtitle with fixed parameters
fig.text(
    grid_left + (N_COLS * panel_w + (N_COLS - 1) * col_gap) / 2,
    1.0 - to_fy(TOP_MARGIN) + to_fy(0.08),
    f"r = {R},  $\\alpha$ = {A}",
    ha="center", va="bottom",
    fontsize=TITLE_SIZE, fontstyle="italic",
)

# Main title
# fig.text(
#     grid_left + (N_COLS * panel_w + (N_COLS - 1) * col_gap) / 2,
#     1.0 - to_fy(0.08),
#     "Single Layer Simulated LGN Activity with Varying P/T",
#     ha="center", va="top",
#     fontsize=TITLE_SIZE + 2, fontweight="bold",
# )

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

# plt.savefig("lgn_pt_sweep.pdf", bbox_inches="tight")
plt.savefig("fig2b_lgn_pt_sweep.png", bbox_inches="tight", dpi=300)
plt.show()

print("Saved: fig2b_lgn_pt_sweep.pdf / fig2b_lgn_pt_sweep.png")