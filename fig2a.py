"""
Publication-quality figure: Alpha and Simulated LGN Layer Similarity.

Produces a high-resolution, journal-ready figure with:
  - Consistent font sizing (Nature / Journal of Neuroscience style)
  - Tight layout with no wasted whitespace
  - Panel labels (a–o) in the top-left corner of every panel
  - Shared column and row annotations
  - 300 dpi output suitable for print
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from lgn_ibv import LGN, calculate_optimal_p

# ---------------------------------------------------------------------------
# Plot settings  (edit here to match your target journal's style guide)
# ---------------------------------------------------------------------------

ALPHA_VALUES   = [0.00, 0.25, 0.5, 0.75, 1.0]
R, T           = 4, 3
P_SHIFT        = 0.01
LGN_WIDTH      = 256
RANDOM_SEED    = 42          # fixed seed → reproducible waves

# Typography  (points)
FONT_FAMILY    = "DejaVu Sans"   # swap for "Arial" if journal requires it
TITLE_SIZE     = 12
LABEL_SIZE     = 11
TICK_SIZE      = 10

# Layout (inches)
COL_WIDTH      = 1.5             # width of each image panel
ROW_HEIGHT     = 1.5             # height of each image panel (match COL_WIDTH for square images)
LEFT_MARGIN    = 0.82            # more room so labels sit outside each corresponding plot
RIGHT_MARGIN   = 0.12
TOP_MARGIN     = 0.40            # space for column titles
BOTTOM_MARGIN  = 0.20
COL_GAP        = 0.10            # gap between columns
ROW_GAP        = 0.08            # gap between rows

N_ROWS = len(ALPHA_VALUES)
N_COLS = 4
ZOOM_COL_GAP = 0.18
ZOOM_PANEL_W = COL_WIDTH
ZOOM_PANEL_H = ROW_HEIGHT
ZOOM_REGION = (40, 60, 110, 130)
ZOOM_FACTOR = round((LGN_WIDTH - 1) / (ZOOM_REGION[2] - ZOOM_REGION[0]))

FIG_W = LEFT_MARGIN + 2 * COL_WIDTH + 2 * ZOOM_PANEL_W + COL_GAP + ZOOM_COL_GAP + RIGHT_MARGIN
FIG_H = TOP_MARGIN  + N_ROWS * ROW_HEIGHT + (N_ROWS - 1) * ROW_GAP  + BOTTOM_MARGIN

# ---------------------------------------------------------------------------
# Global matplotlib style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":        FONT_FAMILY,
    "font.size":          LABEL_SIZE,
    "axes.titlesize":     TITLE_SIZE,
    "axes.labelsize":     LABEL_SIZE,
    "xtick.labelsize":    TICK_SIZE,
    "ytick.labelsize":    TICK_SIZE,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "pdf.fonttype":       42,    # embed fonts as Type 1 → editable in Illustrator
    "ps.fonttype":        42,
})

# ---------------------------------------------------------------------------
# Build axes positions manually so spacing is exact
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(FIG_W, FIG_H))

# Convert margin / gap values to figure fractions
def to_fig_x(inch): return inch / FIG_W
def to_fig_y(inch): return inch / FIG_H

panel_w_frac = to_fig_x(COL_WIDTH)
panel_h_frac = to_fig_y(ROW_HEIGHT)
col_gap_frac = to_fig_x(COL_GAP)
row_gap_frac = to_fig_y(ROW_GAP)

# Bottom-left corner of the grid (figure coordinates)
grid_left   = to_fig_x(LEFT_MARGIN)
grid_bottom = to_fig_y(BOTTOM_MARGIN)

axes = {}
for row in range(N_ROWS):
    for col in range(N_COLS):
        if col < 2:
            x = grid_left + col * (panel_w_frac + col_gap_frac)
            w = panel_w_frac
        else:
            x = grid_left + 2 * (panel_w_frac + col_gap_frac) + ZOOM_COL_GAP / FIG_W + (col - 2) * (to_fig_x(ZOOM_PANEL_W) + col_gap_frac * 0.5)
            w = to_fig_x(ZOOM_PANEL_W)
        y = 1.0 - to_fig_y(TOP_MARGIN) - (row + 1) * panel_h_frac - row * row_gap_frac
        ax = fig.add_axes([x, y, w, panel_h_frac])
        ax.set_aspect('equal')
        axes[(row, col)] = ax

# ---------------------------------------------------------------------------
# Generate and draw data
# ---------------------------------------------------------------------------

COLUMN_TITLES = ["Layer 1", "Layer 2", f"{ZOOM_FACTOR}x zoom Layer 1", f"{ZOOM_FACTOR}x zoom Layer 2"]
for row, a in enumerate(ALPHA_VALUES):
    p = calculate_optimal_p(T, R, a) + P_SHIFT

    np.random.seed(RANDOM_SEED + row)  # different wave per row, but reproducible
    lgn = LGN(
        width=LGN_WIDTH, p=p, r=R, t=T,
        trans=a, num_layers=2, make_wave=True,
    )
    mat = lgn.make_img_mat()

    layer1 = mat[0]
    layer2 = mat[1]
    x0, y0, x1, y1 = ZOOM_REGION
    zoom1 = layer1[y0:y1, x0:x1]
    zoom2 = layer2[y0:y1, x0:x1]

    images = [layer1, layer2, zoom1, zoom2]

    for col, img in enumerate(images):
        ax = axes[(row, col)]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_aspect('equal')

        if col < 2:
            rect = mpatches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                fill=False, edgecolor='red', linewidth=1.0,
                linestyle='-', transform=ax.transData
            )
            ax.add_patch(rect)

        # Remove all tick marks and spines
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#444444")

        # Column titles on the top row only
        if row == 0:
            ax.set_title(COLUMN_TITLES[col], fontsize=TITLE_SIZE,
                         fontweight="normal", pad=4)

        # Row alpha value on the left of the first column
        if col == 0:
            ax.set_ylabel(
                rf"$\alpha = {a:.2f}$",
                fontsize=LABEL_SIZE,
                rotation=0,
                labelpad=28,
                va="center",
            )

# ---------------------------------------------------------------------------
# Axis annotations
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

# OUTPUT_PATH = "lgn_layer_similarity_pub.pdf"   # PDF for vector submission
PNG_PATH    = "figur2a.png"   # PNG for preview / supplementary

# plt.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.savefig(PNG_PATH,    bbox_inches="tight", dpi=300)
plt.show()

# print(f"Saved: {OUTPUT_PATH}")
print(f"Saved: {PNG_PATH}")