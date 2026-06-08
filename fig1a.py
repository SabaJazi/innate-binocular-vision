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

ALPHA_VALUES   = [0.00, 0.16, 0.25, 0.38, 0.50]
R, T           = 4, 3
P_SHIFT        = 0.01
LGN_WIDTH      = 128
RANDOM_SEED    = 42          # fixed seed → reproducible waves

# Typography  (points)
FONT_FAMILY    = "DejaVu Sans"   # swap for "Arial" if your journal requires it
TITLE_SIZE     = 9
LABEL_SIZE     = 8
TICK_SIZE      = 7

# Layout (inches)
COL_WIDTH      = 1.5             # width of each image panel
ROW_HEIGHT     = 1.5             # height of each image panel (match COL_WIDTH for square images)
LEFT_MARGIN    = 0.55            # space for the 'a' axis label + row values
RIGHT_MARGIN   = 0.12
TOP_MARGIN     = 0.40            # space for column titles
BOTTOM_MARGIN  = 0.20
COL_GAP        = 0.10            # gap between columns
ROW_GAP        = 0.08            # gap between rows

N_ROWS = len(ALPHA_VALUES)
N_COLS = 3

FIG_W = LEFT_MARGIN + N_COLS * COL_WIDTH + (N_COLS - 1) * COL_GAP + RIGHT_MARGIN
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
        x = grid_left  + col * (panel_w_frac + col_gap_frac)
        # rows go top-to-bottom so row 0 is at the top
        y = 1.0 - to_fig_y(TOP_MARGIN) - (row + 1) * panel_h_frac - row * row_gap_frac
        ax = fig.add_axes([x, y, panel_w_frac, panel_h_frac])
        axes[(row, col)] = ax

# ---------------------------------------------------------------------------
# Generate and draw data
# ---------------------------------------------------------------------------

COLUMN_TITLES = ["Layer 1", "Layer 2", "Difference"]
for row, a in enumerate(ALPHA_VALUES):
    p = calculate_optimal_p(T, R, a) + P_SHIFT

    np.random.seed(RANDOM_SEED + row)  # different wave per row, but reproducible
    lgn = LGN(
        width=LGN_WIDTH, p=p, r=R, t=T,
        trans=a, num_layers=2, make_wave=True,
    )
    mat = lgn.make_img_mat()

    layer1     = mat[0]
    layer2     = mat[1]
    difference = np.abs(layer1 - layer2)

    images = [layer1, layer2, difference]

    for col, img in enumerate(images):
        ax = axes[(row, col)]

        ax.imshow(img, cmap="gray", vmin=0, vmax=1,
                  aspect="auto", interpolation="nearest")

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
                f"{a:.2f}",
                fontsize=LABEL_SIZE,
                rotation=0,
                labelpad=18,
                va="center",
            )

# ---------------------------------------------------------------------------
# Axis annotations
# ---------------------------------------------------------------------------

# Shared 'a' label centred vertically on the left margin
fig.text(
    to_fig_x(0.08), grid_bottom + N_ROWS * panel_h_frac / 2 + (N_ROWS - 1) * row_gap_frac / 2,
    r"$\alpha$",
    va="center", ha="center",
    fontsize=TITLE_SIZE + 1, fontstyle="italic",
    rotation=90,
)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

OUTPUT_PATH = "lgn_layer_similarity_pub.pdf"   # PDF for vector submission
PNG_PATH    = "lgn_layer_similarity_pub.png"   # PNG for preview / supplementary

plt.savefig(OUTPUT_PATH, bbox_inches="tight")
plt.savefig(PNG_PATH,    bbox_inches="tight", dpi=300)
plt.show()

print(f"Saved: {OUTPUT_PATH}")
print(f"Saved: {PNG_PATH}")