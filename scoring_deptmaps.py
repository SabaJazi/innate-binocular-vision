import os
import cv2
import json
import matplotlib.pyplot as plt
import math
from decimal import Decimal
from collections import defaultdict

path_to_dm = 'C:\\Users\\19404\\innate-binocular-vision\\2026-05-15\\images\\depthmaps'
path_to_json = 'C:\\Users\\19404\\innate-binocular-vision\\2026-05-15\\json\\'

json_files = [p for p in os.listdir(path_to_json) if p.endswith('.json')]

items = []
for jf in json_files:
    with open(os.path.join(path_to_json, jf)) as f:
        jd = json.load(f)
    try:
        lgn_a = Decimal(str(jd.get('lgn_a', 'nan')))
    except Exception:
        lgn_a = Decimal('NaN')
    lgn_t = int(jd.get('lgn_t', -999))
    corr = float(jd.get('corr', 0))
    items.append({
        'json': jf,
        'dm': jf[:-5] + '.png',
        'lgn_a': lgn_a,
        'lgn_t': lgn_t,
        'corr': corr
    })

if not items:
    print("No JSON files found in", path_to_json)
    raise SystemExit

# Group by lgn_a and sort each group by lgn_t
groups = defaultdict(list)
for it in items:
    groups[it['lgn_a']].append(it)
for k in groups:
    groups[k].sort(key=lambda x: x['lgn_t'])

# Build rows: each row corresponds to up to `cols` items of the same lgn_a.
cols = 6
row_slices = []
for lgn_a in sorted(groups.keys()):
    lst = groups[lgn_a]
    for i in range(0, len(lst), cols):
        row_slices.append((lgn_a, lst[i:i+cols]))

rows = len(row_slices)
fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.1), dpi=100)
if rows == 1:
    axes = [axes] if cols == 1 else axes
axes_flat = axes.flatten() if hasattr(axes, 'flatten') else list(axes)

for row_idx, (lgn_a, slice_items) in enumerate(row_slices):
    for col_idx in range(cols):
        ax = axes_flat[row_idx * cols + col_idx]
        if col_idx < len(slice_items):
            it = slice_items[col_idx]
            img_path = os.path.join(path_to_dm, it['dm'])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            title = f"lgn_t={it['lgn_t']}\ncorr={it['corr']:.2f}"

            if img is None:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
            else:
                ax.imshow(img, cmap='gray')
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                ax.set_ylabel(f"lgn_a={lgn_a}", fontsize=9)      # show lgn_a at left of each row
            if row_idx == rows - 1:
                ax.set_xlabel(f"lgn_t={it['lgn_t']}", fontsize=9)  # show lgn_t under bottom-row columns
       
        else:
            ax.axis('off')
        ax.axis('off')
fig.supylabel('lgn_a', fontsize=11)
fig.supxlabel('lgn_t', fontsize=11)
# plt.tight_layout(pad=0.1)
plt.subplots_adjust(wspace=0.12, hspace=0.68, left=0.02, right=0.98, top=0.96, bottom=0.02)

plt.show()