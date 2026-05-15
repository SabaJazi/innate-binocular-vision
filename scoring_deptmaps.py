import os
import cv2
import json
import matplotlib.pyplot as plt
import math
from decimal import Decimal

path_to_dm = 'C:\\Users\\19404\\innate-binocular-vision\\2026-05-15\\images\\depthmaps'

path_to_json = 'C:\\Users\\19404\\innate-binocular-vision\\2026-05-15\\json\\'

json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]
# jsons_data = pd.DataFrame(columns=['id','corr','lgn_p','lgn_a', 'lgn_r', 'lgn_t'])
# for index, js in enumerate(json_files):
#   with open(os.path.join(path_to_json, js)) as json_file:
#         json_text = json.load(json_file)
#         id = json_text['id']
#         corr = json_text['corr']
#         lgn_p = json_text['lgn_p']
#         lgn_a = json_text['lgn_a']
#         lgn_r = json_text['lgn_r']
#         lgn_t = json_text['lgn_t']
#         jsons_data.loc[index] = [id, corr, lgn_p, lgn_a, lgn_r, lgn_t]
lgn_a_values = []
dm_files = []
sorted_json_files = []

for json_file in json_files:
    with open(os.path.join(path_to_json, json_file)) as f:
        json_data = json.load(f)
        # if json_data['lgn_t'] == 2 and json_data['lgn_r'] ==2:
        lgn_a_values.append(Decimal(str(json_data['lgn_a'])))
        dm_files.append(json_file[:-5] + '.png')
        sorted_json_files.append(json_file)

sorted_indices = sorted(range(len(lgn_a_values)), key=lambda k: lgn_a_values[k])
sorted_dm_files = [dm_files[i] for i in sorted_indices]
sorted_json_files = [sorted_json_files[i] for i in sorted_indices]

fig, axes = plt.subplots(1, len(sorted_dm_files), figsize=(15, 5))

for ax, dm_file, json_file in zip(axes, sorted_dm_files, sorted_json_files):
    img = cv2.imread(os.path.join(path_to_dm, dm_file), cv2.IMREAD_GRAYSCALE)
    with open(os.path.join(path_to_json, json_file)) as f:
        json_data = json.load(f)
    corr_value = round(json_data['corr'], 2)
    lgn_a_value = round(json_data['lgn_a'], 3)
    ax.imshow(img, cmap='gray')
    # Set title for each image using the "corr" value and "lgn_a" value
    # ax.set_title(f'corr={corr_value}\nlgn_a={lgn_a_value}')
    ax.set_title(f'lgn_a={lgn_a_value}\ncorr={corr_value}')

    ax.axis('off')

plt.tight_layout(pad=1.0)

plt.show()