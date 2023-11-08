import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import subprocess
import LGN
import numpy as np
import matplotlib.pyplot as plt
def run_script():
    # Collect parameter values from the text boxes
    lgn_a = entry_lgn_a.get().split()
    lgn_r = entry_lgn_r.get().split()
    lgn_p = entry_lgn_p.get().split()
    lgn_t = entry_lgn_t.get().split()
    # print("LGN Parameter 'a':", lgn_a)
    # print("LGN Parameter 'r':", lgn_r)
    print("LGN Parameter 'p':", float(lgn_p[0]))
    # print("LGN Parameter 't':", lgn_t)
    L = LGN.LGN(width=64, p=float(lgn_p[0]), r=float(lgn_r[0]),
     t=float(lgn_t[0]), trans=float(lgn_a[0]),
            make_wave=True, num_layers=2)
    generated_activity = L.make_img_mat()
    differences = np.abs(generated_activity[0]-generated_activity[1])
    fig, (layer_1, layer_2, diff) = plt.subplots(1, 3, sharex=True)
    layer_1.axis("off")
    layer_2.axis("off")
    diff.axis("off")

    layer_1.imshow(generated_activity[0], cmap="Greys_r")
    layer_2.imshow(generated_activity[1], cmap="Greys_r")
    diff.imshow(differences, cmap="Reds_r")
    filename = "p_0.15-r_4.0-t_5.0_a_{}.png".format(1)
    plt.savefig(filename,dpi=300)
    plt.show()
# Create the main window
root = tk.Tk()
root.title("Experiment Parameter Input")

# Create labels and entry widgets for LGN parameters
label_lgn_a = tk.Label(root, text="LGN Parameter 'a' (min max step):")
entry_lgn_a = ttk.Entry(root)
entry_lgn_a.insert(0,"0.0")
label_lgn_r = tk.Label(root, text="LGN Parameter 'r' (min max step):")
entry_lgn_r = ttk.Entry(root)
entry_lgn_r.insert(0,"4.0")
label_lgn_p = tk.Label(root, text="LGN Parameter 'p' (min max step):")
entry_lgn_p = ttk.Entry(root)
entry_lgn_p.insert(0,"0.15")
label_lgn_t = tk.Label(root, text="LGN Parameter 't' (min max step):")
entry_lgn_t = ttk.Entry(root)
entry_lgn_t.insert(0,"5.0")

# Create Run button
button_run = ttk.Button(root, text="Run Script", command=run_script)

# Organize widgets using grid layout
label_lgn_a.grid(row=0, column=0, padx=10, pady=5)
entry_lgn_a.grid(row=0, column=1, padx=10, pady=5)
label_lgn_r.grid(row=1, column=0, padx=10, pady=5)
entry_lgn_r.grid(row=1, column=1, padx=10, pady=5)
label_lgn_p.grid(row=2, column=0, padx=10, pady=5)
entry_lgn_p.grid(row=2, column=1, padx=10, pady=5)
label_lgn_t.grid(row=3, column=0, padx=10, pady=5)
entry_lgn_t.grid(row=3, column=1, padx=10, pady=5)
button_run.grid(row=4, columnspan=2, padx=10, pady=15)

# Start the GUI event loop
root.mainloop()