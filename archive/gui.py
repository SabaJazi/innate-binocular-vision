import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import subprocess
import LGN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
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
    
    canvas = FigureCanvasTkAgg(plt.gcf(), master=frame)
    canvas.get_tk_widget().grid(row=0, column=0)  # Adjust the row and column values

   # Create the main window
root = tk.Tk()
root.title("Experiment Parameter Input")
root.geometry("1000x700")
input_frame = ttk.Frame(root,width=350, height=660, relief="ridge")

# input_frame.grid(row=0, column=0, padx=10, pady=10)
input_frame.pack(side="left", padx=5, pady=10)

frame = ttk.Frame(root, width=600, height=660, relief="ridge")
# frame.grid(row=1, column=0,padx=10, pady=10)
frame.pack(side="right", padx=10, pady=10)
# Add a vertical separator (border) between the frames

# Create labels and entry widgets for LGN parameters
label_title1 = tk.Label(input_frame, text="Enter values or run the default")

label_lgn_a = tk.Label(input_frame, text="LGN Parameter 'a' (min max step):")
entry_lgn_a = ttk.Entry(input_frame)
entry_lgn_a.insert(0,"0.0")
label_lgn_r = tk.Label(input_frame, text="LGN Parameter 'r' (min max step):")
entry_lgn_r = ttk.Entry(input_frame)
entry_lgn_r.insert(0,"4.0")
label_lgn_p = tk.Label(input_frame, text="LGN Parameter 'p' (min max step):")
entry_lgn_p = ttk.Entry(input_frame)
entry_lgn_p.insert(0,"0.15")
label_lgn_t = tk.Label(input_frame, text="LGN Parameter 't' (min max step):")
entry_lgn_t = ttk.Entry(input_frame)
entry_lgn_t.insert(0,"5.0")

# Create Run button
button_run = ttk.Button(input_frame, text="Run Script", command=run_script)

# Organize widgets using grid layout
label_title1.grid(row=0, column=0, padx=10, pady=25)
label_lgn_a.grid(row=1, column=0, padx=10, pady=25)
entry_lgn_a.grid(row=1, column=1, padx=10, pady=25)
label_lgn_r.grid(row=2, column=0, padx=10, pady=25)
entry_lgn_r.grid(row=2, column=1, padx=10, pady=25)
label_lgn_p.grid(row=3, column=0, padx=10, pady=25)
entry_lgn_p.grid(row=3, column=1, padx=10, pady=25)
label_lgn_t.grid(row=4, column=0, padx=10, pady=25)
entry_lgn_t.grid(row=4, column=1, padx=10, pady=25)
button_run.grid(row=5, column=0, columnspan=2, padx=10, pady=40)



# Start the GUI event loop
root.mainloop()