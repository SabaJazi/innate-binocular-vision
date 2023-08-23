import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import subprocess

def run_script():
    # Collect parameter values from the text boxes
    lgn_a_values = entry_lgn_a.get().split()
    lgn_r_values = entry_lgn_r.get().split()
    lgn_p_values = entry_lgn_p.get().split()
    lgn_t_values = entry_lgn_t.get().split()


    # print("LGN Parameter 'a':", lgn_a)
    # print("LGN Parameter 'r':", lgn_r)
    # print("LGN Parameter 'p':", lgn_p)
    # print("LGN Parameter 't':", lgn_t)

    # Construct the command to run the script with parameters
    command = [
        "python", "create_exp_local.py",
        "-la", *lgn_a_values,
        "-lr", *lgn_r_values,
        "-lp", *lgn_p_values,
        "-lt", *lgn_t_values
    ]
    print("Command:", command)
    # Run the script with subprocess
    subprocess.call(command)

# Create the main window
root = tk.Tk()
root.title("Experiment Parameter Input")

# Create labels and entry widgets for LGN parameters
label_lgn_a = tk.Label(root, text="LGN Parameter 'a' (min max step):")
entry_lgn_a = ttk.Entry(root)
label_lgn_r = tk.Label(root, text="LGN Parameter 'r' (min max step):")
entry_lgn_r = ttk.Entry(root)
label_lgn_p = tk.Label(root, text="LGN Parameter 'p' (min max step):")
entry_lgn_p = ttk.Entry(root)
label_lgn_t = tk.Label(root, text="LGN Parameter 't' (min max step):")
entry_lgn_t = ttk.Entry(root)

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