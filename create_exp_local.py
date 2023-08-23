import os
import numpy as np
import argparse
import datetime
import json

def generate_parameter_steps(min_value, max_value, step):
    return np.linspace(min_value, max_value, step)

def generate_lgn_parameter_set(a_array, r_array, p_array, t_array):
    lgn_parameter_set = []
    for lgn_a in a_array:
        for lgn_r in r_array:
            for lgn_t in t_array:
                for lgn_p in p_array:
                    #p = calculate_optimal_p(lgn_t, lgn_r, lgn_a) * lgn_p
                    #switch lgn_p to p and uncomment the above line to handle percentages
                    name = "a{:0.2f}_r{:.2f}_p{:.2f}_t{:.2f}".format(lgn_a,lgn_r,lgn_p,lgn_t)
                    parameter = {
                    "lgn_a": lgn_a,
                    "lgn_r": lgn_r,
                    "lgn_p": lgn_p,
                    "lgn_t": lgn_t,
                    "name" : name,
                    }
                    lgn_parameter_set.append(dict(parameter))
    return lgn_parameter_set

def generate_experiment_set(experiment_id, depthmap_name, autostereogram_name, autostereogram_patch, num_filters, num_components, num_patches, patch_size, lgn_size, lgn_parameter_set):
    experiment_set = {
    "experiment_id": experiment_id,
    "parameter_path": "experiments/{}/inputs/parameters".format(experiment_id),
    "depthmap_path": "experiments/{}/inputs/{}".format(experiment_id, depthmap_name),
    "autostereogram_path": "experiments/{}/inputs/{}".format(experiment_id,autostereogram_name),
    "autostereogram_patch": autostereogram_patch,
    "num_filters": num_filters,
    "num_components": num_components,
    "num_patches": num_patches,
    "patch_size": patch_size,
    "lgn_size": lgn_size,
    "lgn_parameter_set": lgn_parameter_set,
    }
    return experiment_set
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Python script to create ibv experiment parameter file")
    parser.add_argument("-la", "--lgn_a", help="LGN model alpha (min, max, step)", nargs=3, metavar=("min", "max", "step"), type=float, required=True)
    parser.add_argument("-lr", "--lgn_r", help="LGN model radius (min, max, step)", nargs=3, metavar=("min", "max", "step"), type=float, required=True)
    parser.add_argument("-lp", "--lgn_p", help="LGN model proportion (min, max, step)", nargs=3, metavar=("min", "max", "step"), type=float, required=True)
    parser.add_argument("-lt", "--lgn_t", help="LGN model threshold (min, max, step)", nargs=3, metavar=("min", "max", "step"), type=float, required=True)
    args = parser.parse_args()

    # Generate LGN parameter arrays
    a_array = generate_parameter_steps(args.lgn_a[0], args.lgn_a[1], int(args.lgn_a[2]))
    r_array = generate_parameter_steps(args.lgn_r[0], args.lgn_r[1], int(args.lgn_r[2]))
    p_array = generate_parameter_steps(args.lgn_p[0], args.lgn_p[1], int(args.lgn_p[2]))
    t_array = generate_parameter_steps(args.lgn_t[0], args.lgn_t[1], int(args.lgn_t[2]))

    # Generate LGN parameter set
    pset = generate_lgn_parameter_set(a_array, r_array, p_array, t_array)

    # Generate experiment set
    # args.autostereogram_patch=5, nf=2000, nc=20,np=100000 ,ps=8,lgn_size=64
    exp = generate_experiment_set(
        1 ,'depthmap_name', 'autosterogram_name', 5,
        2000, 20, 100000, 8,
        64, pset
    )

    # Save experiment parameter file locally
    with open("experiment.json", "w") as json_file:
        json.dump(exp, json_file, indent=4)

    print("Experiment parameter file 'experiment.json' created.")