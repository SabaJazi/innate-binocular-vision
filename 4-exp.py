import os
import ibv
import json
import random
import datetime
import argparse
import numpy as np

def get_parameters(experiment_id):
    # parameters_file = r"C:\vscode\innate-binocular-vision\innate-binocular-vision\experiment{}.json".format(experiment_id)
    parameters_file = r"C:\Users\19404\innate-binocular-vision\experiment{}.json".format(experiment_id)

    with open(parameters_file, "r") as f:
        experiment_parameters = json.load(f)
    return experiment_parameters
 
 #--------------------------
def run_workload(experiment_parameters):

    started = []
    total = []
    # path = r"C:\vscode\innate-binocular-vision\innate-binocular-vision\experiments\{}\outputs\json".format(experiment_parameters["experiment_id"])
    path = r"C:\Users\19404\innate-binocular-vision"

    # List files in the output JSON directory
    started = os.listdir(path)
    
    for lgn_parameters in experiment_parameters["lgn_parameter_set"]:
        total.append(lgn_parameters["name"])
    
    while len(set(total)) != len(set(started)):
        diff = list(set(total) - set(started))
        if len(diff) == 0:
            break
        # choose one of names from json file and
        #  get the parameters of it
        selection = random.choice(diff)
        experiment_subparameters = extract_subparameters(
            experiment_parameters,
            experiment_parameters["lgn_parameter_set"][total.index(selection)]
        )
        # run experiment for that set of parameters
        work(experiment_subparameters)
#  ------------------------ 

def extract_subparameters(experiment_parameters, lgn_parameters):
    #pass index instead of lgn_parameters?
    subparameters = {
    "experiment_id": experiment_parameters["experiment_id"],
    "parameter_path": experiment_parameters["parameter_path"],
    "depthmap_path": experiment_parameters["depthmap_path"],
    "autostereogram_path": experiment_parameters["autostereogram_path"],
    "autostereogram_patch": experiment_parameters["autostereogram_patch"],
    "num_filters": experiment_parameters["num_filters"],
    "num_components": experiment_parameters["num_components"],
    "num_patches": experiment_parameters["num_patches"],
    "patch_size": experiment_parameters["patch_size"],
    "lgn_size": experiment_parameters["lgn_size"],
    "lgn_parameters": lgn_parameters,
    "started": None,
    "finished": None,
    "correlation": None,
    # "lgn_dump": "C:\\vscode\\innate-binocular-vision\\innate-binocular-vision\\experiments\\{}\\outputs\\images\\{}\\layers".format(experiment_parameters["experiment_id"],lgn_parameters["name"]),
    "lgn_dump": "C:\\Users\\19404\\innate-binocular-vision\\innate-binocular-vision\\experiments\\{}\\outputs\\images\\{}\\layers".format(experiment_parameters["experiment_id"],lgn_parameters["name"]),
    # "patch_dump": "C:\\vscode\\innate-binocular-vision\\innate-binocular-vision\\experiments\\{}\\outputs\\images\\{}\\patches".format(experiment_parameters["experiment_id"],lgn_parameters["name"]),
    "patch_dump": "CC:\\Users\\19404\\innate-binocular-vision\\innate-binocular-vision\\experiments\\{}\\outputs\\images\\{}\\patches".format(experiment_parameters["experiment_id"],lgn_parameters["name"]),
    # "filter_dump": "C:\\vscode\\innate-binocular-vision\\innate-binocular-vision\\experiments\\{}\\outputs\\images\\{}\\filters".format(experiment_parameters["experiment_id"],lgn_parameters["name"]),
    "filter_dump": "C:\\Users\\19404\\innate-binocular-vision\\innate-binocular-vision\\experiments\\{}\\outputs\\images\\{}\\filters".format(experiment_parameters["experiment_id"],lgn_parameters["name"]),
    # "activity_dump": "C:\\vscode\\innate-binocular-vision\\innate-binocular-vision\\experiments\\{}\\outputs\\images\\{}\\activity".format(experiment_parameters["experiment_id"],lgn_parameters["name"])
    "activity_dump": "C:\\Users\\19404\\innate-binocular-vision\\innate-binocular-vision\\experiments\\{}\\outputs\\images\\{}\\activity".format(experiment_parameters["experiment_id"],lgn_parameters["name"])
    }
    return subparameters


def work(experiment_subparameters):
    # print(experiment_subparameters["depthmap_path"])
    try:
        results = ibv.local_experiment(experiment_subparameters, 5, 5)
    except ValueError as err:
        results = experiment_subparameters
        results["finished"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        # Check error and set correlation
        if str(err) == 'LGN: activity less than low bound':
            results["correlation"] = -1.0

        if str(err) == 'LGN: activity greater than high bound':
            results["correlation"] = 2.0

def run():
    experiment_id = 1
    p = get_parameters(experiment_id)
    run_workload(p)


run()