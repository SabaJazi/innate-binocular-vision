import os
import ibv
import json
import random
import datetime
import argparse
import numpy as np

def get_parameters(experiment_id):
    parameters_file = "C:\vscode\innate-binocular-vision\innate-binocular-vision\experiment{}.json".format(experiment_id)
    with open(parameters_file, "r") as f:
        experiment_parameters = json.load(f)
    return experiment_parameters
 
 #--------------------------
def run_workload(experiment_parameters):

    started = []
    total = []
    path = r"C:\vscode\innate-binocular-vision\innate-binocular-vision\experiments\{}\outputs\json".format(experiment_parameters["experiment_id"])

    # List files in the output JSON directory
    started = os.listdir(path)
    
    for lgn_parameters in experiment_parameters["lgn_parameter_set"]:
        total.append(lgn_parameters["name"])
    
    while len(set(total)) != len(set(started)):
        diff = list(set(total) - set(started))
        if len(diff) == 0:
            break
        
        selection = random.choice(diff)
        experiment_subparameters = extract_subparameters(
            experiment_parameters,
            experiment_parameters["lgn_parameter_set"][total.index(selection)]
        )
        
        ex = check_log(experiment_subparameters)
        work(ex)
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
    "lgn_dump": "experiments/{}/outputs/images/{}/layers".format(experiment_parameters["experiment_id"],lgn_parameters["name"]),
    "patch_dump": "experiments/{}/outputs/images/{}/patches".format(experiment_parameters["experiment_id"],lgn_parameters["name"]),
    "filter_dump": "experiments/{}/outputs/images/{}/filters".format(experiment_parameters["experiment_id"],lgn_parameters["name"]),
    "activity_dump": "experiments/{}/outputs/images/{}/activity".format(experiment_parameters["experiment_id"],lgn_parameters["name"])
    }
    return subparameters


def check_log(experiment_subparameters):
    experiment_id = experiment_subparameters["experiment_id"]
    lgn_parameter_name = experiment_subparameters["lgn_parameters"]["name"]
    log_path = r"C:\vscode\innate-binocular-vision\innate-binocular-vision\experiments\{}\outputs\json\{}.json".format(experiment_id, lgn_parameter_name)
    
    if not os.path.exists(log_path):
        if experiment_subparameters["started"] is None:
            experiment_subparameters["started"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            print("started")
            print(experiment_subparameters["lgn_parameters"]["name"])
            print(experiment_subparameters)
            
            with open(log_path, "w") as log_file:
                json.dump(experiment_subparameters, log_file, indent=4, separators=(',', ': '))
            
            return experiment_subparameters
    else:
        # Check this logic
        if experiment_subparameters["correlation"] is not None:
            experiment_subparameters["finished"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            
            with open(log_path, "w") as log_file:
                json.dump(experiment_subparameters, log_file, indent=4, separators=(',', ': '))
            
            return experiment_subparameters

def work(experiment_subparameters):
    print(experiment_subparameters["depthmap_path"])
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

    check_log(results)



def run():
    parser = argparse.ArgumentParser(description="Python script to create ibv experiment parameter file")
    parser.add_argument("experiment_id", help="specify experiment id")
    args = parser.parse_args()
    p = get_parameters(args.experiment_id)
    run_workload(p)


run()