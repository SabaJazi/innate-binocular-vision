import importlib.util
import json
import datetime
import argparse
from pathlib import Path


def load_ibv_module():
    module_path = Path(__file__).resolve().parent / "3-ibv.py"
    spec = importlib.util.spec_from_file_location("ibv", module_path)
    if spec is None or spec.loader is None:
        raise ImportError("Unable to load IBV module from {}".format(module_path))

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ibv = load_ibv_module()


def resolve_input_path(path_value):
    candidate = Path(path_value)
    if candidate.exists():
        return str(candidate)

    repo_candidate = Path(__file__).resolve().parent / candidate.name
    if repo_candidate.exists():
        return str(repo_candidate)

    raise FileNotFoundError(
        "Could not resolve input path '{}' or fallback '{}'".format(path_value, repo_candidate)
    )

def get_parameters(parameters_file):
    with open(parameters_file, "r", encoding="utf-8") as f:
        experiment_parameters = json.load(f)
    return experiment_parameters
 
 #--------------------------------------------------------
def run_workload(experiment_parameters, output_root):
    results = []
    for lgn_parameters in experiment_parameters["lgn_parameter_set"]:
        experiment_subparameters = extract_subparameters(
            experiment_parameters,
            lgn_parameters,
            output_root,
        )
        results.append(work(experiment_subparameters))
    return results
#  ---------------------------------------------------------- 

def extract_subparameters(experiment_parameters, lgn_parameters, output_root):
    #pass index instead of lgn_parameters?
    output_root = Path(output_root)
    image_root = output_root / "images" / lgn_parameters["name"]

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
    "lgn_dump": str(image_root / "layers"),
    "patch_dump": str(image_root / "patches"),
    "filter_dump": str(image_root / "filters"),
    "activity_dump": str(image_root / "activity"),
    }
    return subparameters


def work(experiment_subparameters):
    # print(experiment_subparameters["depthmap_path"])
    try:
        experiment_subparameters = dict(experiment_subparameters)
        experiment_subparameters["depthmap_path"] = resolve_input_path(experiment_subparameters["depthmap_path"])
        experiment_subparameters["autostereogram_path"] = resolve_input_path(experiment_subparameters["autostereogram_path"])
        results = ibv.local_experiment(experiment_subparameters, 5, 5)
    except ValueError as err:
        results = experiment_subparameters
        results["finished"] = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        
        # Check error and set correlation
        if str(err) == 'LGN: activity less than low bound':
            results["correlation"] = -1.0

        if str(err) == 'LGN: activity greater than high bound':
            results["correlation"] = 2.0

    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Run IBV workload from an experiment JSON file")
    parser.add_argument("--experiment-file", default="experiment1.json", help="Path to experiment JSON")
    parser.add_argument("--output", default="workload_results.json", help="Path to output results JSON")
    parser.add_argument(
        "--output-root",
        default=str(Path("experiments") / "outputs"),
        help="Root path for generated image/output folders",
    )
    return parser.parse_args()


def run():
    args = parse_args()

    parameters = get_parameters(args.experiment_file)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results = run_workload(parameters, output_root)
    with open(args.output, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=4)

    print("Saved {} workload result(s) to {}".format(len(results), args.output))


run()