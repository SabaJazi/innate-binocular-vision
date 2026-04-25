import argparse
import subprocess
import sys
from pathlib import Path

def run_script(args):
    script_path = Path(__file__).resolve().parent / "2-create_exp_local.py"

    command = [
        sys.executable,
        str(script_path),
        "-la", *(str(v) for v in args.lgn_a),
        "-lr", *(str(v) for v in args.lgn_r),
        "-lp", *(str(v) for v in args.lgn_p),
        "-lt", *(str(v) for v in args.lgn_t),
        "--experiment-id", str(args.experiment_id),
        "--depthmap-name", args.depthmap_name,
        "--autostereogram-name", args.autostereogram_name,
        "--autostereogram-patch", str(args.autostereogram_patch),
        "--num-filters", str(args.num_filters),
        "--num-components", str(args.num_components),
        "--num-patches", str(args.num_patches),
        "--patch-size", str(args.patch_size),
        "--lgn-size", str(args.lgn_size),
        "--output", args.output,
    ]

    print("Command:", command)
    subprocess.check_call(command)


def parse_args():
    parser = argparse.ArgumentParser(
        description="CLI helper to generate IBV experiment parameters."
    )
    parser.add_argument("-la", "--lgn_a", nargs=3, type=float, metavar=("min", "max", "step"), required=True)
    parser.add_argument("-lr", "--lgn_r", nargs=3, type=float, metavar=("min", "max", "step"), required=True)
    parser.add_argument("-lp", "--lgn_p", nargs=3, type=float, metavar=("min", "max", "step"), required=True)
    parser.add_argument("-lt", "--lgn_t", nargs=3, type=float, metavar=("min", "max", "step"), required=True)

    parser.add_argument("--experiment-id", type=int, default=1)
    parser.add_argument("--depthmap-name", default="dm.png")
    parser.add_argument("--autostereogram-name", default="autostereogram.png")
    parser.add_argument("--autostereogram-patch", type=int, default=5)
    parser.add_argument("--num-filters", type=int, default=2000)
    parser.add_argument("--num-components", type=int, default=20)
    parser.add_argument("--num-patches", type=int, default=100000)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--lgn-size", type=int, default=64)
    parser.add_argument("--output", default="experiment1.json")

    return parser.parse_args()


if __name__ == "__main__":
    run_script(parse_args())