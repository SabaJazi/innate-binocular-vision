import argparse
import shutil
from pathlib import Path


def iter_targets(repo_root, include_outputs=True, include_backups=True, include_pycache=True):
    if include_outputs:
        yield repo_root / "workload_results.json"
        yield repo_root / "experiments" / "outputs"

    if include_backups:
        yield from repo_root.glob("*.json~")
        yield from repo_root.glob("*~")

    if include_pycache:
        yield from repo_root.glob("**/__pycache__")


def remove_target(path, dry_run=False):
    if not path.exists():
        return False

    if dry_run:
        print("Would remove: {}".format(path))
        return True

    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()

    print("Removed: {}".format(path))
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean generated experiment files and temporary artifacts."
    )
    parser.add_argument("--no-outputs", action="store_true", help="Keep workload output files")
    parser.add_argument("--no-backups", action="store_true", help="Keep editor backup files like *.json~")
    parser.add_argument("--no-pycache", action="store_true", help="Keep __pycache__ directories")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be removed without deleting anything")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent

    removed_count = 0
    for target in iter_targets(
        repo_root,
        include_outputs=not args.no_outputs,
        include_backups=not args.no_backups,
        include_pycache=not args.no_pycache,
    ):
        if remove_target(target, dry_run=args.dry_run):
            removed_count += 1

    if args.dry_run:
        print("Dry run complete. {} matching item(s) found.".format(removed_count))
    else:
        print("Cleanup complete. {} item(s) removed.".format(removed_count))


if __name__ == "__main__":
    main()