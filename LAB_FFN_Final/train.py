"""Compatibility entry point for the report workflow; open the two standalone files to study models."""

import argparse
from FFN_classification_improved import ROOT, OUT, DATA, FFN, clean


def run(task):
    if task == "classification":
        from FFN_classification_improved import run as run_task
    elif task == "regression":
        from FFN_regression_improved import run as run_task
    else:
        raise ValueError(f"Unknown task: {task}")
    return run_task(show=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=["all", "classification", "regression"], default="all"
    )
    args = parser.parse_args()
    for task in ["classification", "regression"] if args.task == "all" else [args.task]:
        run(task)
