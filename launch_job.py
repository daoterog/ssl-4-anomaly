import argparse
import os
from pathlib import Path
from typing import List

import submitit

from parse_arguments import get_training_parser


def build_input_dict(
    arg_parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
    group_name_list: List[str],
) -> dict:
    """Build input dictionary from parser groups."""
    inputs = {}
    for group in arg_parser._action_groups:  # pylint: disable=protected-access
        if group.title not in group_name_list:
            for action in group._group_actions:  # pylint: disable=protected-access
                arg = arguments.get(action.dest)

                if isinstance(arg, list):
                    if len(arg) == 0:
                        continue
                    # Turn into string if necessary
                    arg = " ".join(str(_) for _ in arg)

                if arg is None:
                    # Ensure that None values are included in the job command
                    arg = "None"

                inputs[action.dest] = arg

    del inputs["help"]
    return inputs


def run_training_with_submitit(inputs: dict):
    """Run training job using submitit."""

    data_dir = Path(__file__).parent / "data" / "sewer-ml"
    inputs["train_annots"] = data_dir / "SewerML_Train.csv"
    inputs["valid_annots"] = data_dir / "SewerML_Val.csv"
    inputs["test_annots"] = data_dir / "SewerML_Val.csv"

    inputs["train_data"] = data_dir / "images"
    inputs["valid_data"] = data_dir / "images"
    inputs["test_data"] = data_dir / "images"

    # Define job command to be run with submitit
    code_dir = Path(__file__).parent / "main.py"
    job_command = f"python {code_dir} " + " ".join(
        [f"--{k} {v}" for k, v in inputs.items()]
    )

    # Configure the executor
    executor = submitit.AutoExecutor(folder="logs")
    executor.update_parameters(
        mem_gb=32,  # Adjust memory
        gpus_per_node=1,  # Number of GPUs
        tasks_per_node=1,  # Number of tasks
        cpus_per_task=8,  # Number of CPUs
        timeout_min=60 * 10,  # Max duration of the job
        slurm_partition="partition_name",  # Change this as necessary
    )

    # Submit the job
    job = executor.submit(os.system, job_command)
    print(f"Submitted job {job.job_id} with command: {job_command}")


if __name__ == "__main__":
    # Parse arguments
    parser = get_training_parser()
    args = vars(parser.parse_args())

    # Create job command from parser groups
    input_dict = build_input_dict(
        parser,
        args,
        [
            "data_inputs",
            "outputs",
        ],
    )

    # Run job using submitit
    run_training_with_submitit(input_dict)
