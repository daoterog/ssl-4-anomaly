"""Script to unzip a file and delete the zip file after extraction."""

import os
import zipfile
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    """Argument parser for the script."""
    parser = ArgumentParser(
        description="Unzip a file and delete the zip file after extraction."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="Input path (directory or file) to examine for zip files.",
    )
    parser.add_argument(
        "--extract_to",
        required=True,
        type=str,
        help="Path to the directory to extract the files to.",
    )

    return parser.parse_args()


def unzip_and_delete(zip_file_path: str, extract_to: str) -> None:
    """Unzip the file and delete the zip file after extraction."""
    # Ensure the target directory exists
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    # Unzip the file
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted all files to: {extract_to}")

    # Delete the zip file after extraction
    os.remove(zip_file_path)
    print(f"Deleted zip file: {zip_file_path}")


def examine_directory(input_directory: str, extract_to: str) -> None:
    """Examine the contents of a directory and unzip files ending in .zip."""
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.endswith(".zip"):
                zip_file_path = os.path.join(root, file)
                unzip_and_delete(zip_file_path, extract_to)


if __name__ == "__main__":
    # Get arguments
    args = parse_args()
    input_path = Path(args.input_path)

    # Check if input_path is a folder or a zip file
    if input_path.is_dir():
        examine_directory(input_path, args.extract_to)
    elif input_path.is_file():
        unzip_and_delete(input_path, args.extract_to)
    else:
        print(f"{input_path} does not exist.")
