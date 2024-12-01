import argparse
from src.evaluate import rejudge_responses
from src.structures import DatasetName
from src.utils import load_rules_file
from src.llm import llm_models
from src.constants import dataset_file_paths
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safety evaluation experiment")
    parser.add_argument(
        "--experiment_path",
        type=str,
        required=True,
        help="File path of the experiment results that need reevaluation",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[name.value for name in DatasetName],
        help="Name of dataset to use",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The name of the output file to write to",
    )

    args = parser.parse_args()
    return args    


if __name__ == "__main__":
    args = parse_args()

    llm_rules: str = (
        load_rules_file("data/llmrules.txt")
    )
    rejudge_responses(
        experiment_file=args.experiment_path,
        dataset_file=dataset_file_paths[DatasetName(args.dataset)],
        save_filepath=args.output_file,
        llm_rules=llm_rules
    )
