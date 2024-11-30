import argparse
import os
from typing import Optional, Tuple
import json

from src.bow_model import BowModel
from src.constants import (
    PROMPTS_USING_EXAMPLES,
    PROMPTS_USING_RULES,
    experiment_outputs_dir,
)
from src.dataset import Dataset, DatasetName
from src.evaluate import Judge
from src.llm import BaseLLM, create_llm, ModelName
from src.prompt import generate_prompt
from src.structures import DatasetName


def load_rules_file(rules_path: str) -> str:
    with open(rules_path, "r") as file:
        rules = "\n".join([line.strip() for line in file.readlines() if line.strip()])
        return rules


def get_save_filepath(experiment_name: str, dataset_name: str, model: str, use_bow: bool) -> str:
    import os

    os.makedirs(experiment_outputs_dir, exist_ok=True)
    # TODO: add the dataset name to the filepath
    return f"{experiment_outputs_dir}/{experiment_name}_{dataset_name}_{model}_bow-{use_bow}_results.json"


def load_examples_for_prompt(examples_path: str) -> str:
    if examples_path is None or not os.path.exists(examples_path):
        raise ValueError(
            f"Examples path {examples_path} does not exist. It is required for prompt number {args.prompt_injection_number}"
        )

    with open(examples_path, "r") as file:
        data = json.load(file)
    
    formatted_examples = ""
    for item in data["questions"]:
        formatted_examples += f"Question: {item['question']}\n"
        formatted_examples += f"Unsafe Answers: {item['unsafeAnswers']}\n"
        formatted_examples += f"Safe Answer: {item['safeAnswer']}\n\n"
    
    return formatted_examples


def initialize_variables(
    experiment_name: str,
    rules_path: str,
    examples_path: str,
    dataset_name: str,
    model: str,
    bow_path: str,
    use_bow: bool,
    danger_threshold: float,
    judge_model_name: str,
    prompt_injection_number: int,
) -> Tuple[str, Dataset, BaseLLM, Optional[BowModel], Judge]:
    output_filepath: str = get_save_filepath(experiment_name, dataset_name, model, use_bow)
    llm_rules: str = (
        load_rules_file(rules_path)
        if prompt_injection_number in PROMPTS_USING_RULES
        else ""
    )
    examples_for_prompt: str = (
        load_examples_for_prompt(examples_path)
        if prompt_injection_number in PROMPTS_USING_EXAMPLES
        else ""
    )
    system_prompt: str = generate_prompt(
        prompt_injection_number, llm_rules, examples_for_prompt
    )

    dataset = Dataset(DatasetName(dataset_name))
    llm_model: BaseLLM = create_llm(model, system_prompt)
    bow_model: Optional[BowModel] = (
        BowModel(bow_path, danger_threshold) if use_bow else None
    )
    judge: Judge = Judge(ModelName(judge_model_name), llm_rules)

    return output_filepath, dataset, llm_model, bow_model, judge
