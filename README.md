# cs537-final-project

# run_experiments.py

This script runs a safety evaluation experiment using various models and datasets.

## Usage

```bash
python run_experiments.py [OPTIONS]
```

## Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--experiment_name` | str | Yes | - | Name of the experiment |
| `--dataset` | str | Yes | - | Name of dataset to use |
| `--model` | str | Yes | - | Name of model to use |
| `--danger_threshold` | int | No | 0 | The threshold for the score to be considered dangerous for the BoW model |
| `--use_bow` | bool | No | True | Whether to use the BoW model |
| `--bow_path` | str | No | - | Path to the bag of words model's dangerous word list (Required if `--use_bow` is True) |
| `--prompt_number` | int | Yes | - | Integer of the prompt injection to use |
| `--judge_model_name` | str | Yes | - | Name of the LLM model to use for the judge (Not case-sensitive) |
| `--rules_path` | str | Yes | - | Path to the rules file indicating what are safe and unsafe for children |
| `--examples_path` | str | No | - | Path to the examples file indicating what are safe and unsafe for children (Required for prompt numbers 4 and 5) |

## Detailed Description

### `--experiment_name`
- **Type**: string
- **Required**: Yes
- **Description**: Specifies the name of the experiment being run.

### `--dataset`
- **Type**: string
- **Required**: Yes
- **Choices**: Values from the `DatasetName` enum
- **Description**: Specifies the name of the dataset to use in the experiment.

### `--model`
- **Type**: string
- **Required**: Yes
- **Choices**: `"gpt_mini"`, `"gemini"`, `"llama_mini"`
- **Description**: Specifies the name of the model to use in the experiment.

### `--danger_threshold`
- **Type**: integer
- **Required**: No
- **Default**: 0
- **Description**: Sets the threshold for the score to be considered dangerous in the Bag of Words (BoW) model.

### `--use_bow`
- **Type**: boolean
- **Required**: No
- **Default**: True
- **Description**: Determines whether to use the Bag of Words (BoW) model in the experiment.

### `--bow_path`
- **Type**: string
- **Required**: Only if `--use_bow` is True
- **Description**: Specifies the path to the file containing the list of dangerous words for the BoW model.

### `--prompt_number`
- **Type**: integer
- **Required**: Yes
- **Description**: Specifies the integer corresponding to the prompt injection to be used in the experiment.

### `--judge_model_name`
- **Type**: string
- **Required**: Yes
- **Choices**: `"gpt_mini"`, `"gemini"`, `"llama_mini"`
- **Description**: Specifies the name of the LLM model to be used as the judge (not case-sensitive).

### `--rules_path`
- **Type**: string
- **Required**: Yes
- **Description**: Specifies the path to the file containing rules that define what is safe and unsafe for children.

### `--examples_path`
- **Type**: string
- **Required**: Only for prompt numbers 4 and 5
- **Description**: Specifies the path to the file containing examples of safe and unsafe content for children, used for prompt injection.

## Example Usage

```bash
python run_experiments.py --experiment_name "safety_test_1" --dataset "toxigen" --model "gpt_mini" --prompt_number 3 --judge_model_name "gemini" --rules_path "./rules.txt" --bow_path "./dangerous_words.txt"
```

This command runs an experiment named "safety_test_1" using the "toxigen" dataset and the "gpt_mini" model. It uses prompt injection number 3, with "gemini" as the judge model. The rules for safety are defined in "rules.txt", and the BoW model uses the dangerous words list from "dangerous_words.txt".