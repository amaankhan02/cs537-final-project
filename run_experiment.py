import argparse

from src.bow_model import BowModel
from src.constants import llm_models, experiment_outputs_dir
from src.dataset import Dataset, DatasetName
from src.evaluate import Judge, run_inference_and_eval
from src.structures import DatasetName


def load_llm(model_name: str, system_prompt: str):
    if model_name not in llm_models:
        raise ValueError(f"Model {model_name} not supported")
    return llm_models[model_name](system_prompt)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safety evaluation experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=[name.value for name in DatasetName],
        help="Name of dataset to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=llm_models.keys(),
        help="Name of model to use",
    )
    parser.add_argument(
        "--danger_threshold",
        type=int,
        default=0,
        required=False,
        help="The threshold for the score to be considered dangerous for the BoW model. Defaults to 0.",
    )
    parser.add_argument(
        "--use_bow",
        type=bool,
        default=True,
        required=False,
        help="Whether to use the BoW model. Defaults to True.",
    )
    parser.add_argument(
        "--bow_path",
        type=str,
        required=False,
        help="Path to the bag of words model's dangerous word list. Required if --use_bow is True.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment",
    )
    parser.add_argument(
        "--prompt_injection_path",
        type=str,
        required=True,
        help="Path to the prompt injection file",
    )

    args = parser.parse_args()

    return args


def get_save_filepath(experiment_name: str) -> str:
    import os

    os.makedirs(experiment_outputs_dir, exist_ok=True)
    return f"{experiment_outputs_dir}/{experiment_name}_results.json"

def read_prompt_injections(prompt_injection_path: str) -> str:
    with open(prompt_injection_path, "r") as file:
        return file.read()


if __name__ == "__main__":
    args = parse_args()

    system_prompt = read_prompt_injections(args.prompt_injection_path)
    dataset = Dataset(DatasetName(args.dataset))
    llm_model = load_llm(args.model, system_prompt)
    if args.use_bow:    
        bow_model = BowModel(args.bow_path, args.danger_threshold)
    else:
        bow_model = None
    judge = Judge()

    output_filepath = get_save_filepath(args.experiment_name)

    run_inference_and_eval(
        args.experiment_name, output_filepath, dataset, llm_model, bow_model, judge
    )


"""
Todo and steps: 

[X] Preprocesssing of DoNotAnswer dataset:
    1. Right now, for each query, it has the query + llm_response + is_harmful flag for each of the LLM models' outputs. Instead
    for each query, change it to 
        a) query 
        b) safe_responses: []
        c) unsafe_responses: []
    and you do this based on if the is_harmful flag is true or false for each of the LLM models' outputs. Sometimes though, 
    the safe_responses could be empty. However, the unsafe_responses will never be empty since the parquet file have been filtered
    so that only questions with at least one unsafe response is present. 

[X] Parsing args and loading:
    [X] 1. load in the specific test dataset that we are using here
    [X] 2. load in the specific prompts that we are adding to the prompt-engineering
    [X] 3. take in the model name and instantiate that specific model's object

[X] Add BoW model:
    [X] 1. load in the BoW model
    [X] 2. Add new words to the BoW model

[X] Run Inference:
    1. Iterate through the test dataset. Test dataset contains a) question, b) safe response, c) unsafe response
    2. For each question:
        a) Record start time
        b) Feed the question to the BoW model and determine if its very dangerous or not. If it is, then we don't need to run the LLM.
           If you don't run the LLM, then you still have to record the time, and the response is the default like "I can't answer that"
           and then judge response is "safe" since that's a safe response.
           Else, if it's not very dangerous, then you continue to the next step
        c) Record another start time here so that you can get just the LLM time. 
           Feed the question + prompts into the LLM model
        d) Get the response from the LLM model
        e) Record end time
        f) Run the Judge on the response. So that the judge knows what a safe and unsafe response is, tell it
            here are examples of a safe response: [...] and here are examples of an unsafe response: [...]. 
            Another thing maybe to add the harm_type to the judge to tell it like why this is unsafe? this is only for DoNotAnswer dataset though.
            Note: if safe_responses is empty, then don't put that part and just say here are examples of an unsafe response: [...]
        g) Record this output in a list 
    3. Save this output data list to a JSON file (figure out a name where its the experiment_name + ... + .json). 
        JSON file looks like this:
            {
                "dataset_name": dataset_name, 
                "model_name": model_name, 
                "prompts": [prompt1, prompt2, ...],   # idk if we need this, but doesn't hurt?
                "experiment_name": experiment_name, 
                "y_hat":  # list of the outputs essentially
                [
                    {
                        "question": question, 
                        "llm_response": llm_response, 
                        "judge_response": judge_response (either "safe" or "unsafe"),
                        "total_time_taken": end_time - start_time
                        "bow_time_taken": ...
                        "llm_time_taken": ...
                    },
                    ...
                ]
                "accuracy": percentage of the y_hat values in which the judge response is "safe"
                "average_time_taken": average time taken for the LLM to respond
                "median_time_taken": median time taken for the LLM to respond
            }
            
Future things to think about or implement:
    - Judge Metrics: could we do like a similarity metric between the response and the safe/unsafe response? that way if it is safe,
        how similar is it? or if its unsafe, or unsafe is it? not sure. maybe there's a better way.
    - 


All experiments: 
  > default model, no bow, no prompts -- baseline - do for each of the models and datasets
  > With and without BoW Model
  > No prompts
  > Basic prompt addition
  > Prompt with llmrules.txt
  > Prompt with 'promptinject.json' (rename that file later)
  > Prompt with llmrules.txt + 'promptinject.json'
  > Prompt with harm_types from DoNotAnswer dataset. Add this to the best performing prompt from above?
  > Then alter for each of the datasets
  > Alter for each of the models (gemini, gpt4-mini, llama-3.2-1B-instruct)
"""
