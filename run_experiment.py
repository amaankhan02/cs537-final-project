from src.dataset import DatasetName, Dataset
from src.structures import ExperimentResult, Sample, DatasetName, DataClassEncoder
from src.llm import GPTMini, Gemini, LlamaMini
from src.constants import llm_models
import argparse

def load_llm(model_name: str):
    if model_name not in llm_models:
        raise ValueError(f"Model {model_name} not supported")
    return llm_models[model_name]()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run safety evaluation experiment')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=[name.value for name in DatasetName],
                       help='Name of dataset to use')
    parser.add_argument('--model', type=str, required=True, 
                       choices=llm_models.keys(),
                       help='Name of model to use')
    args = parser.parse_args()
    
    args.dataset = DatasetName(args.dataset)
    args.model = load_llm(args.model)
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    dataset = Dataset(args.dataset)
    llm_model = args.model
    
    



'''
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

[ ] Parsing args and loading:
    [ ] 1. load in the specific test dataset that we are using here
    [ ] 2. load in the specific prompts that we are adding to the prompt-engineering
    [ ] 3. take in the model name and instantiate that specific model's object

[ ] Add BoW model:
    [ ] 1. load in the BoW model
    [ ] 2. Add new words to the BoW model

[ ] Run Inference:
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
            
[ ] Future thins to think about or implement:
    - Judge Metrics: could we do like a similarity metric between the response and the safe/unsafe response? that way if it is safe,
        how similar is it? or if its unsafe, or unsafe is it? not sure. maybe there's a better way.
    - 


All experiments: 
  > With and without BoW Model
  > No prompts
  > Basic prompt addition
  > Prompt with llmrules.txt
  > Prompt with 'promptinject.json' (rename that file later)
  > Prompt with llmrules.txt + 'promptinject.json'
  > Then alter for each of the datasets
  > Alter for each of the models (gemini, gpt4-mini, llama-3.2-1B-instruct)
'''