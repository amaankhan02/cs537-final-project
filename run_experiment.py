from src.dataset import DatasetName, Dataset
from src.structures import ExperimentResult, Sample, DatasetName, DataClassEncoder
import argparse

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run safety evaluation experiment')
    parser.add_argument('--dataset', type=str, required=True, 
                       choices=[name.value for name in DatasetName],
                       help='Name of dataset to use')
    args = parser.parse_args()
    
    args.dataset = DatasetName(args.dataset)
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    dataset = Dataset(args.dataset)
    


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

[ ] Run Inference:
    1. Iterate through the test dataset. Test dataset contains a) question, b) safe response, c) unsafe response
    2. For each question:
        a) Record start time
        b) Feed the question + prompts into the LLM model
        c) Get the response from the LLM model
        d) Record end time
        e) Run the Judge on the response. So that the judge knows what a safe and unsafe response is, tell it
            here are examples of a safe response: [...] and here are examples of an unsafe response: [...]. 
            Another thing maybe to add the harm_type to the judge to tell it like why this is unsafe? this is only for DoNotAnswer dataset though.
            Note: if safe_responses is empty, then don't put that part and just say here are examples of an unsafe response: [...]
        f) Record this output in a list 
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
                        "time_taken": end_time - start_time
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

'''