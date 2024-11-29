

if __name__ == "__main__":
    pass



'''
Todo and steps: 

Preprocesssing of DoNotAnswer dataset:
1. Right now, for each query, it has the query + llm_response + is_harmful flag for each of the LLM models' outputs. Instead
 for each query, change it to 
    a) query 
    b) safe_responses: []
    c) unsafe_responses: []
and you do this based on if the is_harmful flag is true or false for each of the LLM models' outputs. Sometimes though, 
the safe_responses could be empty. However, the unsafe_responses will never be empty since the parquet file have been filtered
so that only questions with at least one unsafe response is present. 

Parsing args and loading:
1. load in the specific test dataset that we are using here
2. load in the specific prompts that we are adding to the prompt-engineering
3. take in the model name and instantiate that specific model's object

Run Inference:
1. Iterate through the test dataset. Test dataset contains a) question, b) safe response, c) unsafe response
2. For each question:
    a) Feed the question + prompts into the LLM model
    b) Get the response from the LLM model
    c) Run the Judge on the response. So that the judge knows what a safe and unsafe response is, tell it
        here are examples of a safe response: [...] and here are examples of an unsafe response: [...]
        Note: if safe_responses is empty, then don't put that part and just say here are examples of an unsafe response: [...]
    d) Record this output in a list 
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
                      "judge_response": judge_response (either "safe" or "unsafe")
                  },
                  ...
              ]
            "accuracy": percentage of the y_hat values in which the judge response is "safe"
        }
        

'''