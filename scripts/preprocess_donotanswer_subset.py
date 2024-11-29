import pandas as pd
import json

def preprocess(output_file_name: str):
    # Read the parquet file
    df = pd.read_parquet('data/donotanswer_subset_difficult.parquet')
    
    # Initialize list to store processed records
    processed_data = []
    
    # Process each row
    for _, row in df.iterrows():
        # Save the harm_type for future reference. maybe it can be helpful for the judge.
        record = {
            'id': row['id'],
            'harm_type': row['types_of_harm'] + ": " + row['specific_harms'],
            'question': row['question'],
            'safe_responses': [],
            'unsafe_responses': []
        }
        
        # Check each LLM's response and categorize based on harmful flag
        llm_names = ['GPT4', 'ChatGPT', 'Claude', 'llama2-7b-chat']
        
        for llm_name in llm_names:
            response = row[f'{llm_name}_response']
            is_harmful = row[f'{llm_name}_harmful']
            
            if is_harmful == 0:
                record['safe_responses'].append(response)
            else:
                record['unsafe_responses'].append(response)
        
        processed_data.append(record)
    
    # Save to JSON file
    with open(f'{output_file_name}.json', 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    save_filename = 'data/processed_donotanswer'
    print("Starting preprocesssing...")
    preprocess(save_filename)
    
    print(f"Preprocessing complete. Results saved to {save_filename}")