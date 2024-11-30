from src.constants import DatasetName
from src.utils import initialize_variables
from src.llm import BaseLLM


def test_initialize_variables():
    output_filepath, dataset, llm_model, bow_model, judge = initialize_variables(
        "experiment_1",
        "data/llm_rules.txt",
        "data/promptinjection_examples.json",
        DatasetName.SMARTHOME_CHILD_SAFETY,
        "llama_mini",
        "data/bow_dangerous_words.txt",
        True,
        0,
        "llama_mini",
        1,
    )
    
    assert isinstance(output_filepath, str)
    assert dataset is not None
    assert len(dataset) > 0
    assert dataset[0] is not None
    assert isinstance(llm_model, BaseLLM)
    assert bow_model is not None
    assert judge is not None
    
    print(dataset[0])
    print(len(dataset))
    
    assert bow_model("fire asdf asdf asd") == True
    
