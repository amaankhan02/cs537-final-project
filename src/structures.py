from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
from typing import List, Any, Optional
import json

class DatasetName(str, Enum):
    # subset of the DoNotAnswer dataset containing only the difficult samples where an LLM fails and outputs unsafe response
    DIFFICULT_DONOTANSWER = "DifficultDoNotAnswer"   
    
    # the custom dataset that we create ourselves more specific for smart-homes
    SMARTHOME_CHILD_SAFETY = "SmartHomeChildSafety"  # TODO: think of a better name. This is the custom dataset we made

@dataclass
class Sample:
    """Represents a single sample from the dataset, storing the query, safe responses, unsafe responses, and any other relevant information."""
    query: str
    safe_responses: List[str]
    unsafe_responses: List[str]
    harm_type: Optional[str]

@dataclass 
class Label:
    """Represents the output label after feeding a Sample and evaluating it through the Judge"""
    question: str
    y_hat: str
    is_y_hat_safe: bool
    total_time_taken: float
    bow_time_taken: float
    llm_time_taken: float
    
@dataclass
class ExperimentResult:
    """Represents the result of a single experiment. This is what is saved to the JSON file."""
    dataset_name: str
    model_name: str
    system_prompt: str      # the prompt injections that were used
    experiment_name: str
    accuracy: float
    average_total_time_taken: float
    average_bow_time_taken: float
    average_llm_time_taken: float
    y_hat: List[Label]



# helper class to encode dataclasses to JSON. Passed into json.dump() when saving a list of dataclass objects.
class DataClassEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)