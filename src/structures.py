from dataclasses import dataclass, asdict, is_dataclass
from enum import Enum
from typing import List, Any
import json

class DatasetName(str, Enum):
    # subset of the DoNotAnswer dataset containing only the difficult samples where an LLM fails and outputs unsafe response
    DIFFICULT_DONOTANSWER = "DifficultDoNotAnswer"   
    
    # the custom dataset that we create ourselves more specific for smart-homes
    SMARTHOME_CHILD_SAFETY = "SmartHomeChildSafety"  # TODO: think of a better name    

@dataclass
class Sample:
    """Represents a single sample from the dataset, storing the query, safe responses, unsafe responses, and any other relevant information."""
    query: str
    safe_responses: List[str]
    unsafe_responses: List[str]
    # harm_type: str
    # time_taken: float
    
@dataclass
class ExperimentResult:
    """Represents the output label of the LLM, storing the query, response, time taken, judge response, and any other relevant information."""
    # TODO: implement this
    pass

# helper class to encode dataclasses to JSON. Passed into json.dump() when saving a list of dataclass objects.
class DataClassEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        return super().default(obj)