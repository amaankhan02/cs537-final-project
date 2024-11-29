# from datasets import load_dataset
import random
from deprecated import deprecated
from typing import List, Dict, Any
from structures import DatasetName, Sample
from constants import dataset_file_paths
import json

class Dataset:
    def __init__(self, name: DatasetName): 
        """Constructor for the Abstract Base Dataset class.

        Args:
            name (DatasetName): Name of the dataset
            data_path (str): Path to the dataset
        """
        self.name: DatasetName = name
        self.data_path: str = dataset_file_paths[name]
        self.samples: List[Sample] = []
        self._load_data()
    
    
    def _load_data(self):
        if self.name == DatasetName.DIFFICULT_DONOTANSWER:
            self.samples = self._load_difficult_donotanswer_data() 
        elif self.name == DatasetName.SMARTHOME_CHILD_SAFETY:
            self.samples = self._load_smarthome_child_safety_data()
            
    def _load_difficult_donotanswer_data(self) -> List[Sample]:
        with open(self.data_path, "r") as f:
            data = json.load(f)
        
        ret = []
        for entry in data:
            sample = Sample(
                query=entry["question"],    
                safe_responses=entry["safe_responses"],
                unsafe_responses=entry["unsafe_responses"],
                harm_type=entry["harm_type"]
            )
            ret.append(sample)
        return ret
    
    def _load_smarthome_child_safety_data(self) -> List[Sample]:
        with open(self.data_path, "r") as f:
            data = json.load(f)
        
        ret = []
        for entry in data['questions']:
            sample = Sample(
                query=entry["question"],    
                safe_responses=entry["safeAnswers"],
                unsafe_responses=entry["unsafeAnswers"],
                harm_type=None
            )
            ret.append(sample)
        return ret  
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
    
    @property
    def samples(self) -> List[Sample]:
        return self.samples
    
    @property
    def name(self) -> DatasetName:
        return self.name
    
    
@deprecated(reason="We're not using the entire DoNotAnswer dataset anymore.")
class OriginalDoNotAnswerDataset:
    def __init__(self, cache_dir="./data/DoNotAnswer"):
        self.ds = load_dataset("LibrAI/do-not-answer", cache_dir=cache_dir)
        self.train_data = self.ds["train"]

    def get_random_sample(self):
        return random.choice(self.train_data)

    def get_sample_by_index(self, index):
        return self.train_data[index]

    def get_samples_by_risk_area(self, risk_area, limit=None):
        filtered = [item for item in self.train_data if item["risk_area"] == risk_area]
        return filtered[:limit] if limit else filtered

    def get_all_risk_areas(self):
        return list(set(item["risk_area"] for item in self.train_data))

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        if index < len(self.train_data):
            return self.train_data[index]
        else:
            raise IndexError("Index out of range")
        