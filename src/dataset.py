# from datasets import load_dataset
import json
import random
from typing import List, Iterator

from datasets import load_dataset
from deprecated import deprecated

from src.constants import dataset_file_paths
from src.structures import DatasetName, Sample


class Dataset:
    def __init__(self, name: DatasetName):
        """Constructor for the Abstract Base Dataset class.

        Args:
            name (DatasetName): Name of the dataset
            data_path (str): Path to the dataset
        """
        self._name: DatasetName = name
        self._data_path: str = dataset_file_paths[name]
        self._samples: List[Sample] = []
        self._load_data()

    def _load_data(self):
        if self._name == DatasetName.DIFFICULT_DONOTANSWER:
            self._samples = self._load_difficult_donotanswer_data()
        elif self._name == DatasetName.SMARTHOME_CHILD_SAFETY:
            self._samples = self._load_smarthome_child_safety_data()

    def _load_difficult_donotanswer_data(self) -> List[Sample]:
        with open(self._data_path, "r", encoding='utf-8') as f:
            data = json.load(f)

        ret = []
        for entry in data:
            sample = Sample(
                question=entry["question"],
                safe_responses=entry["safe_responses"],
                unsafe_responses=entry["unsafe_responses"],
                harm_type=entry["harm_type"],
            )
            ret.append(sample)
        return ret

    def _load_smarthome_child_safety_data(self) -> List[Sample]:
        with open(self._data_path, "r") as f:
            data = json.load(f)

        ret = []
        for entry in data["questions"]:
            sample = Sample(
                question=entry["question"],
                safe_responses=entry["safeAnswers"],
                unsafe_responses=entry["unsafeAnswers"],
                harm_type=None,
            )
            ret.append(sample)
        return ret

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Sample:
        return self._samples[index]

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)
    
    @property
    def data_path(self) -> str:
        return self._data_path

    @property
    def name(self) -> DatasetName:
        return self._name


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
