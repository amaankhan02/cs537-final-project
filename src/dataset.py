from datasets import load_dataset
import random
from deprecated import deprecated


@deprecated(reason="We're not using the entire DoNotAnswer dataset anymore.")
class DoNotAnswerDataset:
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


if __name__ == "__main__":
    # testing the code here
    dataset = DoNotAnswerDataset()
    sample = dataset.get_random_sample()
    print(sample)