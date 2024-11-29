import re
from collections import Counter
from enum import Enum, auto
from typing import Dict, List
    
class BowModel:
    def __init__(self, dangerous_words_path: str, danger_threshold: int = 0):
        """Initialize a Bag of Words model.

        Args:
            dangerous_words_path (str): Path to a file containing a list of dangerous words split by newlines.
            danger_threshold (int, optional): The threshold for the score to be considered dangerous. Defaults to 0.
        """
        self.dangerous_words = self._load_words(dangerous_words_path)
        self.model = self._create_bag_of_words(self.dangerous_words)
        self.danger_threshold = danger_threshold
        
    def __call__(self, query: str) -> bool:
        """Classify a query as dangerous or not.

        Args:
            query (str): The query to classify.

        Returns:
            bool: Whether the query is classified as dangerous. True if it is. False otherwise.
        """
        bow = self._create_bag_of_words(query)
        return self._get_score(bow) > self.danger_threshold

    def _load_words(self, path: str) -> List[str]:
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()] 

    def _create_bag_of_words(self, text: str) -> Dict[str, int]:
        words = re.findall(r"\w+", text.lower())
        return Counter(words) # TODO: are there ways to make a better bag of words model?
    
    def _get_score(self) -> int:
        return sum(self.model[word] for word in self.dangerous_words)  