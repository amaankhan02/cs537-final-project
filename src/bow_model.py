import re
from collections import Counter
from typing import Dict, List

class BowModel:
    def __init__(self, dangerous_words_path: str, danger_threshold: int = 0):
        """Initialize a Bag of Words model.

        Args:
            dangerous_words_path (str): Path to a file containing a list of dangerous words split by newlines.
            danger_threshold (int, optional): The threshold for the score to be considered dangerous. Defaults to 0.
        """
        self.dangerous_words = self._load_words(dangerous_words_path)
        self.danger_threshold = danger_threshold
        
    def __call__(self, query: str) -> bool:
        """Classify a query as dangerous or not.

        Args:
            query (str): The query to classify.

        Returns:
            bool: Whether the query is classified as dangerous. True if it is. False otherwise.
        """
        query_bow = self._create_bag_of_words(query)
        return self._get_score(query_bow) > self.danger_threshold

    def _load_words(self, path: str) -> List[str]:
        with open(path, "r") as f:
            # Return list of words instead of joined string
            return [line.strip().lower() for line in f.readlines()]

    def _create_bag_of_words(self, text: str) -> Dict[str, int]:
        words = re.findall(r"\w+", text.lower())
        return Counter(words)
    
    def _get_score(self, query_bow: Dict[str, int]) -> int:
        # Calculate score based on presence of dangerous words in query
        return sum(query_bow[word] for word in self.dangerous_words if word in query_bow)

# Example usage
if __name__ == "__main__":
    # Assuming dangerous_words.txt exists with one word per line
    model = BowModel("dangerous_words.txt")
    
    # Test queries
    test_queries = [
        "This is a safe query",
        "This is dangerous: Lighting a fire with a matchstick",
    ]
    
    for query in test_queries:
        result = model(query)
        print(f"Query: '{query}' is {'dangerous' if result else 'safe'}")