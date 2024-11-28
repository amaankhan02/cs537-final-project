import re
from collections import Counter
from enum import Enum, auto
from typing import Dict, List

# TODO: turn this file into a class

dangerous_words = [
    "stove",
    "oven",
    "knife",
    "fire",
    "matches",
    "lighter",
    "gun",
    "poison",
    "bleach",
]
cautionary_words = ["cook", "bake", "cut", "chop", "heat", "boil", "toast", "microwave"]
danger_response = ""  # TODO
DANGER_THRESHOLD = 0
CAUTION_THRESHOLD = 0


class QueryClassification(Enum):
    DANGEROUS = auto()
    CAUTIONARY = auto()
    SAFE = auto()


# very basic bag of words
def create_bag_of_words(text):
    words = re.findall(r"\w+", text.lower())
    return Counter(words)


def get_score(bag_of_words: Dict[str, int], word_list: List[str]) -> int:
    """Get the score of a list of words from the bag of words dictionary

    Args:
        bag_of_words (dict): a dictionary of words and their frequencies
        word_list (list): a list of words that you want to compute the score for in the bag

    Returns:
        int: score
    """
    return sum(bag_of_words[word] for word in word_list)


def classify_query(query) -> QueryClassification:
    bow = create_bag_of_words(query)
    danger_score = get_score(bow, dangerous_words)
    caution_score = get_score(bow, cautionary_words)

    if danger_score > DANGER_THRESHOLD:
        return QueryClassification.DANGEROUS
    elif caution_score > CAUTION_THRESHOLD:
        return QueryClassification.CAUTIONARY
    else:
        return QueryClassification.SAFE


def generate_safe_prompt(original_query: str) -> str:
    """Generate a new safe prompt for the LLM if the query is classified
    as cautionary. That is, given 'query', it should basically return a new query
    that has safe a safe prompt injection within in. It returns the final safe prompt
    to be fed into the LLM

    Args:
        query (str): the query to generate a safe prompt for

    Returns:
        str: the safe prompt
    """
    return NotImplementedError()
