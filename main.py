import re
from collections import Counter
from enum import Enum, auto
from typing import Dict, List

from input_checker import (
    QueryClassification,
    classify_query,
    danger_response,
    generate_safe_prompt,
)
from llm import BaseLLM, LlamaLLM


def process_query(query, llm_model: BaseLLM):
    # main function to process the childs query and then call the LLM if it can

    classification = classify_query(query)
    if classification == QueryClassification.DANGEROUS:
        return danger_response

    if classification == QueryClassification.CAUTIONARY:
        query = generate_safe_prompt(query)

    return llm_model(query)


def main():
    llama_model = LlamaLLM()
    test_queries = [
        "How do I turn on the stove?",
        "How can I toast this bread?",
        "What's the weather like today?",
        "How do I use a knife to cut vegetables?",
        "Can you tell me a bedtime story?",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        response = process_query(query, llama_model)
        # TODO: add response checker here
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
